import os

import tensorflow as tf
import keras
import numpy as np
import pickle as pkl

import cv2

class SmplGM() :
    def __init__(self, _means, _covars, _weight,) :
        assert (_covars.shape[0] == _covars.shape[1])
        assert (_covars.shape[0] == _means.shape[0])

        self.D = tf.Variable(float(_covars.shape[0]))

        means = tf.cast(tf.Variable(_means), tf.float32)
        covars = tf.cast(tf.Variable(_covars), tf.float32)
        weight = tf.cast(tf.Variable(_weight), tf.float32)

        self.means = means
        self.det = tf.matrix_determinant(covars)
        self.covars_inv = tf.matrix_inverse(covars)
        self.weight = weight
        self.PI = 3.1415926535898

        #compute coeff
        self.coeff = 0.5*tf.log(
            self.weight *
            1.0 / (
                (2.0 * self.PI) ** (0.5*self.D) *
                tf.sqrt(self.det)
            )
        )

        #print('det ', self.det)

    def get_loss(self, para) :
        loss = tf.matmul(
            tf.matmul(
                tf.reshape(para - self.means, (1, self.D)),
                self.coeff * self.covars_inv
            ),
            tf.reshape(para - self.means, (self.D, 1))
        )
        return loss

class SmplGMM():
    def __init__(self, path_gmm=os.path.join('assets', 'gmm_08.pkl'), flag_print=False):

        # load pkl file
        with open(path_gmm, 'rb') as f:
            src_data = pkl.load(f, ) # encoding="latin1"
        if flag_print:
            print('Load pkl done, the keys of the dict are', src_data.keys())

        assert (src_data['weights'].shape[0] == src_data['covars'].shape[0] and
                src_data['weights'].shape[0] == src_data['means'].shape[0])

        # nums of gaussion model
        self.num_gm = src_data['weights'].shape[0]

        assert (src_data['covars'].shape[1] == src_data['covars'].shape[2] and
                src_data['covars'].shape[1] == src_data['means'].shape[1])

        # dimensions of gaussian model
        self.D = src_data['covars'].shape[1]

        # pre-precoss each gaussain model
        self.gms = self.distribute_to_gm(src_data)

        if flag_print:
            print('Load and pre-process done. Pre-prcossing includes get det and inv of each covars.')
            print('The numbers of GMs: ', self.num_gm)
            print('The dimension of each GM: ', self.D)

    def get_loss(self, para):
        para = para[3:72]

        loss = tf.zeros([self.num_gm], dtype=tf.float32)

        loss_list = []
        for i in range(0, self.num_gm):
            loss_list.append(self.gms[i].get_loss(para))

        return tf.reduce_min(loss_list)

    @staticmethod
    def distribute_to_gm(src_data):
        gms = []
        for i in range(src_data['weights'].shape[0]):
            gm = SmplGM(src_data['means'][i],
                        src_data['covars'][i],
                        src_data['weights'][i],
                        )
            gms.append(gm)
        return gms

def rodrigues_batch(rvecs):
    """
    Convert a batch of axis-angle rotations in rotation vector form shaped
    (batch, 3) to a batch of rotation matrices shaped (batch, 3, 3).
    See
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    batch_size = tf.shape(rvecs)[0]
    tf.assert_equal(tf.shape(rvecs)[1], 3)

    thetas = tf.norm(rvecs, axis=1, keepdims=True)
    is_zero = tf.equal(tf.squeeze(thetas), 0.0)
    u = rvecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = tf.zeros([batch_size])  # for broadcasting
    Ks_1 = tf.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = tf.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = tf.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = tf.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    Rs = tf.eye(3, batch_shape=[batch_size]) + \
         tf.sin(thetas)[..., tf.newaxis] * Ks + \
         (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)

    # Avoid returning NaNs where division by zero happened
    return tf.where(is_zero,
                    tf.eye(3, batch_shape=[batch_size]), Rs)
def rodrigues_m(r):
  """
  Rodrigues' rotation formula that turns axis-angle tensor into rotation
  matrix in a batch-ed manner.

  Parameter:
  ----------
  r: Axis-angle rotation tensor of shape [batch_size, 1, 3].

  Return:
  -------
  Rotation matrix of shape [batch_size, 3, 3].

  """
  theta = tf.norm(r + tf.random_normal(r.shape, 0, 1e-8, dtype=tf.float32), axis=(1, 2), keepdims=True)
  # avoid divide by zero
  r_hat = r / theta
  cos = tf.cos(theta)
  z_stick = tf.zeros(theta.get_shape().as_list()[0], dtype=tf.float32)
  m = tf.stack(
    (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
     r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
     -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick),
      axis=1)
  m = tf.reshape(m, (-1, 3, 3))
  i_cube = tf.expand_dims(tf.eye(3, dtype=tf.float32), axis=0) + tf.zeros((theta.get_shape().as_list()[0], 3, 3), dtype=tf.float32)
  A = tf.transpose(r_hat, (0, 2, 1))
  B = r_hat
  dot = tf.matmul(A, B)
  R = cos * i_cube + (1 - cos) * dot + tf.sin(theta) * m

  return R

def rodrigues_v(R):
    theta = tf.math.acos((tf.trace(R)-1) * 0.5)
    #if theta == 0.0
    sin = tf.math.sin(theta)
    sin_reciprocal = tf.reciprocal(sin)
    Right = ( R - tf.transpose(R, (0,2,1)) ) * 0.5
    l = tf.einsum('ijk,i->ijk', Right, sin_reciprocal)
    #r = tf.zeros([R.shape[0], 3])
    a = (l[:,2,1] - l[:,1,2])*0.5
    b = (l[:,0,2] - l[:,2,0])*0.5
    c = (l[:,1,0] - l[:,0,1])*0.5
    r = tf.stack([a,b,c], axis=1)

    r = tf.einsum("ij,i->ij", r, theta)

    return r


# For testing only
if __name__ == '__main__':
    # np.random.seed(100)

    rvecs = np.random.randn(2, 3).astype(np.float32)
    print (rvecs)

    rvecs_tf = tf.constant(rvecs)
    # Rs = rodrigues_batch(tf.reshape(rvecs_tf, [-1, 3]))
    Rs = rodrigues_m(tf.reshape(rvecs_tf, [rvecs_tf.shape[0], -1, 3]))
    r = rodrigues_v(Rs)
    #R_ = rodrigues_m(tf.reshape(r, [r.shape[0], -1, 3]))
    with tf.Session() as sess:
        print("TensorFlow: ")

        print(sess.run(Rs))
        #print(sess.run(theta))
        print(sess.run(r))

        #print(sess.run(R_))

    print("\nOpenCV: ")
    for rvec in np.reshape(rvecs, [-1, 3]):
        mat, _ = cv2.Rodrigues(rvec)
        v, _ = cv2.Rodrigues(mat)
        print(mat)
        print(v)


