import pickle

import numpy as np
import torch
from utils.general import numpy2tensor
from torch.nn import Module


class SMPLModel(Module):

    def __init__(self,
                 device=None,
                 model_path='./model.pkl',
                 use_posematrix=False):

        super(SMPLModel, self).__init__()
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='iso-8859-1')
        self.J_regressor = \
            numpy2tensor(np.array(params['J_regressor'].todense())
                         if type(params['J_regressor']) is not np.ndarray
                         else params['J_regressor'])

        self.weights = numpy2tensor(np.array(params['weights']))
        self.posedirs = numpy2tensor(np.array(params['posedirs']))
        self.v_template = numpy2tensor(np.array(params['v_template']))
        self.shapedirs = numpy2tensor(np.array(params['shapedirs']))
        self.kintree_table = params['kintree_table']
        self.faces = params['f']
        self.device = device if device is not None else torch.device('cpu')
        for name in [
                'J_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs'
        ]:
            _tensor = getattr(self, name)
            # print(' Tensor {} shape: '.format(name), _tensor.shape)
            setattr(self, name, _tensor.to(device))

        self.use_posematrix = use_posematrix

    @staticmethod
    def rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.
        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
        Return:
        -------
        Rotation matrix of shape [batch_size * angle_num, 3, 3].
        """
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
             -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick),
            dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = \
            (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) +
             torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
        Parameter:
        ---------
        x: Tensor to be appended.
        Return:
        ------
        Tensor after appending of shape [4,4]
        """
        ones = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32) \
            .expand(x.shape[0], -1, -1).to(x.device)
        ret = torch.cat((x, ones), dim=1)
        return ret

    @staticmethod
    def pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]
        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.
        """
        zeros43 = torch.zeros((x.shape[0], x.shape[1], 4, 3),
                              dtype=torch.float32).to(x.device)
        ret = torch.cat((zeros43, x), dim=3)
        return ret

    def write_obj(self, verts, file_name, write_face=True):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            if write_face:
                for f in self.faces + 1:
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def forward(self, betas, pose, trans, displacements, simplify=False):
        """Construct a compute graph that takes in parameters and outputs a
        tensor as model vertices. Face indices are also returned as a numpy
        ndarray.

        20190128: Add batch support.
        Parameters:
        ---------
        pose: Also known as 'theta', an [N, 24, 3] tensor indicating
        child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.
        betas: Parameter for model shape. A tensor of shape [N, 10] as
        coefficients of
        PCA components. Only 10 components were released by SMPL author.
        trans: Global translation tensor of shape [N, 3].
        Return:
        ------
        A 3-D tensor of [N * 6890 * 3] for vertices,
        and the corresponding [N * 19 * 3] joint positions.
        """
        batch_num = betas.shape[0]
        id_to_col = {
            self.kintree_table[1, i]: i
            for i in range(self.kintree_table.shape[1])
        }
        parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        v_shaped = torch.tensordot(
            betas, self.shapedirs, dims=([1], [2])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)

        if self.use_posematrix:
            R_cube_big = pose
        else:
            R_cube_big = \
                self.rodrigues(pose.view(-1, 1, 3)) \
                    .reshape(batch_num, -1, 3, 3)

        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[:, 1:, :, :]
            I_cube = \
                (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) +
                 torch.zeros((batch_num, R_cube.shape[1], 3, 3),
                             dtype=torch.float32)).to(self.device)
            lrotmin = \
                (R_cube - I_cube).reshape(batch_num, -1, 1).squeeze(dim=2)
            v_posed = v_shaped + torch.tensordot(
                lrotmin, self.posedirs, dims=([1], [2]))

            rescale = True
            if rescale is True:
                body_height = (v_posed[:, 2802, 1] + v_posed[:, 6262, 1]) - \
                              (v_posed[:, 2237, 1] + v_posed[:, 6728, 1])
                scale = 1.66 / body_height  # np.reshape(, (-1, 1, 1))
                v_posed = v_posed * torch.reshape(scale, [-1, 1, 1])

            self.v_shaped = v_posed.clone()
            v_posed = v_posed + displacements
            self.v_shaped_personal = v_posed.clone()
            self.v_offsets = \
                self.v_template * torch.reshape(scale, [-1, 1, 1]) \
                + displacements

        results = []
        results.append(
            self.with_zeros(
                torch.cat(
                    (R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))),
                    dim=2)))
        for i in range(1, self.kintree_table.shape[1]):
            results.append(
                torch.matmul(
                    results[parent[i]],
                    self.with_zeros(
                        torch.cat(
                            (R_cube_big[:, i],
                             torch.reshape(J[:, i, :] - J[:, parent[i], :],
                                           (-1, 3, 1))),
                            dim=2))))

        stacked = torch.stack(results, dim=1)
        results = \
            stacked - \
            self.pack(
                torch.matmul(
                    stacked,
                    torch.reshape(
                        torch.cat((J,
                                   torch.zeros((batch_num, 24, 1),
                                               dtype=torch.float32)
                                   .to(self.device)),
                                  dim=2),
                        (batch_num, 24, 4, 1)
                    )
                )
            )
        # Restart from here
        T = torch.tensordot(
            results, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
        rest_shape_h = \
            torch.cat((v_posed,
                       torch.ones((batch_num, v_posed.shape[1], 1),
                                  dtype=torch.float32).to(self.device)
                       ), dim=2)
        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]
        result = v + torch.reshape(trans, (batch_num, 1, 3))

        # 3D joint locations
        # print(result.shape)
        # print(self.joint_regressor.shape)
        # joints =  torch.tensordot(result, self.joint_regressor,
        #                           dims=([1], [1])).transpose(1, 2)
        return result  # , joints
