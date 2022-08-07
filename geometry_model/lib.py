import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle as pkl
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    softmax_rgb_blend
)
from pytorch3d.ops import interpolate_face_attributes


def write_obj(vs, vt=None, fs=None, ft=None, path=None, write_mtl=False):
    name = path.split('/')[-1].split('.')[0]
    with open(path, 'w') as fp:
        if write_mtl == True:
            fp.write('mtllib {}.obj.mtl\n'.format(name))
            fp.write('usemtl material_0\n')

        if vs.shape[1] == 3:
            [fp.write('v %f %f %f\n' % (v[0], v[1], v[2])) for v in vs]
        elif vs.shape[1] == 6:
            [fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], v[3], v[4], v[5])) for v in vs]

        if vt is not None:
            [fp.write('vt %f %f \n' % (t[0], t[1])) for t in vt]
        # for f, t in fs + 1, ft + 1:
        #     fp.write('f %d/%d %d/%d %d/%d\n' % (f[0], t[0], f[1], t[1], f[2], t[2]))
        if fs is not None and ft is not None:
            [fp.write('f %d/%d %d/%d %d/%d\n' % (
                fs[i][0] + 1, ft[i][0] + 1, fs[i][1] + 1, ft[i][1] + 1, fs[i][2] + 1, ft[i][2] + 1)) for i in range(fs.shape[0])]
        elif fs is not None:
            [fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1)) for f in fs]

    if write_mtl == True:
        with open(path + '.mtl', 'w') as fp:  # 'examples/our_models/{}.obj.mtl'.format(name),
            fp.write('newmtl material_0\n')
            fp.write('map_Kd texture.png\n')


def schmidt_orthogonalization(pose):
    pose = pose.view([-1, 3, 3])
    pose_0 = F.normalize(pose[:, 0, :], dim=-1)
    # pose_1 = F.normalize(pose[:,1,:], dim=-1)
    # pose_2 = F.normalize(pose[:,2,:], dim=-1)
    # a = pose_0
    pose_1 = pose[:, 1, :] - torch.sum(F.normalize(pose[:, 1, :], dim=-1) * pose_0, dim=-1, keepdim=True) * pose_0
    pose_1 = F.normalize(pose_1, dim=-1)
    pose_2 = pose[:, 2, :] - torch.sum(F.normalize(pose[:, 2, :], dim=-1) * F.normalize(pose_0, dim=-1), dim=-1,
                                       keepdim=True) * pose_0 \
             - torch.sum(F.normalize(pose[:, 2, :], dim=-1) * F.normalize(pose_1, dim=-1), dim=-1,
                         keepdim=True) * pose_1
    pose_2 = F.normalize(pose_2, dim=-1)
    s_pose = torch.stack([pose_0, pose_1, pose_2], dim=-2)

    # pose_2 = F.normalize(pose[:,1,:] - torch.matmul(pose[:,1,:], pose[:,0,:]) * pose_1, dim=-1)

    # m = torch.matmul(s_pose, s_pose.transpose(1, 2))
    # m_1 =
    # a = m.detach().cpu().numpy()

    s_pose = s_pose.view([-1, 24, 3, 3])
    return s_pose

def cal_perspective_matrix(f=[1080,1080], c=[540,540], w=1080, h=1080, near=0.1, far=10.):
    f = 0.5 * (f[0] + f[1])
    pixel_center_offset = 0.5
    right = (w - (c[0] + pixel_center_offset)) * (near / f)
    left = -(c[0] + pixel_center_offset) * (near / f)
    top = (c[1] + pixel_center_offset) * (near / f)
    bottom = -(h - c[1] + pixel_center_offset) * (near / f)

    elements = [
        [2. * near / (right - left), 0., (right + left) / (right - left), 0.],
        [0., 2. * near / (top - bottom), (top + bottom) / (top - bottom), 0.],
        [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
        [0., 0., -1., 0.]
    ]

    return torch.transpose(torch.from_numpy(np.array(elements)), 0, 1).type(torch.float32)

def get_f_ft_vt(isupsample):
    if isupsample == False:
        f = np.loadtxt('./assets/smpl_f_ft_vt/smpl_f.txt')
        ft = np.loadtxt('./assets/smpl_f_ft_vt/smpl_ft.txt')
        vt = np.loadtxt('./assets/smpl_f_ft_vt/smpl_vt.txt')
    else:
        f = np.loadtxt('./assets/upsmpl_f_ft_vt/smpl_f.txt')
        ft = np.loadtxt('./assets/upsmpl_f_ft_vt/smpl_ft.txt')
        vt = np.loadtxt('./assets/upsmpl_f_ft_vt/smpl_vt.txt')
    return f, ft, vt

def project(v, perspective_matrix):
    v_h = torch.cat([v, torch.ones_like(v[:, :, -1:])], dim=2)
    perspective_vertices = torch.matmul(v_h, perspective_matrix)
    perspective_vertices = perspective_vertices / perspective_vertices[:,:,-1:]
    # l = perspective_vertices.detach().cpu().numpy()
    return perspective_vertices

def get_regressor():
    with open('./assets/J_regressor.pkl','rb') as file:
        body25_reg = pkl.load(file, encoding='iso-8859-1')

    coo = body25_reg.tocoo()
    indices = np.array([coo.row, coo.col]) # .transpose()
    body25_reg_tensor = torch.sparse_coo_tensor( torch.from_numpy(indices), torch.from_numpy(coo.data), coo.shape)
    body25_reg_tensor = torch.transpose(body25_reg_tensor,0,1)

    with open('./assets/face_regressor.pkl', 'rb') as file:
        face_reg = pkl.load(file, encoding='iso-8859-1')
    coo = face_reg.tocoo()
    indices = np.array([coo.row, coo.col]) # .transpose()
    face_reg_tensor = torch.sparse_coo_tensor( torch.from_numpy(indices), torch.from_numpy(coo.data), coo.shape)
    face_reg_tensor = torch.transpose(face_reg_tensor,0, 1)

    body25_reg_tensor = body25_reg_tensor.to_dense()
    face_reg_tensor = face_reg_tensor.to_dense()

    return body25_reg_tensor, face_reg_tensor

def get_upsample_regressor():
    body25_reg_tensor = np.load('./assets/up_body25_regressor.npy')
    face_reg_tensor = np.load('./assets/up_face_regressor.npy')

    return torch.from_numpy(body25_reg_tensor).cuda(), torch.from_numpy(face_reg_tensor).cuda()
    pass

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
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R

# under writing
# results with some problems
def rodrigues_v(R):
    trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) #(torch.sum(torch.diag(R)) - 1) * 0.5
    # theta = torch.acos((trace-1)*0.5)
    radian =  (trace - 1) * 0.5
    radian = torch.clamp(radian, -1 + 0.0001, 1 - 0.0001) # for differentiable
    theta = torch.acos(radian)
    # if theta == 0.0
    sin = torch.sin(theta) + 1e-8
    sin_reciprocal = torch.reciprocal(sin)
    Right = (R - R.permute(0, 2, 1)) * 0.5
    l = torch.einsum('ijk,i->ijk', Right, sin_reciprocal)
    # r = tf.zeros([R.shape[0], 3])
    a = (l[:, 2, 1] - l[:, 1, 2]) * 0.5
    b = (l[:, 0, 2] - l[:, 2, 0]) * 0.5
    c = (l[:, 1, 0] - l[:, 0, 1]) * 0.5
    n = torch.stack([a, b, c], dim=1)

    r = torch.einsum("ij,i->ij", n, theta)

    return r, n

def calcuate_normal(v, fs, isupsample=False):
    batch_size = v.shape[0]
    f = fs.reshape([-1])
    f_v = v.index_select(dim=1, index=f)
    f_v = f_v.reshape([batch_size, fs.shape[1], 3, 3])
    ab = f_v[:, :, 1, :] - f_v[:, :, 0, :]
    ac = f_v[:, :, 2, :] - f_v[:, :, 0, :]
    # f_v
    face_normal = torch.cross(ab, ac, dim=2)
    face_normal = F.normalize(face_normal, dim=2)

    if isupsample == False:
        adjacency_face_set = np.load('./assets/adjacency_face_set.npy')
    else:
        adjacency_face_set = np.load('./assets/upsample_adjacency_face_set.npy')

    _face_normal = torch.cat([face_normal, torch.zeros([batch_size, 1, 3]).cuda()], dim=1)
    indices = torch.from_numpy(adjacency_face_set[:, :-1].astype('int64').reshape((-1,))).cuda()
    weights = torch.from_numpy(np.expand_dims(adjacency_face_set[:, -1:].astype('float32'), axis=0)).cuda()
    # a = indices.detach().cpu().numpy()
    weights = torch.reciprocal(weights).repeat((batch_size, 1, 3))

    _face_normals = _face_normal.index_select(index=indices, dim=1)
    _face_normals = _face_normals.reshape((batch_size, v.shape[1], -1, 3))
    vertex_normal = torch.sum(_face_normals, dim=2)

    vertex_normal = vertex_normal * weights

    vertex_normal = F.normalize(vertex_normal, dim=2)

    # --------------------------
    face_gravity_vertex = torch.sum(f_v, dim=2)

    return face_normal, vertex_normal, face_gravity_vertex

def calcuate_edge_length(v, fs):
    batch_size = v.shape[0]
    f = fs.reshape([-1])
    f_v = v.index_select(dim=1, index=f)
    f_v = f_v.reshape([batch_size, fs.shape[1], 3, 3])
    ab = f_v[:, :, 1, :] - f_v[:, :, 0, :]
    ac = f_v[:, :, 2, :] - f_v[:, :, 0, :]
    bc = f_v[:, :, 1, :] - f_v[:, :, 2, :]

    return torch.mean(torch.square(ab)) + torch.mean(torch.square(ac)) + torch.mean(torch.square(bc))


def NormalCalcuate(meshes, fragments):
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    # pixel_coords = interpolate_face_attributes(
    #     fragments.pix_to_face, fragments.bary_coords, faces_verts
    # )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals #  torch.ones_like()
    )
    return pixel_normals

class NormalShader(nn.Module):
    def __init__(self, device="cuda", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        # texels = meshes.sample_textures(fragments)
        normals = NormalCalcuate(meshes, fragments)
        images = softmax_rgb_blend(normals, fragments, blend_params)[:,:,:,:3]

        images = F.normalize(images, dim=3)
        return images
