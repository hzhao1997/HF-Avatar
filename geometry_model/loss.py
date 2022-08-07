import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import os
import torch
import numpy as np
import pickle
from utils.general import tensor2numpy, numpy2tensor
from geometry_model.lib import write_obj

class LaplacianLoss(nn.Module):
    def __init__(self, device, isupsample=False, stage='1'):
        super(LaplacianLoss, self).__init__()

        self.isupsample = isupsample

        self.device = device
        self.stage = stage


        self.v_ids = self.get_bodypart_vertex_ids()
        self.vertex_label = self.get_vertex_label(self.isupsample)
        self.get_all_parts()

        self.adjacency_set = np.load('./assets/adjacency_set.npy') if isupsample == False \
            else np.load('./assets/upsample_adjacency_set.npy')
        self.laplace_w = numpy2tensor(self.regularize_laplace()[:, np.newaxis]).to(self.device)

    def get_all_parts(self):

        if self.isupsample == False:
            with open('./assets/neutral_smpl.pkl', 'rb') as file:
                params = pkl.load(file, encoding='iso-8859-1')
            v_template = params['v_template']
            self.armpits_ids = self.get_parts(v_template, 'armpits')
        else:
            with open('./assets/upsample_neutral_smpl.pkl', 'rb') as file:
                params = pkl.load(file, encoding='iso-8859-1')
            v_template = params['v_template']
            self.armpits_and_crotch_ids = self.get_parts(v_template, 'armpits_and_crotch')
            self.nose_ids = self.get_parts(v_template, 'nose')
            self.chest_ids = self.get_parts(v_template, 'chest')
            self.waist_ids = self.get_parts(v_template, 'waist')

    def get_parts(self, v_template, part_name='armpits_and_crotch'):
        # ---------------
        # data = np.loadtxt('./assets/armpits_and _crotch.txt', dtype='str')
        with open('./assets/selected_obj/{}.obj'.format(part_name), 'r') as file:
            data = file.readlines()
        # armpits_and_crotchnp_parts = np.read()
        v = []
        for line in data:
            odom = line.split()
            # a = len(odom)
            if len(odom) > 0 and odom[0] == 'v':
                v.append(
                    [float(odom[1]), float(odom[2]), float(odom[3]), float(odom[4]), float(odom[5]), float(odom[6])])
        v = np.array(v)
        index = []
        for idx in range(v.shape[0]):
            if v[idx][3] == 1.0 and v[idx][4] == 0.0 and v[idx][5] == 0.0:
                index.append(idx)

        write_obj(vs=np.take(v_template, index, axis=0), fs=None, path='./results/{}.obj'.format(part_name))
        # write_obj(params['v_template'], params['f'], './results/v_template.obj')
        return np.array(index)

    def get_bodypart_vertex_ids(self):
        path = './assets/bodyparts.pkl' if self.isupsample == False \
            else './assets/upsample_bodyparts.pkl'

        with open(path, 'rb') as fp:
            _cache = pkl.load(fp, encoding='iso-8859-1')

        return _cache

    @staticmethod
    def get_vertex_label(isupsample):
        path = './assets/vertex_label.pkl' if isupsample == False \
            else './assets/upsample_vertex_label.pkl'
        with open(path, 'rb') as fp:
            _cache = pkl.load(fp, encoding='iso-8859-1')

        return _cache

    def regularize_laplace(self):
        if self.isupsample==False:
            reg = np.ones(6890)
        else:
            reg= np.ones(27554)
        v_ids = self.v_ids # self.get_bodypart_vertex_ids()
        vertex_label = self.vertex_label # self.get_vertex_label()

        if self.isupsample == False:
            scale = 3
            reg[v_ids['face']] = 8. * 2
            reg[v_ids['hand_l']] = 5. * scale
            reg[v_ids['hand_r']] = 5. * scale
            reg[v_ids['fingers_l']] = 8. * scale
            reg[v_ids['fingers_r']] = 8. * scale
            reg[v_ids['foot_l']] = 5. * scale
            reg[v_ids['foot_r']] = 5. * scale
            reg[v_ids['toes_l']] = 8. * scale # 8.
            reg[v_ids['toes_r']] = 8. * scale # 8.
            reg[v_ids['ear_l']] = 10. * scale
            reg[v_ids['ear_r']] = 10. * scale
            reg[self.armpits_ids] = 10

        else:
            reg[vertex_label['left_arm']] = 800  # 150
            reg[vertex_label['right_arm']] = 800  # 150
            reg[vertex_label['left_hand']] = 10
            reg[vertex_label['right_hand']] = 10

            reg[vertex_label['forward_body']] = 50
            reg[vertex_label['backward_body']] = 10
            reg[vertex_label['left_leg']] = 10
            reg[vertex_label['right_leg']] = 10
            reg[vertex_label['left_head']] = 10
            reg[vertex_label['right_head']] = 10
            reg[vertex_label['left_foot']] = 10
            reg[vertex_label['right_foot']] = 10

            # ---------------------------
            reg[v_ids['face']] = 1600 # 3200.
            reg[v_ids['hand_l']] = 500.
            reg[v_ids['hand_r']] = 500.
            reg[v_ids['fingers_l']] = 800.
            reg[v_ids['fingers_r']] = 800.
            reg[v_ids['foot_l']] = 500.
            reg[v_ids['foot_r']] = 500.
            reg[v_ids['toes_l']] = 800.  # 8.
            reg[v_ids['toes_r']] = 800.  # 8.
            reg[v_ids['ear_l']] = 1000.
            reg[v_ids['ear_r']] = 1000.

            reg[self.nose_ids] = 2 # 10
            reg[self.chest_ids] = 400
            reg[self.waist_ids] = 400
            reg[self.armpits_and_crotch_ids] = 800

        return reg

    def laplace_coord(self, v, adjacency_set):
        # m = v.detach().cpu().numpy()

        vertex = torch.cat([v, torch.zeros([1, 3]).to(self.device)], dim=0)
        indices = numpy2tensor(adjacency_set[:, :9].reshape((-1,)), np.int64).to(self.device)
        weights = numpy2tensor(adjacency_set[:, -1:], np.float32).to(self.device)
        # a = indices.detach().cpu().numpy()
        weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))

        # i = torch.from_numpy(np.array(-1).astype('int64')).to(device)
        vertices = vertex.index_select(index=indices, dim=0)  # .reshape([6890,-1,3]) indices
        vertices = vertices.reshape((v.shape[0], -1, 3)) # 6890
        # b = vertices.detach().cpu().numpy()
        laplace = torch.sum(vertices, dim=1)
        # laplace_v = torch.mul(laplace, weights)
        # c = laplace.detach().cpu().numpy()


        laplace_vertex = v - torch.mul(laplace, weights)

        return laplace_vertex

    def compute_laplacian_diff(self, v_1, v_2):
        lap1 = self.laplace_coord(v_1, self.adjacency_set)
        lap2 = self.laplace_coord(v_2, self.adjacency_set)
        # save_obj(lap1.detach().cpu().numpy(), f=self.f, path='results/lap1.obj')
        # save_obj(lap2.detach().cpu().numpy(), f=self.f, path='results/lap2.obj')

        laplace_loss = torch.square(lap1 - lap2) # L2 distance
        laplace_loss = torch.mean(laplace_loss * self.laplace_w)
        return laplace_loss

    def forward(self, v_1, v_2):
        return self.compute_laplacian_diff(v_1, v_2)

class SymmetryLoss(nn.Module):
    def __init__(self):
        super(SymmetryLoss, self).__init__()
        self.idx = numpy2tensor(np.load('./assets/vert_sym_idxs.npy')).cuda()
        self.symmetry_w = numpy2tensor(self.regularize_symmetry()[:,np.newaxis]).cuda()

    def get_bodypart_vertex_ids(self):
        with open('./assets/bodyparts.pkl', 'rb') as fp:
            _cache = pkl.load(fp, encoding='iso-8859-1')
        return _cache

    def regularize_symmetry(self):
        reg = np.ones(6890)
        v_ids = self.get_bodypart_vertex_ids()

        reg[v_ids['face']] = 10.
        reg[v_ids['hand_l']] = 10.
        reg[v_ids['hand_r']] = 10.
        reg[v_ids['foot_l']] = 10. # 10.
        reg[v_ids['foot_r']] = 10. # 10.
        reg[v_ids['ear_l']] = 5.
        reg[v_ids['ear_r']] = 5.

        return reg

    def get_v_mirror(self, v):
        v_mirror = v.index_select(index=self.idx, dim=0) * torch.from_numpy(np.array([-1, 1, 1])).cuda()
        return v_mirror

    def forward(self, v):
        v_mirror = v.index_select(index=self.idx, dim=0) * torch.from_numpy(np.array([-1, 1, 1])).cuda()

        symmetry_loss = torch.square(v - v_mirror)
        symmetry_loss = torch.mean(symmetry_loss * self.symmetry_w)
        return symmetry_loss

class TemporalLoss(nn.Module):
    def __init__(self):
        super(TemporalLoss, self).__init__()
        pass

    def forward(self, x):
        forward_x = torch.roll(x, shifts=(1), dims=(0))
        backward_x = torch.roll(x, shifts=(-1), dims=(0))
        bias =  (x - (forward_x + backward_x) / 2)[1:-1]
        loss = torch.mean(torch.square(bias))
        return loss

class EDGELoss(nn.Module):
    def __init__(self, device, isupsample=False):
        super(EDGELoss, self).__init__()

        face_path = './assets/smpl_f_ft_vt/smpl_f.txt' if isupsample == False \
            else './assets/upsmpl_f_ft_vt/smpl_f.txt'
        self.f = np.loadtxt(face_path)

        self.fs = numpy2tensor(np.expand_dims(self.f, axis=0), np.long).repeat([1, 1, 1]).to(device)


    def forward(self, v):
        batch_size = v.shape[0]

        f = self.fs.reshape([-1])
        f_v = v.index_select(dim=1, index=f)
        f_v = f_v.reshape([batch_size, self.fs.shape[1], 3, 3])
        ab = f_v[:, :, 1, :] - f_v[:, :, 0, :]
        # m, n = torch.max(ab), torch.min(ab)
        ac = f_v[:, :, 2, :] - f_v[:, :, 0, :]
        bc = f_v[:, :, 1, :] - f_v[:, :, 2, :]

        # return torch.mean(torch.square(ab)) + torch.mean(torch.square(ac)) + torch.mean(torch.square(bc))

        return torch.mean(torch.abs(ab)) + torch.mean(torch.abs(ac)) + torch.mean(torch.abs(bc))

class Joint2dLoss(nn.Module):
    def __init__(self):
        super(Joint2dLoss, self).__init__()
        pass

    def forward(self, perspective_joints_h, joints2d_gt):
        return torch.mean(
            torch.square((perspective_joints_h[..., :2] - joints2d_gt[ ..., :2])) * torch.unsqueeze(joints2d_gt[ ..., 2], -1))


class OrthogonalMatrixRegurlation(nn.Module):
    def __int__(self):
        super(OrthogonalMatrixRegurlation, self).__int__()

    def forward(self, pose_matrix):
        return torch.mean( torch.square( pose_matrix[:,:,0] * pose_matrix[:,:,1])
                        + torch.square( pose_matrix[:,:,0] * pose_matrix[:,:,2])
                        + torch.square( pose_matrix[:,:,1] * pose_matrix[:,:,2]) ) \
                + torch.mean( torch.square( torch.sum( pose_matrix[:,:,0] * pose_matrix[:,:,0], dim=1) - 1)
                        + torch.square( torch.sum( pose_matrix[:,:,1] * pose_matrix[:,:,1], dim=1) - 1)
                        + torch.square( torch.sum( pose_matrix[:,:,2] * pose_matrix[:,:,2], dim=1) - 1) )
        # return torch.sum(pose_matrix[:,:,0] * pose_matrix[:,:,1] + pose_matrix[:,:,0] * pose_matrix[:,:,2] + pose_matrix[:,:,1] * pose_matrix[:,:,2] \
        #     + pose_matrix[:,0,:] * pose_matrix[:,1,:] + pose_matrix[:,0,:] * pose_matrix[:,2,:] + pose_matrix[:,1,:] * pose_matrix[:,2,:] \
        #     + (pose_matrix[:,:,0] ** 2 - 1) + (pose_matrix[:,:,1] ** 2 - 1) + (pose_matrix[:,:,2] ** 2 - 1) \
        #     + (pose_matrix[:,0,:] ** 2 - 1) + (pose_matrix[:,1,:] ** 2 - 1) + (pose_matrix[:,2,:] ** 2 - 1) )

class AposePriorLoss(nn.Module):
    def __init__(self, device):
        super(AposePriorLoss, self).__init__()
        self.device = device
        self.index = numpy2tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                           12, 13, 14, 15, 18, 19, 20, 21, 22, 23]).reshape((-1,)), np.long).to(
            self.device)

        self.index2 = numpy2tensor(np.array([16, 17]).reshape((-1,)), np.long).to(self.device)

        mean_a_pose = np.load('./assets/mean_a_pose.npy') # params_data['poses'][:1] # np.expand_dims(, 0)
        mean_a_pose[:,:3] = 0.
        # np.save('./assets/mean_a_pose.npy', mean_a_pose)
        # mean_a_pose = mean_a_pose
        self.mean_a_pose = numpy2tensor(mean_a_pose.reshape([-1, 3]), np.float32).to(self.device)

        pass
    def forward(self, pose):
        pose = pose.reshape([-1, 24, 3]) # self.dr_network.pose.

        selected_pose = pose.index_select(index=self.index, dim=1)
        l2_prioir_loss = 0.00001 * torch.sum(selected_pose.pow(2))

        pose_bias = pose - self.mean_a_pose.unsqueeze(0).repeat(pose.shape[0], 1, 1)
        a_pose_prior_loss = 0.0005 * torch.sum((pose_bias.index_select(index=self.index, dim=1)).pow(2)) \
                            + 0.0001 * torch.sum((pose_bias.index_select(index=self.index2, dim=1)).pow(2))

        return a_pose_prior_loss


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    np.random.seed(10)
    poses = torch.from_numpy(np.random.rand(100, 72)) * 10
    # poses = torch.from_numpy(np.zeros([1, 72]))

    path_gmm = '../assets/gmm_08.pkl'
    smplgmm = GMMLoss(device=device, path_gmm=path_gmm) # SmplGMM(path_gmm, flag_print=False)
    batchsmplgmm = BatchGMMLoss(device=device, path_gmm=path_gmm) # BatchSmplGMM(path_gmm, flag_print=False)

    loss_gmm_1 = 12 * 1024 * 1024 * 0.0072 * smplgmm(poses) / (69 * 69 * 10.0)
    loss_gmm_2 = 12 * 1024 * 1024 * 0.0072 * batchsmplgmm(poses) / (69 * 69 * 10.0)
    print(loss_gmm_1, loss_gmm_2)

    pass

