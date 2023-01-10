import os
import sys
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
from utils.general import numpy2tensor
from geometry_model.lib import write_obj


class LaplacianLoss(nn.Module):

    def __init__(self, device, isupsample=False, stage='1'):
        super(LaplacianLoss, self).__init__()

        self.isupsample = isupsample

        self.device = device
        self.stage = stage

        v_ids_path = './assets/smpl/vertices_label.pkl' if isupsample is False\
            else './assets/upsmpl/vertices_label.pkl'

        with open(v_ids_path, 'rb') as file:
            self.v_ids = pkl.load(file)

        self.adjacency_set = \
            np.load('./assets/smpl/adjacency_set.npy') if isupsample is False \
            else np.load('./assets/upsmpl/adjacency_set.npy')
        self.laplace_w = \
            numpy2tensor(self.regularize_laplace()[:, np.newaxis])\
            .to(self.device)

    def regularize_laplace(self):
        reg = np.ones(6890) if self.isupsample is False else np.ones(27554)
        v_ids = self.v_ids

        if self.isupsample is False:
            scale = 4
            reg[v_ids['face']] = 8. * 2
            reg[v_ids['left_hand']] = 5. * scale
            reg[v_ids['right_hand']] = 5. * scale
            reg[v_ids['left_fingers']] = 8. * scale
            reg[v_ids['right_fingers']] = 8. * scale
            reg[v_ids['left_foot']] = 5. * scale
            reg[v_ids['right_foot']] = 5. * scale
            reg[v_ids['left_toes']] = 8. * scale  # 8.
            reg[v_ids['right_toes']] = 8. * scale  # 8.
            reg[v_ids['left_ear']] = 10. * scale
            reg[v_ids['right_ear']] = 10. * scale
            reg[v_ids['armpits']] = 10

        else:
            reg[v_ids['left_arm']] = 800  # 150
            reg[v_ids['right_arm']] = 800  # 150
            reg[v_ids['left_hand']] = 10
            reg[v_ids['right_hand']] = 10

            reg[v_ids['forward_body']] = 50
            reg[v_ids['backward_body']] = 10
            reg[v_ids['left_leg']] = 10
            reg[v_ids['right_leg']] = 10
            reg[v_ids['left_head']] = 10
            reg[v_ids['right_head']] = 10
            reg[v_ids['left_foot']] = 10
            reg[v_ids['right_foot']] = 10

            # ---------------------------
            reg[v_ids['face']] = 1600  # 3200.
            reg[v_ids['left_hand']] = 500.
            reg[v_ids['right_hand']] = 500.
            reg[v_ids['left_fingers']] = 800.
            reg[v_ids['right_fingers']] = 800.
            reg[v_ids['left_foot']] = 500.
            reg[v_ids['right_foot']] = 500.
            reg[v_ids['left_toes']] = 800.  # 8.
            reg[v_ids['right_toes']] = 800.  # 8.
            reg[v_ids['left_ear']] = 1000.
            reg[v_ids['right_ear']] = 1000.

            reg[v_ids['nose']] = 2  # 10
            reg[v_ids['chest']] = 400
            reg[v_ids['waist']] = 400
            reg[v_ids['armpits_and_crotch']] = 800

        return reg

    def laplace_coord(self, v, adjacency_set):

        vertex = torch.cat([v, torch.zeros([1, 3]).to(self.device)], dim=0)
        indices = numpy2tensor(adjacency_set[:, :9].reshape((-1, )),
                               np.int64).to(self.device)
        weights = numpy2tensor(adjacency_set[:, -1:],
                               np.float32).to(self.device)

        weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))

        vertices = vertex.index_select(index=indices, dim=0)  #
        vertices = vertices.reshape((v.shape[0], -1, 3))  # 6890

        laplace = torch.sum(vertices, dim=1)

        laplace_vertex = v - torch.mul(laplace, weights)

        return laplace_vertex

    def compute_laplacian_diff(self, v_1, v_2):
        lap1 = self.laplace_coord(v_1, self.adjacency_set)
        lap2 = self.laplace_coord(v_2, self.adjacency_set)

        laplace_loss = torch.square(lap1 - lap2)  # L2 distance
        laplace_loss = torch.mean(laplace_loss * self.laplace_w)
        return laplace_loss

    def forward(self, v_1, v_2):
        return self.compute_laplacian_diff(v_1, v_2)


class TemporalLoss(nn.Module):

    def __init__(self):
        super(TemporalLoss, self).__init__()
        pass

    def forward(self, x):
        forward_x = torch.roll(x, shifts=(1), dims=(0))
        backward_x = torch.roll(x, shifts=(-1), dims=(0))
        bias = (x - (forward_x + backward_x) / 2)[1:-1]
        loss = torch.mean(torch.square(bias))
        return loss


class EDGELoss(nn.Module):

    def __init__(self, device, isupsample=False):
        super(EDGELoss, self).__init__()

        face_path = './assets/smpl_f_ft_vt/smpl_f.txt' if isupsample is False \
            else './assets/upsmpl_f_ft_vt/smpl_f.txt'
        self.f = np.loadtxt(face_path)
        self.fs = numpy2tensor(np.expand_dims(self.f, axis=0),
                               np.long).repeat([1, 1, 1]).to(device)

    def forward(self, v):
        batch_size = v.shape[0]

        f = self.fs.reshape([-1])
        f_v = v.index_select(dim=1, index=f)
        f_v = f_v.reshape([batch_size, self.fs.shape[1], 3, 3])
        ab = f_v[:, :, 1, :] - f_v[:, :, 0, :]
        # m, n = torch.max(ab), torch.min(ab)
        ac = f_v[:, :, 2, :] - f_v[:, :, 0, :]
        bc = f_v[:, :, 1, :] - f_v[:, :, 2, :]

        return \
            torch.mean(torch.abs(ab)) + \
            torch.mean(torch.abs(ac)) + \
            torch.mean(torch.abs(bc))


class Joint2dLoss(nn.Module):

    def __init__(self):
        super(Joint2dLoss, self).__init__()
        pass

    def forward(self, perspective_joints_h, joints2d_gt):
        return torch.mean(
            torch.square(
                (perspective_joints_h[..., :2] - joints2d_gt[..., :2])) *
            joints2d_gt[..., 2:])


class OrthogonalMatrixRegurlation(nn.Module):

    def __int__(self):
        super(OrthogonalMatrixRegurlation, self).__int__()

    def forward(self, pose_matrix):
        return \
            torch.mean(
                torch.square(pose_matrix[:, :, 0] * pose_matrix[:, :, 1]) +
                torch.square(pose_matrix[:, :, 0] * pose_matrix[:, :, 2]) +
                torch.square(pose_matrix[:, :, 1] * pose_matrix[:, :, 2])) + \
            torch.mean(
                torch.square(torch.sum(pose_matrix[:, :, 0] *
                                       pose_matrix[:, :, 0],
                                       dim=1) - 1) +
                torch.square(torch.sum(pose_matrix[:, :, 1] *
                                       pose_matrix[:, :, 1],
                                       dim=1) - 1) +
                torch.square(torch.sum(pose_matrix[:, :, 2] *
                                       pose_matrix[:, :, 2],
                                       dim=1) - 1))


class AposePriorLoss(nn.Module):

    def __init__(self, device):
        super(AposePriorLoss, self).__init__()
        self.device = device
        self.index = \
            numpy2tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                   12, 13, 14, 15, 18, 19, 20, 21, 22, 23])
                         .reshape((-1,)),
                         np.long).to(self.device)

        self.index2 = \
            numpy2tensor(np.array([16, 17]).reshape((-1,)),
                         np.long).to(self.device)

        mean_a_pose = np.load('./assets/mean_a_pose.npy')
        mean_a_pose[:, :3] = 0.
        # np.save('./assets/mean_a_pose.npy', mean_a_pose)
        # mean_a_pose = mean_a_pose
        self.mean_a_pose = numpy2tensor(
            mean_a_pose.reshape([-1, 3]), np.float32).to(self.device)

        pass

    def forward(self, pose):
        pose = pose.reshape([-1, 24, 3])  # self.dr_network.pose.

        pose_bias = \
            pose - self.mean_a_pose.unsqueeze(0).repeat(pose.shape[0], 1, 1)

        a_pose_prior_loss = \
            0.0005 * torch.sum(
                (pose_bias.index_select(index=self.index, dim=1)).pow(2)) \
            + 0.0001 * torch.sum(
                (pose_bias.index_select(index=self.index2, dim=1)).pow(2))

        return a_pose_prior_loss


def get_parts(part_name='armpits_and_crotch'):
    # ---------------
    # data = np.loadtxt('./assets/armpits_and _crotch.txt', dtype='str')
    with open('./base_assets/selected_obj/{}.obj'.format(part_name), 'r') as file:
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

    # write_obj(vs=np.take(v_template, index, axis=0), fs=None, path='./results/{}.obj'.format(part_name))
    # write_obj(params['v_template'], params['f'], './results/v_template.obj')
    return np.array(index)


def merge():
    if False:
        path = './base_assets/bodyparts.pkl'
        with open(path, 'rb') as fp:
            bodyparts = pkl.load(fp, encoding='iso-8859-1')

        path = './base_assets/vertex_label.pkl'
        with open(path, 'rb') as fp:
            vertex_label = pkl.load(fp, encoding='iso-8859-1')

        armpits_ids = get_parts('armpits')
        smpl_vertices_label = {}
        smpl_vertices_label['face'] = bodyparts['face']
        smpl_vertices_label['left_hand'] = bodyparts['hand_l']
        smpl_vertices_label['right_hand'] = bodyparts['hand_r']
        smpl_vertices_label['left_fingers'] = bodyparts['fingers_l']
        smpl_vertices_label['right_fingers'] = bodyparts['fingers_r']
        smpl_vertices_label['left_foot'] = bodyparts['foot_l']
        smpl_vertices_label['right_foot'] = bodyparts['foot_r']
        smpl_vertices_label['left_toes'] = bodyparts['toes_l']
        smpl_vertices_label['right_toes'] = bodyparts['toes_r']
        smpl_vertices_label['left_ear'] = bodyparts['ear_l']
        smpl_vertices_label['right_ear'] = bodyparts['ear_r']


        smpl_vertices_label['left_arm'] = vertex_label['right_arm']
        smpl_vertices_label['right_arm'] = vertex_label['left_arm']
        smpl_vertices_label['left_hand']  = vertex_label['right_hand']
        smpl_vertices_label['right_hand'] = vertex_label['left_hand']
        smpl_vertices_label['forward_body'] = vertex_label['forward_body']
        smpl_vertices_label['backward_body'] = vertex_label['backward_body']
        smpl_vertices_label['left_leg'] = vertex_label['right_leg']
        smpl_vertices_label['right_leg'] = vertex_label['left_leg']
        smpl_vertices_label['left_head'] = vertex_label['right_head']
        smpl_vertices_label['right_head'] = vertex_label['left_head']
        smpl_vertices_label['left_foot'] = vertex_label['right_foot']
        smpl_vertices_label['right_foot'] = vertex_label['left_foot']

        smpl_vertices_label['armpits'] = armpits_ids

        with open('./assets/smpl_vertices_label.pkl', 'wb') as file:
            pkl.dump(smpl_vertices_label, file)

        with open('./assets/smpl_vertices_label.pkl', 'rb') as file:
            smpl_vertices_label = pkl.load(file)

        with open('./assets/neutral_smpl.pkl', 'rb') as file:
            neutral_smpl = pkl.load(file, encoding='iso-8859-1')

        os.makedirs('./results/smpl_vertices/', exist_ok=True)
        for _, key in enumerate(smpl_vertices_label):

            write_obj(vs=np.take(neutral_smpl['v_template'], smpl_vertices_label[key], axis=0),
                      path=f'./results/smpl_vertices/{key}.obj')

    # sys.exit(0)
    # -----------------------------------------------
    path = './base_assets/upsample_bodyparts.pkl'

    with open(path, 'rb') as fp:
        bodyparts = pkl.load(fp, encoding='iso-8859-1')

    path = './base_assets/upsample_vertex_label.pkl'
    with open(path, 'rb') as fp:
        vertex_label = pkl.load(fp, encoding='iso-8859-1')

    armpits_and_crotch_ids = get_parts('armpits_and_crotch')
    nose_ids = get_parts('nose')
    chest_ids = get_parts('chest')
    waist_ids = get_parts('waist')

    smpl_vertices_label = {}
    smpl_vertices_label['face'] = bodyparts['face']
    smpl_vertices_label['left_hand'] = bodyparts['hand_l']
    smpl_vertices_label['right_hand'] = bodyparts['hand_r']
    smpl_vertices_label['left_fingers'] = bodyparts['fingers_l']
    smpl_vertices_label['right_fingers'] = bodyparts['fingers_r']
    smpl_vertices_label['left_foot'] = bodyparts['foot_l']
    smpl_vertices_label['right_foot'] = bodyparts['foot_r']
    smpl_vertices_label['left_toes'] = bodyparts['toes_l']
    smpl_vertices_label['right_toes'] = bodyparts['toes_r']
    smpl_vertices_label['left_ear'] = bodyparts['ear_l']
    smpl_vertices_label['right_ear'] = bodyparts['ear_r']

    smpl_vertices_label['left_arm'] = vertex_label['right_arm']
    smpl_vertices_label['right_arm'] = vertex_label['left_arm']
    smpl_vertices_label['left_hand'] = vertex_label['right_hand']
    smpl_vertices_label['right_hand'] = vertex_label['left_hand']
    smpl_vertices_label['forward_body'] = vertex_label['forward_body']
    smpl_vertices_label['backward_body'] = vertex_label['backward_body']
    smpl_vertices_label['left_leg'] = vertex_label['right_leg']
    smpl_vertices_label['right_leg'] = vertex_label['left_leg']
    smpl_vertices_label['left_head'] = vertex_label['right_head']
    smpl_vertices_label['right_head'] = vertex_label['left_head']
    smpl_vertices_label['left_foot'] = vertex_label['right_foot']
    smpl_vertices_label['right_foot'] = vertex_label['left_foot']

    smpl_vertices_label['nose'] = nose_ids
    smpl_vertices_label['chest'] = chest_ids
    smpl_vertices_label['waist_ids'] = waist_ids
    smpl_vertices_label['armpits_and_crotch'] = armpits_and_crotch_ids

    with open('./assets/upsmpl_vertices_label.pkl', 'wb') as file:
        pkl.dump(smpl_vertices_label, file)

    with open('./assets/upsmpl_vertices_label.pkl', 'rb') as file:
        smpl_vertices_label = pkl.load(file)

    with open('./assets/upsample_neutral_smpl.pkl', 'rb') as file:
        neutral_smpl = pkl.load(file, encoding='iso-8859-1')

    os.makedirs('./results/upsmpl_vertices/', exist_ok=True)
    for _, key in enumerate(smpl_vertices_label):
        write_obj(vs=np.take(neutral_smpl['v_template'], smpl_vertices_label[key], axis=0),
            path=f'./results/upsmpl_vertices/{key}.obj')


if __name__ == '__main__':



    pass
