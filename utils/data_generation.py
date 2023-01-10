import numpy as np
import os
import copy

from math import sqrt
import cv2
import tqdm
import argparse

import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
# from smpl_model.smpl_np import SMPLModel
from smpl_model.batch_smpl_torch import SMPLModel
from utils.Render import Render
from utils.general import tensor2numpy, numpy2tensor
from smpl_model.smpl_np import SMPLModel as SMPLModelNUMPY

from geometry_model.lib import write_obj

# get front and back image
def select_keyframes(json_dir):
    body_joints, face_joints = [], []
    joints_list = sorted(os.listdir(json_dir)) # f'{root_dir}2d_joints/{name}/json/'

    for j_idx in range( (int)(len(joints_list)/8)*8 ):
        body_joint, face_joint = load_joints_from_openpose(f'{json_dir}{joints_list[j_idx]}') # {root_dir}2d_joints/{name}/json/
        body_joints.append(body_joint)
        face_joints.append(face_joint)

    body_joints = np.array(body_joints)
    face_joints = np.array(face_joints)
    hand_left_joints = body_joints[:, 7, :]
    hand_right_joints = body_joints[:, 4, :]

    # -------------------------------------------------
    # select front and back keyframes
    face_scores = face_joints[:,:,2]
    face_scores = np.mean(face_scores, axis=1)

    front_or_back = np.zeros(shape=(len(body_joints)), dtype=np.int8)
    front_or_back[np.where(face_scores >= 0.4)] = 1 # front is 1  # back is 0

    hand_left_pos_y = hand_left_joints[:, 1]
    hand_right_pos_y = hand_right_joints[:, 1]


    hand_distance_y = np.abs(hand_left_pos_y - hand_right_pos_y)

    front_hand_distance_y = np.ones_like(hand_distance_y) * np.Inf
    front_hand_distance_y[np.where(front_or_back == 1)] = hand_distance_y[np.where(front_or_back == 1)]
    back_hand_distance_y = np.ones_like(hand_distance_y) * np.Inf
    back_hand_distance_y[np.where(front_or_back == 0)] = hand_distance_y[np.where(front_or_back == 0)]

    selected_front_frame_idx = np.argmin(front_hand_distance_y)
    selected_back_frame_idx = np.argmin(back_hand_distance_y)

    # --------------------------
    return [selected_front_frame_idx, selected_back_frame_idx]


class transformation_flow():
    def __init__(self, device, vertices = None, faces = None, textures = None, img_size=1080):
        # super(tranformation_flow, self).__init__()

        # load data
        self.vertices = vertices
        self.faces = faces
        self.textures = textures
        self.img_size = img_size

        batch_size = 1
        # create renderer
        f = self.img_size
        R = np.tile(np.eye(3), [batch_size, 1, 1])
        t = np.tile(np.zeros(3), [batch_size, 1, 1])
        K = np.tile(np.array([[f, 0, f/2],
                              [0, f, f/2],
                              [0, 0, 1]
                              ]), [batch_size, 1, 1])

        self.renderer = nr.Renderer(camera_mode='projection', R=R, t=t, K=K, image_size=self.img_size)
        # self.texture_img_size = 256

        f = np.loadtxt('./assets/smpl_f_ft_vt/smpl_f.txt')
        ft = np.loadtxt('./assets/smpl_f_ft_vt/smpl_ft.txt')
        vt = np.loadtxt('./assets/smpl_f_ft_vt/smpl_vt.txt')

        self.fs = torch.from_numpy(np.expand_dims(f, axis=0)).repeat([batch_size,1,1]).type(torch.long).cuda() #.to(device)
        self.fts = torch.from_numpy(np.expand_dims(ft, axis=0)).repeat([batch_size,1,1]).type(torch.long).cuda()  #.to(device)
        self.vts = torch.from_numpy(np.expand_dims(vt, axis=0)).repeat([batch_size,1,1]).type(torch.float).cuda()  # .to(device)

        texture_size = 2
        self.textures = torch.ones(1, 13776, texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
        self.device = device

    def cal_bc_transform(self, from_faces_verts_on_img, dst_fims, dst_wims, visibility, image_size):
        """
        Args:
            src_f2pts: (bs, 13776, 3, 2)
            dst_fims:  (bs, 256, 256)
            dst_wims:  (bs, 256, 256, 3)
        Returns:

        """
        bs = 1 # src_f2pts.shape[0]
        T = -1 * torch.ones((bs, image_size * image_size, 2), dtype=torch.float32, device=self.device)
        mask = torch.zeros((bs, image_size * image_size, 1), dtype=torch.float32, device=self.device)

        for i in range(bs):
            # to_face_index_map
            to_tar_face_index_map = dst_fims[i]
            # to_weight_map
            to_weight_map = dst_wims[i]
            # (256, 256) -> (256*256, )
            to_tar_face_index_map = to_tar_face_index_map.long().reshape(-1)

            tar_face_index = torch.unique(to_tar_face_index_map)

            # (256, 256, 3) -> (256*256, 3)
            to_weight_map = to_weight_map.reshape(-1, 3)

            to_exist_mask = (to_tar_face_index_map != -1)
            # (exist_face_num,)
            to_exist_face_idx = to_tar_face_index_map[to_exist_mask]
            # (exist_face_num, 3)
            to_exist_face_weights = to_weight_map[to_exist_mask]
            # (exist_face_num, 3, 2) * (exist_face_num, 3) -> sum -> (exist_face_num, 2)
            exist_smpl_T = (from_faces_verts_on_img[0][to_exist_face_idx] * to_exist_face_weights[:, :, None]).sum(dim=1)
            # (256, 256, 2)
            T[i, to_exist_mask] = exist_smpl_T
            mask[i, to_exist_mask] = visibility[to_exist_face_idx] # torch.ones([exist_smpl_T.shape[0], 1]).cuda()

        T = T.view(bs, image_size, image_size, 2)
        mask = mask.view(bs, image_size, image_size, 1)
        # T = torch.clamp(T, -2, 2)
        return T, mask


    def load(self, tar_vertices, faces=None, textures=None):

        # self.src_vertices = torch.from_numpy(np.expand_dims(src_vertices, axis=0)).cuda()
        self.tar_vertices = tar_vertices# torch.from_numpy(np.expand_dims(tar_vertices, axis=0)).cuda()
        # self.faces = faces if faces is not None else self.fs
        # self.textures = textures if textures is not None else self.textures

    def load_src_vertices_list(self, src_vertices_list):
        self.src_vertices_list = src_vertices_list
        self.faces = self.fs
        self.textures = self.textures

        self.src_model_images_list = []
        self.src_p2verts_list = []
        self.src_fim_list = []
        self.src_wim_list = []
        self.from_faces_verts_on_img_list = []
        self.visibility_list = []

        def erode(map):
            _map = copy.deepcopy(map)  # np.zeros_like(map)
            pos = np.argwhere(map != -1)
            for p_idx in range(pos.shape[0]):
                # for idx in range(1, map.shape[0]-1):
                #     for jIdx in range(map.shape[1]):
                jIdx = pos[p_idx][0]
                idx = pos[p_idx][1]
                if map[jIdx, idx] == -1:
                    _map[jIdx, idx] = map[jIdx, idx]
                elif map[jIdx, idx - 1] == -1:
                    _map[jIdx, idx] = -1
                elif map[jIdx, idx + 1] == -1:
                    _map[jIdx, idx] = -1
                else:
                    _map[jIdx, idx] = map[jIdx, idx]
            return _map

        for idx in range(len(src_vertices_list)):
            src_vertices = src_vertices_list[idx] # torch.from_numpy(np.expand_dims(src_vertices_list[idx], axis=0)).cuda()

            src_model_images, _, _ = self.renderer(src_vertices, self.faces, self.textures)

            src_f2verts, src_fim, src_wim = self.renderer(src_vertices, self.faces, mode='fim_and_wim')
            src_p2verts = src_f2verts[:, :, :, 0:2]
            src_p2verts[:, :, :, 1] *= -1

            self.src_model_images_list.append(src_model_images)
            self.src_p2verts_list.append(src_p2verts)
            self.src_fim_list.append(src_fim)
            self.src_wim_list.append(src_wim)

            # (13776, 3, 2)
            from_faces_verts_on_img = src_p2verts[0].clone()

            # cal area
            ab = (from_faces_verts_on_img[:, 1, :] - from_faces_verts_on_img[:, 0, :]) * 10
            ac = (from_faces_verts_on_img[:, 2, :] - from_faces_verts_on_img[:, 0, :]) * 10
            ab = torch.cat([ab, torch.zeros_like(ab[:, -1:])], dim=1)
            ac = torch.cat([ac, torch.zeros_like(ac[:, -1:])], dim=1)
            cross_dot = torch.cross(ab, ac, dim=1)
            s = torch.abs(torch.sum(cross_dot, dim=1))
            s = s.detach().cpu().numpy()

            # -------------------------------
            visibility = torch.zeros([from_faces_verts_on_img.shape[0], 1]).cuda()

            to_src_face_index_map = src_fim[0]
            _face_index_map = to_src_face_index_map.detach().cpu().numpy()

            # _eroded_face_index_map = erode(erode(erode(erode(erode(_face_index_map)))))
            _eroded_face_index_map = copy.deepcopy(_face_index_map)

            eroded_face_index_map = torch.from_numpy(_eroded_face_index_map)
            to_src_face_index_map = to_src_face_index_map.long().reshape(-1)
            src_face_index = torch.unique(eroded_face_index_map) # torch.unique(to_src_face_index_map)
            # -------------------------------
            _src_face_index = src_face_index.detach().cpu().numpy()
            _src_face_index = np.delete(_src_face_index, 0)
            _vis_face_index = []
            _invis_face_index = [] # np.zeros((13776 - len(src_face_index)))
            for num_idx in range(from_faces_verts_on_img.shape[0]):
                if num_idx in _src_face_index and s[num_idx] > 0.0001: # square area
                    _vis_face_index.append(num_idx)
                else:
                    _invis_face_index.append(num_idx)
            invis_src_face_index = torch.from_numpy(np.array(_invis_face_index))
            vis_src_face_index = torch.from_numpy(np.array(_vis_face_index))

            from_faces_verts_on_img[invis_src_face_index] = -1 * torch.ones([3, 2]).cuda()
            visibility[vis_src_face_index] = torch.ones([1]).cuda()

            self.from_faces_verts_on_img_list.append(from_faces_verts_on_img)
            self.visibility_list.append(visibility)

            # -------------------------------
        pass

    def warp(self, src_img, src_idx):
        src_img = src_img.unsqueeze(0)

        tar_model_images, _, _ = self.renderer(self.tar_vertices, self.faces, self.textures)
        tar_f2verts, tar_fim, tar_wim = self.renderer(self.tar_vertices, self.faces, mode='fim_and_wim')
        T, mask = self.cal_bc_transform(self.src_p2verts_list[src_idx], tar_fim, tar_wim, self.visibility_list[src_idx], self.img_size)

        tsf_img = F.grid_sample(src_img, T)

        return tsf_img, self.src_model_images_list[src_idx], tar_model_images, mask

    def forward(self, src_img):
        src_model_images, _, _ = self.renderer(self.src_vertices, self.faces, self.textures)
        #tar_model_images, _, _ = self.renderer(self.tar_vertices, self.faces, self.textures)

        # image = src_images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        # cv2.imwrite('../output/s.png', image[:, :, :] * 255)
        image = tensor2numpy(src_model_images)[0].transpose((1, 2, 0))
        cv2.imwrite('../output/source.png', image[:, :, ::-1] * 255)
        # image = tar_model_images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        # cv2.imwrite('../output/target.png', image[:, :, :] * 255)
        # image = tsf_img.detach().cpu().numpy()[0].transpose((1, 2, 0))
        # cv2.imwrite('../output/target.png', image[:, :, ::-1] * 255)

class UVMapGenerator:
    def __init__(self, device, root_dir, src_data_path, name):

        self.device = device
        self.isupsample = True
        self.root_dir = root_dir

        self.name = name

        uv = np.load('./assets/uv.npy')
        self.uv = torch.from_numpy(np.expand_dims(uv, axis=0))
        # uv = uv.repeat(1, 1, 1, 1)

        # num = (int)(name.split('_')[-1])
        # self.num = num - num % 8
        model_path = './assets/smpl/neutral_smpl.pkl' if self.isupsample == False else \
            './assets/upsmpl/neutral_smpl.pkl'
        self.smpl = SMPLModel(device = self.device, model_path=model_path, use_posematrix=False)

        # self.smpl = SMPLModel(device = self.device, model_path='./assets/neutral_smpl.pkl', use_posematrix=False) \
        #     if self.isupsample == False \
        #     else UPSMPLModel(device = self.device, model_path='./assets/upsample_neutral_smpl.pkl', use_posematrix=False)


        self.pose = numpy2tensor(np.load(f'{src_data_path}/pose.npy')).to(self.device)
        self.betas = numpy2tensor(np.load(f'{src_data_path}/betas.npy')).to(self.device)
        self.trans = numpy2tensor(np.load(f'{src_data_path}/trans.npy')).to(self.device)
        self.offset = numpy2tensor(np.load(f'{src_data_path}/offsets.npy')).to(self.device)


        # -------------------------------
        vert = self.smpl(pose=self.pose[0:0 + 1],
                         betas=self.betas[0:1],
                         trans=self.trans[0:0 + 1],
                         displacements=self.offset[0:0 + 1])

        f = np.loadtxt('./assets/smpl_f_ft_vt/smpl_f.txt')
        ft = np.loadtxt('./assets/smpl_f_ft_vt/smpl_ft.txt')
        vt = np.loadtxt('./assets/smpl_f_ft_vt/smpl_vt.txt')

        os.makedirs(f'./results/final_texture/', exist_ok=True)
        write_obj(vs=tensor2numpy(vert[0]), vt=vt, fs=f, ft=ft, path=f'./results/final_texture/{name}.obj', write_mtl=True)
        pass

    def generate_uv(self, target_uv_path, target_ref_path=None, img_size=1024, pose_seq = 'src', generate_ref=False):
        os.makedirs(target_uv_path, exist_ok=True)
        if generate_ref == True:
            os.makedirs(target_ref_path, exist_ok=True)

        self.render = Render(img_size=[img_size, img_size], batch_size=1, device=self.device, f=[img_size, img_size],
                      c=[img_size / 2, img_size / 2], isupsample=self.isupsample)
        self.render.load(self.uv)
        verts, src_verts = [], []
        for pose_idx in range(self.pose.shape[0]):
            vert = self.smpl(pose=self.pose[pose_idx:pose_idx + 1], betas=self.betas[0:1],
                             trans=self.trans[pose_idx:pose_idx + 1], displacements=self.offset[pose_idx:pose_idx + 1])
            src_verts.append(vert)

        if pose_seq == 'src':
            verts = src_verts
        elif pose_seq == 'self_rotate':
            pose_c = copy.deepcopy(self.pose[0]).reshape([-1, 3])
            pose_c[0] = torch.Tensor([0, 0, 0]).to(self.device)
            for pose_idx in range(360):
                pose_c[0] = pose_c[0] + torch.Tensor([0, 1 / 360 * 3.14 * 2, 0]).to(self.device)
                vert = self.smpl(pose=pose_c.reshape([-1]), betas=self.betas[0:1],
                                  trans=self.trans[0:1], displacements=self.offset[0:1])
                verts.append(vert)

        if generate_ref == True:
            self.trans_flow = transformation_flow(device=self.device, img_size=img_size)

            keyframes = select_keyframes(f'{self.root_dir}/2d_joints/{self.name}/json/')
            src_vertices_list, src_img_list = [], []
            for keyframe in keyframes:
                src_img = cv2.imread(f'{self.root_dir}/frames_mat/{self.name}/{str(keyframe).zfill(4)}.png')
                src_img = cv2.resize(src_img, (img_size, img_size))
                src_img = np.transpose(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB), [2, 0, 1])
                src_img = numpy2tensor(src_img / 255).to(self.device)

                src_vertices = copy.deepcopy(src_verts[keyframe])
                src_vertices[:, :, 1:] *= -1

                src_img_list.append(src_img)
                src_vertices_list.append(src_vertices)
                self.trans_flow.load_src_vertices_list(src_vertices_list)


        with torch.no_grad():
            for pose_idx in tqdm.tqdm(range(len(verts))):
                # vert = self.smpl(pose=self.pose[pose_idx:pose_idx+1], betas=self.betas[0:1],
                #                             trans=self.trans[pose_idx:pose_idx+1], displacements=self.offset[pose_idx:pose_idx+1])
                uvs = self.render.get_uv_img(verts[pose_idx], output_path='vis_uv')  # .squeeze()
                uv = uvs[0]
                np.savez_compressed(f'{target_uv_path}/{str(pose_idx + 1).zfill(4)}', uv)

                if generate_ref == True:
                    tar_vertice = copy.deepcopy(verts[pose_idx])
                    tar_vertice[:, :, 1:] *= -1
                    final_img = np.zeros([img_size, img_size, 3])
                    full_mask = np.zeros([img_size, img_size, 1])
                    for src_idx in range(len(src_img_list)):
                        src_img = src_img_list[src_idx]
                        self.trans_flow.load(tar_vertice)
                        tar_img, _, _, mask = self.trans_flow.warp(src_img, src_idx)

                        image = tensor2numpy(tar_img)[0].transpose((1, 2, 0))[:, :, ::-1]
                        mask_img = tensor2numpy(mask)[0]

                        mask_img[np.where(full_mask[:, :, 0] == 1)] = 0
                        final_img[np.where(mask_img[:, :, 0] == 1)] = image[np.where(mask_img[:, :, 0] == 1)]
                        full_mask[np.where(mask_img[:, :, 0] == 1)] = 1

                    cv2.imwrite(f'{target_ref_path}/mask_{str(pose_idx+1).zfill(4)}.png', full_mask * 255)
                    cv2.imwrite(f'{target_ref_path}/ref_{str(pose_idx+1).zfill(4)}.png', final_img * 255)




class UVPositionalMapGenerator:
    def __init__(self, src_data_path):
        self.fs = np.loadtxt('./assets/smpl_f_ft_vt/smpl_f.txt').astype(np.int32)
        self.fts = np.loadtxt('./assets/smpl_f_ft_vt/smpl_ft.txt').astype(np.int32)
        self.vts = np.loadtxt('./assets/smpl_f_ft_vt/smpl_vt.txt').astype(np.float32)
        self.smpl = SMPLModelNUMPY(model_path='./assets/smpl/neutral_smpl.pkl', use_posematrix=False)

        # base_path = '../results/diff_rendering/{}_upsample_ground/stage1/'.format(name)
        self.pose = np.load(f'{src_data_path}/pose.npy').astype(np.float32)
        self.betas = np.load(f'{src_data_path}/betas.npy').astype(np.float32)
        self.trans = np.load(f'{src_data_path}/trans.npy').astype(np.float32)
        # self.offsets = np.load(src_data_path + '/offsets.npy').astype(np.float32)

        self.mean_trans = np.mean(self.trans, axis=0).reshape([3])
        self.uv_img_size = 128

    def generate_uv(self, target_uv_path):
        os.makedirs(target_uv_path, exist_ok=True)
        for idx in tqdm.tqdm(range(self.pose.shape[0])):
            naked_vertice = self.smpl.set_params(beta=self.betas[0], pose=self.pose[idx],
                                        trans=self.trans[idx] - self.mean_trans,
                                        )

            uv_positional_map = np.zeros([self.uv_img_size, self.uv_img_size, 3])
            for face_idx in range(self.fs.shape[0]):
                f = self.fs[face_idx]
                ft = self.fts[face_idx]

                vt1 = self.vts[ft[0]]
                vt2 = self.vts[ft[1]]
                vt3 = self.vts[ft[2]]

                posax = ((vt1[0]) * self.uv_img_size)
                posay = ((1 - vt1[1]) * self.uv_img_size)

                posbx = ((vt2[0]) * self.uv_img_size)
                posby = ((1 - vt2[1]) * self.uv_img_size)

                poscx = ((vt3[0]) * self.uv_img_size)
                poscy = ((1 - vt3[1]) * self.uv_img_size)

                v1 = naked_vertice[f[0]]
                v2 = naked_vertice[f[1]]
                v3 = naked_vertice[f[2]]

                uv_positional_map[(int)(posay), (int)(posax), :] = v1
                uv_positional_map[(int)(posby), (int)(posbx), :] = v2
                uv_positional_map[(int)(poscy), (int)(poscx), :] = v3

                lengtha = sqrt((posax - posbx) * (posax - posbx) + (posay - posby) * (posay - posby))
                lengthb = sqrt((posbx - poscx) * (posbx - poscx) + (posby - poscy) * (posby - poscy))
                lengthc = sqrt((poscx - posax) * (poscx - posax) + (poscy - posay) * (poscy - posay))

                length = max(max(lengtha, lengthb), lengthc)

                step = 0.5 / length  # 0.03
                for p in np.arange(0, 1, step):
                    for q in np.arange(0, 1 - p, step):
                        posx = p * posax + q * posbx + (1 - p - q) * poscx
                        posy = p * posay + q * posby + (1 - p - q) * poscy

                        v = p * v1 + q * v2 + (1 - p - q) * v3

                        uv_positional_map[int(posy), int(posx)] = v

            np.save(f'{target_uv_path}/{str(idx+1).zfill(4)}.npy', uv_positional_map)
            cv2.imwrite(f'{target_uv_path}/{str(idx+1).zfill(4)}.png', (uv_positional_map / 2 + 0.5) * 255 )

import neural_renderer as nr
from dataset.make_dataset import load_joints_from_openpose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='Body2D_2040_499')  #  Body2D_2027_286 Body2D_2010_378 Body2D_2061_507 Body2D_2040_499   Body2D_2041_308 Body2D_2070_380
    parser.add_argument('--device_id', type=str, default='2')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id # '3'

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    name = args.name # 'Body2D_2070_380'
    # src_data_path = './results/new_diff_rendering/' + name
    target_path = './results'

    # uv_pos_generator = UVPositionalMapGenerator(src_data_path=f'./results/new_diff_rendering/{name}')
    # uv_pos_generator.generate_uv(target_uv_path=f'{target_path}/naked_vertice_uv/{name}')

    uv_generator = UVMapGenerator(device=device, src_data_path=f'./results/dynamic_offsets/{name}', name=name)
    uv_generator.generate_uv(target_uv_path=f'{target_path}/uvs/{name}',
                             target_ref_path=f'{target_path}/ref/{name}',
                             pose_seq = 'src', img_size=1024, generate_ref=True)
    uv_generator.generate_uv(target_uv_path=f'{target_path}/test_uvs/{name}',
                             target_ref_path=f'{target_path}/test_ref/{name}',
                             pose_seq = 'self_rotate', img_size=1024, generate_ref=True)
    uv_generator.generate_uv(target_uv_path=f'{target_path}/double_test_uvs/{name}',
                             pose_seq='self_rotate', img_size=2048, generate_ref=False)




    pass