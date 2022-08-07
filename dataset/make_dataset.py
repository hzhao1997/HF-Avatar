import numpy as np
import os
import pickle as pkl
import cv2
import random
import json
import copy
import math

import torch
from torch.utils.data import Dataset
from utils.general import tensor2numpy, numpy2tensor

cv2.setNumThreads(0)

def post_process_invisible_joints(body):
    if len(body.shape) == 2:
        if body[4, 2] < 0.2:
            body[4, :2] = 3. / 4 * body[8, :2] + 1. / 4 * body[1, :2]
            body[4, 2] = 0.3
        if body[7, 2] < 0.2:
            body[7, :2] = 3. / 4 * body[8, :2] + 1. / 4 * body[1, :2]
            body[7, 2] = 0.3

    return body

def load_joints_from_openpose(file_path):
    with open(file_path) as f:
        data = json.load(f)['people'][0]

        body = np.array(data['pose_keypoints_2d']).reshape(-1, 3)
        body[:, 2] /= np.expand_dims(np.mean(body[:, 2][body[:, 2] > 0.1]), -1)

        face = np.array(data['face_keypoints_2d']).reshape(-1, 3)

    body = post_process_invisible_joints(body)
    return body, face

def load_normalized_joints_from_openpose(file_path, resolution=(1080, 1080)):
    body, face = load_joints_from_openpose(file_path)

    body = body * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
    body[:, 0] *= 1. * resolution[1] / resolution[0]

    face = face * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
    face[:, 0] *= 1. * resolution[1] / resolution[0]

    return body, face


def load_joints_from_mmpose(root_dir, name, length):
    body_joints = np.load(root_dir + '/2d_joints/' + name + '/body25_keypoints2d.npy')[:length]
    face_joints = np.load(root_dir + '/2d_joints/' + name + '/face70_keypoints2d.npy')[:length]

    return body_joints, face_joints

def load_normalized_joints_from_mmpose(root_dir, name, length, resolution=(1080, 1080)):

    body_joints, face_joints = load_joints_from_mmpose(root_dir, name, length)

    #  body_joints[:, :, 2] /= np.expand_dims(np.mean(body_joints[:, :, 2][body_joints[:, :, 2] > 0.1]), -1)
    body_joints = body_joints * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
    body_joints[:, :, 0] *= 1. * resolution[1] / resolution[0]

    face_joints = face_joints * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
    face_joints[:, :, 0] *= 1. * resolution[1] / resolution[0]

    return body_joints, face_joints

def distinguish_rotate_direction(root_dir, name, length):
    # body_joints = np.load(root_dir + '/2d_joints/' + name + '/body25_keypoints2d.npy')[:length]
    # face_joints = np.load(root_dir + '/2d_joints/' + name + '/face70_keypoints2d.npy')[:length]
    # mid_frame = int(length / 2)
    body_joints, face_joints = load_normalized_joints_from_mmpose(root_dir, name, length)

    occuled_right_hand_id = np.where(body_joints[:, 4, 2] < 0.6)[0]
    occuled_left_hand_id = np.where(body_joints[:, 7, 2] < 0.6)[0]
    # occuled_frame_id = np.concatenate([occuled_right_hand_id, occuled_left_hand_id], axis=0)
    # if body_joints[:,4,:]
    occuled_right_hand_mean_id = np.mean(occuled_right_hand_id)
    occuled_left_hand_mean_id = np.mean(occuled_left_hand_id)
    return 'left' if occuled_left_hand_mean_id < occuled_right_hand_mean_id else 'right'

def get_all_silhouettes(root_dir, name, length=0):
    silhouette_imgs = []
    for idx in range(1, length + 1):
        silhouette_img = cv2.imread(root_dir + '/mask_mat/' + name + '/' + str(idx).zfill(4) + '.png')
        # silhouette_img[np.where(silhouette_img > 128)] = 255
        # silhouette_img[np.where(silhouette_img < 128)] = 0
        silhouette_img = cv2.resize(silhouette_img, (1024, 1024))
        silhouette_imgs.append(silhouette_img[:, :, 0])
    silhouette_imgs = np.array(silhouette_imgs)

    return silhouette_imgs

def get_all_joints_from_openpose(root_dir, name, length=0):
    body_joints = []
    face_joints = []

    # length = length - length % 8
    for idx in range(1, length + 1):
        body, face = load_normalized_joints_from_openpose(
            root_dir + '/2d_joints/' + name + '/json/' + str(idx).zfill(4) + '_keypoints.json', resolution=(1024, 1024))
        # body = post_process_invisible_joints(body)
        body_joints.append(body)
        face_joints.append(face)

    body_joints = np.array(body_joints)
    face_joints = np.array(face_joints)

    body_joints = numpy2tensor(np.array(body_joints)).contiguous()
    face_joints = numpy2tensor(np.array(face_joints)).contiguous()

    return body_joints, face_joints  # , occuled_frame_id, hand_type

def get_all_joints_from_mmpose(root_dir, name, length=0):

    body_joints, face_joints = load_normalized_joints_from_mmpose(root_dir, name, length, resolution=(1024, 1024))
    # occuled_right_hand_id = np.where(body_joints[:, 4, 2] < 0.5)[0]
    # occuled_left_hand_id = np.where(body_joints[:, 7, 2] < 0.5)[0]
    # occuled_frame_id = np.concatenate([occuled_right_hand_id, occuled_left_hand_id], axis=0)
    # hand_type = np.concatenate([np.zeros_like(occuled_right_hand_id), np.ones_like(occuled_left_hand_id)], axis=0) # 0 for right & 1 for left

    body_joints = numpy2tensor(np.array(body_joints)).contiguous()
    face_joints = numpy2tensor(np.array(face_joints)).contiguous()

    return body_joints, face_joints # , occuled_frame_id, hand_type


limbSeq = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
           [1, 8], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
           [0, 15], [0, 16], [15, 17], [16, 18],
           [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24],
           ]
line_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [255, 0, 255], [255, 0, 170], [255, 0, 85],
               [170, 0, 0], [170, 85, 0], [170, 170, 0], [170, 255, 0], [170, 0, 255], [170, 0, 170], [170, 0, 85],
               [85, 0, 0], [85, 85, 0], [85, 170, 0], [85, 255, 0], [85, 0, 255], [85, 0, 170], [85, 0, 85],
               [0, 255, 0], [0, 255, 85], [0, 255, 170]
               ]
point_colors = [[255, 0, 0], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [255, 0, 255], [255, 0, 170],
                [255, 0, 85],
                [170, 0, 0], [170, 85, 0], [170, 170, 0], [170, 255, 0], [170, 0, 255], [170, 0, 170], [170, 0, 85],
                [85, 0, 0], [85, 85, 0], [85, 170, 0], [85, 255, 0], [85, 0, 255], [85, 0, 170], [85, 0, 85],
                [0, 255, 0], [0, 255, 85], [0, 255, 170]
                ]

def generate_pose_encoding(keypoint, idx, resolution=1024):
    img_size = 256
    img = np.zeros([img_size, img_size, 3])

    canvas = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
    stickwidth = 1

    # with open(file_path) as f:
    #     data = json.load(f)['people'][0]
    #     keypoint = np.array(data['pose_keypoints_2d']).reshape(-1, 3)

    keypoint[:,:2] *= img_size / resolution

    # os.makedirs('./results/canvas', exist_ok=True)
    threshold = 0.05

    encoding_list = []
    for i in range(len(limbSeq)):
        X = keypoint[limbSeq[i], 0]
        Y = keypoint[limbSeq[i], 1]
        p = keypoint[limbSeq[i], 2]
        # cur_canvas = canvas.copy()
        if p[0] < threshold or p[1] < threshold:
            joint_dist = np.zeros_like(canvas[:, :, 0:1])
        else:
            mX = np.mean(X)
            mY = np.mean(Y)

            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))

            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, line_colors[i])
            # cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            # canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            joint = np.zeros_like(canvas[:, :, 0:1])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint_dist = cv2.distanceTransform(255 - joint, cv2.DIST_L1, 3)
            joint_dist = np.clip((joint_dist / 0.1), 0, 255).astype(np.uint8)
            joint_dist = np.expand_dims(255 - joint_dist, axis=-1)
        # cv2.imwrite(f'./results/canvas/joints_{i}.png', joint_dist)
        encoding_list.append(joint_dist)

    for i in range(keypoint.shape[0]):
        x, y, p = keypoint[i, 0:3]
        if p < threshold:
            continue
        cv2.circle(canvas, (int(x), int(y)), 2, point_colors[i], thickness=-1)

    # cv2.imwrite(f'./results/canvas/canvas_{idx}.png', canvas)
    encoding_list.append(canvas)

    pose_encoding = np.concatenate(encoding_list, axis=2) / 255

    return pose_encoding

# ----------------------------------------------------------------------------------------

class apose_train_dataset(Dataset):
    def __init__(self, root_dir, name, idx_list, view_num, img_size=(1024, 1024)):
        self.mask_dir = f'{root_dir}/mask_mat/{name}/'
        self.joints_dir = f'{root_dir}/body_hand_face_joints/{name}/'
        self.params_dir = f'{root_dir}/params/{name}/'

        self.idx_list = idx_list
        self.view_num = view_num

        self.img_size = img_size

        with open(self.params_dir + 'param.pkl', 'rb') as file:
            self.params = pkl.load(file, encoding='iso-8859-1')

        pass

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):

        masks, body_joints, face_joints, pose_encodings = [], [], [], []
        for i in range(0, self.view_num):
            target_idx = self.idx_list[item] + i * len(self.idx_list)

            mask = cv2.imread(self.mask_dir + f'{str(target_idx).zfill(4)}.png')[:,:,0]
            mask = cv2.resize(mask, self.img_size)
            mask = np.expand_dims(mask, 0)

            body, face = load_joints_from_openpose(self.joints_dir + 'json/' + str(target_idx).zfill(4) + '_keypoints.json')
            body = post_process_invisible_joints(body)

            pose_encoding = np.load(self.joints_dir + 'encoding/' + str(target_idx).zfill(4) + '.npz')['arr_0']

            masks.append(mask)
            body_joints.append(body)
            pose_encodings.append(pose_encoding)
            # face_joints.append(face)

        index = item # self.idx_list[item] - 1
        pose = self.params['poses'][index: index + self.view_num * len(self.idx_list): len(self.idx_list)]
        trans = self.params['trans'][index: index + self.view_num * len(self.idx_list): len(self.idx_list)]
        shape = self.params['betas'][index: index + self.view_num * len(self.idx_list): len(self.idx_list)]
        shape = np.mean(shape, axis=0, keepdims=True)

        masks = numpy2tensor(np.array(masks) / 255.0).contiguous()  # .permute(0, 3, 1, 2)
        body_joints = numpy2tensor(np.array(body_joints)).contiguous()
        pose_encodings = numpy2tensor(np.array(pose_encodings)).contiguous()
        pose_encodings = pose_encodings.permute(0, 3, 1, 2)
        # face_joints = numpy2tensor(np.array(face_joints)).contiguous()

        pose = numpy2tensor(np.array(pose)).contiguous()
        trans = numpy2tensor(np.array(trans)).contiguous()
        shape = numpy2tensor(np.array(shape)).contiguous()

        # print(item)
        sample = {
            'masks':masks,
            'body_joints':body_joints,
            'pose_encodings':pose_encodings,
        }
        ground_truth = {
            'body_joints': body_joints,
            'pose': pose,
            'trans': trans,
            'shape': shape,
        }

        return item, sample, ground_truth

class apose_test_dataset(Dataset):
    def __init__(self, root_dir, name, length, idx_list, view_num, img_size=(1024, 1024)):
        self.mask_dir = f'{root_dir}/mask_mat/{name}/'
        self.joints_dir = f'{root_dir}/2d_joints/{name}/'

        self.idx_list = idx_list
        self.view_num = view_num
        self.img_size = img_size
        # self.normalized_body_keypoints, self.normalized_face_keypoints = \
        #     load_normalized_joints_from_mmpose(root_dir, name, length, resolution=(1024, 1024))
        # self.body_keypoints, self.face_keypoints = load_joints_from_mmpose(root_dir, name, length)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):

        masks, body_joints, face_joints, pose_encodings = [], [], [], []
        for i in range(0, self.view_num):
            target_idx = self.idx_list[item] + i * len(self.idx_list)
            # mask_path = self.dir + 'frames_mat/' + self.name + '/' + str(target_idx).zfill(4) + '.png'
            mask = cv2.imread(self.mask_dir + f'{str(target_idx).zfill(4)}.png')[:,:,0]
            mask = cv2.resize(mask, self.img_size)
            mask = np.expand_dims(mask, 0)

            body, face = load_normalized_joints_from_openpose(self.joints_dir + 'json/' + str(target_idx).zfill(4)
                                                              + '_keypoints.json', resolution=self.img_size)

            # body, face = self.body_keypoints[target_idx - 1], self.face_keypoints[target_idx - 1]
            # body = post_process_invisible_joints(body)
            _body, _face = load_joints_from_openpose(self.joints_dir + 'json/' + str(target_idx).zfill(4) + '_keypoints.json')

            pose_encoding = generate_pose_encoding(_body, target_idx, resolution=self.img_size[0])
            # pose_encoding = generate_pose_encoding(self._body_keypoints[target_idx - 1], target_idx)

            masks.append(mask)
            body_joints.append(body)
            face_joints.append(face)
            pose_encodings.append(pose_encoding)


        masks = numpy2tensor(np.array(masks) / 255.0).contiguous()  # .permute(0, 3, 1, 2)
        body_joints = numpy2tensor(np.array(body_joints)).contiguous()
        # face_joints = torch.from_numpy(np.array(face_joints).astype(np.float32)).contiguous()
        pose_encodings = numpy2tensor(np.array(pose_encodings)).contiguous()
        pose_encodings = pose_encodings.permute(0, 3, 1, 2)

        sample = {
            'masks':masks,
            'body_joints':body_joints,
            'pose_encodings':pose_encodings
        }

        return item, sample # masks, body_joints, pose_encodings

# ----------------------------------------------------------------------------------------

class diff_optimizer_dataset(Dataset):
    def __init__(self, root_dir, idx_list, name, length=0, view_num=8, img_size=(1024, 1024), load_normal=False):

        self.root_dir = root_dir
        self.idx_list = idx_list
        self.view_num = view_num
        self.img_size = img_size
        self.name = name

        self.load_normal = load_normal
        # with open(dir + 'vertices/' + name + '/frame_data.pkl', 'rb') as file:
        #     self.vertices = pkl.load(file, encoding='latin1')

        # self.body_keypoints = np.load(self.dir + '2d_joints/' + self.name + '/body25_keypoints2d.npy')
        # self.face_keypoints = np.load(self.dir + '2d_joints/' + self.name + '/face70_keypoints2d.npy')
        # self.body_keypoints, self.face_keypoints = load_normalized_joints_from_mmpose(dir, name, length, resolution=img_size)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        # item = 0
        imgs, silhouette_imgs, body_joints, face_joints = [], [], [], []
        # vertices = []
        # interval_num = (int)( (int)(self.name.split('_')[-1]) / self.view_num)
        for i in range(0, self.view_num):
            target_idx = self.idx_list[item] + i * len(self.idx_list)
            img_path = f'{self.root_dir}/frames_mat/{self.name}/{str( target_idx ).zfill(4)}.png'
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            imgs.append(img)

            silhouette_img_path = f'{self.root_dir}/mask_mat/{self.name}/{str( target_idx ).zfill(4)}.png'
            silhouette_img = cv2.imread(silhouette_img_path)
            # silhouette_img[np.where(silhouette_img > 128)] = 255
            # silhouette_img[np.where(silhouette_img < 128)] = 0
            _, silhouette_img = cv2.threshold(silhouette_img, 128, 255, cv2.THRESH_BINARY)

            silhouette_img = cv2.resize(silhouette_img, self.img_size)
            silhouette_imgs.append(silhouette_img[:,:,0])


            body, face = load_normalized_joints_from_openpose(
                f'{self.root_dir}/2d_joints/{self.name}/json/{str( target_idx ).zfill(4)}_keypoints.json',
                resolution=self.img_size)
            # body, face = self.body_keypoints[target_idx - 1], self.face_keypoints[target_idx - 1]

            body_joints.append(body)
            face_joints.append(face)

        imgs = numpy2tensor(np.array(imgs) / 255.0).contiguous()  # .permute(0, 3, 1, 2)
        silhouette_imgs = numpy2tensor(np.array(silhouette_imgs) / 255.0).contiguous()
        # vertices = numpy2tensor(vertices.astype(np.float32) ).contiguous()

        body_joints = numpy2tensor(np.array(body_joints)).contiguous()
        face_joints = numpy2tensor(np.array(face_joints)).contiguous()

        ground_truth = {
            'gt_images': imgs,
            'gt_sil_images': silhouette_imgs,
            'body_joints2d_gt': body_joints,
            'face_joints2d_gt': face_joints
        }

        return item, ground_truth # imgs, silhouette_imgs, body_joints, face_joints # , vertices

class dynamic_offsets_dataset(Dataset):
    def __init__(self, root_dir, idx_list, name, tmp_path, img_size=(1024, 1024), use_normal=False):
        self.root_dir = root_dir
        self.idx_list = idx_list
        self.img_size = img_size
        self.name = name
        self.use_normal = use_normal
        self.tmp_path = tmp_path

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        img_path = f'{self.root_dir}/frames_mat/{self.name}/{str(self.idx_list[item]).zfill(4)}.png'

        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = numpy2tensor(img / 255.0)

        silhouette_img_path = f'{self.root_dir}/mask_mat/{self.name}/{str(self.idx_list[item]).zfill(4)}.png'

        silhouette = cv2.imread(silhouette_img_path)
        silhouette = cv2.resize(silhouette, self.img_size)
        # silhouette[np.where(silhouette > 128)] = 255
        # silhouette[np.where(silhouette < 128)] = 0
        _, silhouette = cv2.threshold(silhouette, 128, 255, cv2.THRESH_BINARY)
        
        silhouette_img = numpy2tensor(silhouette[:, :, 0] / 255.0)

        naked_vertice_uv = np.load(f'{self.tmp_path}/naked_vertice_uv/{self.name}/{str(self.idx_list[item]).zfill(4)}.npy')
        naked_vertice_uv = np.transpose(naked_vertice_uv, [2, 0, 1])
        naked_vertice_uv = numpy2tensor(naked_vertice_uv)


        # load_normal = False
        normal = []
        if self.use_normal == True:

            normal = np.load(f'{self.root_dir}/normal/{self.name}/{str(self.idx_list[item]).zfill(4)}_front.npz')['arr_0']

            normal = cv2.resize(normal, self.img_size)
            normal = cv2.medianBlur(normal, 5)

            a = np.sqrt(np.sum(np.square(normal), axis=-1))
            b = np.tile(np.expand_dims(a, axis=-1), (1, 1, 3))
            normal = normal / b

            normal[np.where(silhouette[:, :, 0] == 0.0)] = [0.0, 0.0, 0.0]
            # normal_img = (normal[:, :, ::-1] * 0.5 + 0.5) * 255
            # cv2.imwrite('./results/normal/{}.png'.format(self.idx_list[item]), normal_img)

            # normal = np.transpose(normal, [2, 0, 1])
            normal = numpy2tensor(normal)


        sample = {
            'naked_vertice_uv':naked_vertice_uv,
        }
        ground_truth = {
            'gt_images': img,
            'gt_sil_images': silhouette_img,
            'gt_normal_images': normal
        }

        return item, sample, ground_truth



# ----------------------------------------------------------------------------------------

class neural_rendering_train_dataset(Dataset):
    def __init__(self, img_dir, uv_dir, ref_dir, idx_list, name, img_size=(1024, 1024)):
        self.idx_list = idx_list
        self.img_dir, self.uv_dir, self.ref_dir = img_dir, uv_dir, ref_dir
        self.name = name
        self.img_size = img_size


    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        shape = self.img_size  # (512,512)
        # img = Image.open(os.path.join(self.dir, 'frame/'+self.idx_list[idx]+'.png'), 'r')
        img = cv2.imread(f'{self.img_dir}/{self.name}/{self.idx_list[idx]}.png')
        img = cv2.resize(img, shape)  # interpolation=cv2.INTER_AREA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        uv_map = np.load(f'{self.uv_dir}/{self.name}/{self.idx_list[idx]}.npz')['arr_0'] #
        # uv_map = cv2.resize(uv_map, shape) # better not resize


        ref_img = cv2.imread(f'{self.ref_dir}/{self.name}/ref_{self.idx_list[idx]}.png')
        ref_img = cv2.resize(ref_img, self.img_size) # interpolation=cv2.INTER_AREA
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        ref_img_mask = cv2.imread(f'{self.ref_dir}/{self.name}/mask_{self.idx_list[idx]}.png')[:,:,:1] # / 255
        ref_img = np.concatenate([ref_img, ref_img_mask], axis=2)

        img = numpy2tensor(img / 255.0).permute(2, 0, 1).contiguous()
        uv_map = numpy2tensor(uv_map).contiguous()

        ref_img = numpy2tensor(ref_img / 255.0).permute(2, 0, 1).contiguous()
        return img, uv_map, ref_img

class neural_rendering_test_dataset(Dataset):
    def __init__(self, img_dir, uv_dir, ref_dir, idx_list, name, img_size=(1024, 1024)):
        self.idx_list = idx_list

        self.img_dir, self.uv_dir, self.ref_dir = img_dir, uv_dir, ref_dir  # test_uvs
        self.name = name

        self.img_size = img_size


    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        uv_map = np.load(f'{self.uv_dir}/{self.name}/{self.idx_list[idx]}.npz')['arr_0'] # uv_map = uv_map[:,:,:2] / 255
        uv_map = numpy2tensor(uv_map)


        ref_img = cv2.imread(f'{self.ref_dir}/{self.name}/ref_{self.idx_list[idx]}.png')  # '0001'
        ref_img = cv2.resize(ref_img, self.img_size)  # interpolation=cv2.INTER_AREA
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        ref_img_mask = cv2.imread(f'{self.ref_dir}/{self.name}/mask_{self.idx_list[idx]}.png')[:,:,:1] # / 255
        ref_img = np.concatenate([ref_img, ref_img_mask], axis=2)

        ref_img = numpy2tensor(ref_img / 255.0).permute(2, 0, 1).contiguous()


        return uv_map, ref_img, self.idx_list[idx]

class neural_rendering_finetune_dataset(Dataset):
    def __init__(self, img_dir, uv_dir, idx_list, name, img_size=(1024, 1024)):
        self.idx_list = idx_list

        self.uv_dir, self.img_dir = uv_dir, img_dir
        self.img_size = img_size
        self.name = name
        # self.img_list = sorted(os.listdir(self.img_dir))  # [:len(self.v_list)]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        shape = self.img_size

        img = cv2.imread(f'{self.img_dir}/{self.name}/render_{self.idx_list[item]}.png')  # self.img_list[item]
        # img = cv2.imread(os.path.join(self.img_dir, self.idx_list[item] + '.png'))

        img = cv2.resize(img, shape)  # interpolation=cv2.INTER_AREA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        uv_map = np.load(f'{self.uv_dir}/{self.name}/{self.idx_list[item]}.npz')['arr_0']
        # uv_map = cv2.resize(uv_map, shape, interpolation=cv2.INTER_NEAREST)

        img = numpy2tensor(img / 255.0).permute(2, 0, 1).contiguous()
        uv_map = numpy2tensor(uv_map).contiguous()
        # face_mask = torch.from_numpy(face_mask.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        return img, uv_map,



if __name__ == '__main__':
    pass
    # select_keyframes('/mnt/8T/zh/vrc/', 'Body2D_2030_243')