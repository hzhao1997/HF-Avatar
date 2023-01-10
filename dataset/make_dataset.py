import json
import math
import pickle as pkl
import tqdm
import cv2
import numpy as np
import os
import torch
import sys


from utils.general import numpy2tensor
from torch.utils.data import Dataset

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


def load_joints_from_openpose(joints_path,
                              normalized: bool = False,
                              img_size=(1024, 1024)):
    with open(joints_path) as f:
        data = json.load(f)['people'][0]

        body = np.array(data['pose_keypoints_2d']).reshape(-1, 3)
        body[:, 2] /= np.expand_dims(np.mean(body[:, 2][body[:, 2] > 0.1]), -1)

        face = np.array(data['face_keypoints_2d']).reshape(-1, 3)

    body = post_process_invisible_joints(body)

    if normalized:
        body = body * \
               np.array([2. / img_size[1], -2. / img_size[0], 1.]) + \
               np.array([-1., 1., 0.])
        body[:, 0] *= 1. * img_size[1] / img_size[0]

        face = face * \
               np.array([2. / img_size[1], -2. / img_size[0], 1.]) + \
               np.array([-1., 1., 0.])
        face[:, 0] *= 1. * img_size[1] / img_size[0]

    return body, face


def load_all_silhouettes(silhouettes_dir, length=0, img_size=(1024, 1024)):
    silhouette_imgs = []
    for idx in range(1, length + 1):
        silhouette_img = \
            cv2.imread(f'{silhouettes_dir}/{str(idx).zfill(4)}.png')
        # silhouette_img[np.where(silhouette_img > 128)] = 255
        # silhouette_img[np.where(silhouette_img < 128)] = 0
        silhouette_img = cv2.resize(silhouette_img, img_size)
        silhouette_imgs.append(silhouette_img[:, :, 0])
    silhouette_imgs = np.array(silhouette_imgs)

    return silhouette_imgs


def load_all_joints_from_openpose(joints_dir,
                                  length=0,
                                  img_size=(1024, 1024)):
    body_joints = []
    face_joints = []

    # length = length - length % 8
    for idx in range(1, length + 1):
        body, face = load_joints_from_openpose(
            f'{joints_dir}' #
            f'json/{str(idx).zfill(4)}_keypoints.json',
            normalized=True,
            img_size=img_size)
        # body = post_process_invisible_joints(body)
        body_joints.append(body)
        face_joints.append(face)

    body_joints = numpy2tensor(np.array(body_joints)).contiguous()
    face_joints = numpy2tensor(np.array(face_joints)).contiguous()

    return body_joints, face_joints


LIMB_CONN = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [8, 12],
    [12, 13],
    [13, 14],
    [0, 15],
    [0, 16],
    [15, 17],
    [16, 18],
    [14, 19],
    [19, 20],
    [14, 21],
    [11, 22],
    [22, 23],
    [11, 24],
]
LINE_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
               [255, 0, 255], [255, 0, 170], [255, 0, 85], [170, 0, 0],
               [170, 85, 0], [170, 170, 0], [170, 255, 0], [170, 0, 255],
               [170, 0, 170], [170, 0, 85], [85, 0, 0], [85, 85, 0],
               [85, 170, 0], [85, 255, 0], [85, 0, 255], [85, 0, 170],
               [85, 0, 85], [0, 255, 0], [0, 255, 85], [0, 255, 170]]
POINT_COLORS = [[255, 0, 0], [255, 0, 0], [255, 85, 0], [255, 170, 0],
                [255, 255, 0], [255, 0, 255], [255, 0, 170], [255, 0, 85],
                [170, 0, 0], [170, 85, 0], [170, 170, 0], [170, 255, 0],
                [170, 0, 255], [170, 0, 170], [170, 0, 85], [85, 0, 0],
                [85, 85, 0], [85, 170, 0], [85, 255, 0], [85, 0, 255],
                [85, 0, 170], [85, 0, 85], [0, 255, 0], [0, 255, 85],
                [0, 255, 170]]


def generate_pose_encoding(keypoint, img_size = 1024, canvas_size = 256):

    canvas = np.zeros((canvas_size, canvas_size, 3)).astype(np.uint8)
    stickwidth = 1

    keypoint[:, :2] *= canvas_size / img_size

    # os.makedirs('./results/canvas', exist_ok=True)
    threshold = 0.05

    encoding_list = []
    for i in range(len(LIMB_CONN)):
        X = keypoint[LIMB_CONN[i], 0]
        Y = keypoint[LIMB_CONN[i], 1]
        p = keypoint[LIMB_CONN[i], 2]
        # cur_canvas = canvas.copy()
        if p[0] < threshold or p[1] < threshold:
            joint_dist = np.zeros_like(canvas[:, :, 0:1])
        else:
            X_mean = np.mean(X)
            Y_mean = np.mean(Y)

            length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))

            polygon = cv2.ellipse2Poly(
                (int(X_mean), int(Y_mean)),
                (int(length / 2), stickwidth),
                int(angle),
                0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, LINE_COLORS[i])
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
        cv2.circle(canvas, (int(x), int(y)), 2, POINT_COLORS[i], thickness=-1)

    # cv2.imwrite(f'./results/canvas/canvas_{idx}.png', canvas)
    encoding_list.append(canvas)

    pose_encoding = np.concatenate(encoding_list, axis=2) / 255

    return pose_encoding

class diff_optimizer_dataset(Dataset):

    def __init__(self,
                 root_dir,
                 idx_list,
                 name,
                 length=0,
                 view_num=8,
                 img_size=(1024, 1024),
                 pose_img_size=(1024, 1024),
                 load_normal=False):
        self.img_dir = f'{root_dir}/frames_mat/{name}/'
        self.mask_dir = f'{root_dir}/mask_mat/{name}/'
        self.joints_dir = f'{root_dir}/2d_joints/{name}/'

        self.root_dir = root_dir
        self.idx_list = idx_list

        self.length = length
        self.view_num = view_num
        self.interval_num = int(self.length / self.view_num)

        self.img_size = img_size
        self.pose_img_size = pose_img_size
        self.name = name

        self.load_normal = load_normal


    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        # item = 0
        images, sil_images, body_joints, face_joints = [], [], [], []

        for i in range(0, self.view_num):
            # target_idx = self.idx_list[item] + i * len(self.idx_list)
            target_idx = self.idx_list[item] + i * self.interval_num
            image_path = f'{self.img_dir}{str(target_idx).zfill(4)}.png'
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            images.append(image)

            sil_image_path = f'{self.mask_dir}{str(target_idx).zfill(4)}.png'
            sil_image = cv2.imread(sil_image_path)
            # sil_image[np.where(sil_image > 128)] = 255
            # sil_image[np.where(sil_image < 128)] = 0
            _, sil_image = \
                cv2.threshold(sil_image, 128, 255, cv2.THRESH_BINARY)

            sil_image = cv2.resize(sil_image, self.img_size)
            sil_images.append(sil_image[:, :, 0])

            body, face = load_joints_from_openpose(
                f'{self.joints_dir}json/'
                f'{str(target_idx).zfill(4)}_keypoints.json',
                normalized=True,
                img_size=self.pose_img_size)

            body_joints.append(body)
            face_joints.append(face)

        images = numpy2tensor(np.array(images) / 255.0).contiguous()
        sil_images = numpy2tensor(np.array(sil_images) / 255.0).contiguous()

        body_joints = numpy2tensor(np.array(body_joints)).contiguous()
        face_joints = numpy2tensor(np.array(face_joints)).contiguous()

        ground_truth = {
            'gt_images': images,
            'gt_sil_images': sil_images,
            'body_joints2d_gt': body_joints,
            'face_joints2d_gt': face_joints
        }

        return self.idx_list[item] - 1, ground_truth


def load_batch_data(root_dir,
                    idx_list,
                    name,
                    length=0,
                    view_num=8,
                    img_size=(1024, 1024),
                    pose_img_size=(1024, 1024),
                    load_normal=False,
                    item=None
                    ):

    img_dir = f'{root_dir}/frames_mat/{name}/'
    mask_dir = f'{root_dir}/mask_mat/{name}/'
    joints_dir = f'{root_dir}/2d_joints/{name}/'

    interval_num = int(length / view_num)

    images, sil_images, body_joints, face_joints = [], [], [], []

    for i in range(0, view_num):
        # target_idx = self.idx_list[item] + i * len(self.idx_list)
        target_idx = idx_list[item] + i * interval_num
        image_path = f'{img_dir}{str(target_idx).zfill(4)}.png'
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image)

        sil_image_path = f'{mask_dir}{str(target_idx).zfill(4)}.png'
        sil_image = cv2.imread(sil_image_path)
        # sil_image[np.where(sil_image > 128)] = 255
        # sil_image[np.where(sil_image < 128)] = 0
        _, sil_image = \
            cv2.threshold(sil_image, 128, 255, cv2.THRESH_BINARY)

        sil_image = cv2.resize(sil_image, img_size)
        sil_images.append(sil_image[:, :, 0])

        body, face = load_joints_from_openpose(
            f'{joints_dir}json/'
            f'{str(target_idx).zfill(4)}_keypoints.json',
            normalized=True,
            img_size=pose_img_size)


        body_joints.append(body)
        face_joints.append(face)

    images = numpy2tensor(np.array(images) / 255.0).contiguous()
    sil_images = numpy2tensor(np.array(sil_images) / 255.0).contiguous()

    body_joints = numpy2tensor(np.array(body_joints)).contiguous()
    face_joints = numpy2tensor(np.array(face_joints)).contiguous()

    ground_truth = {
        'gt_images': images,
        'gt_sil_images': sil_images,
        'body_joints2d_gt': body_joints,
        'face_joints2d_gt': face_joints
    }

    return idx_list[item] - 1, ground_truth

def load_all_data(root_dir,
                  name,
                  length=0,
                  view_num=8,
                  img_size=(1024, 1024),
                  pose_img_size=(1024, 1024),
                  load_normal=False,
                  ):

    img_dir = f'{root_dir}/frames_mat/{name}/'
    mask_dir = f'{root_dir}/mask_mat/{name}/'
    joints_dir = f'{root_dir}/2d_joints/{name}/'

    interval_num = int(length / view_num)

    images, sil_images, body_joints, face_joints = [], [], [], []

    for i in tqdm.tqdm(range(1, length + 1)):
        # print(i)
        # target_idx = self.idx_list[item] + i * len(self.idx_list)
        target_idx = i # idx_list[item] + i * interval_num
        image_path = f'{img_dir}{str(target_idx).zfill(4)}.png'
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image)

        sil_image_path = f'{mask_dir}{str(target_idx).zfill(4)}.png'
        sil_image = cv2.imread(sil_image_path)
        # sil_image[np.where(sil_image > 128)] = 255
        # sil_image[np.where(sil_image < 128)] = 0
        _, sil_image = \
            cv2.threshold(sil_image, 128, 255, cv2.THRESH_BINARY)

        sil_image = cv2.resize(sil_image, img_size)
        sil_images.append(sil_image[:, :, 0])

        body, face = load_joints_from_openpose(
            f'{joints_dir}json/'
            f'{str(target_idx).zfill(4)}_keypoints.json',
            normalized=True,
            img_size=pose_img_size)


        body_joints.append(body)
        face_joints.append(face)

    images = numpy2tensor(np.array(images) / 255.0).contiguous()
    sil_images = numpy2tensor(np.array(sil_images) / 255.0).contiguous()

    body_joints = numpy2tensor(np.array(body_joints)).contiguous()
    face_joints = numpy2tensor(np.array(face_joints)).contiguous()

    ground_truth = {
        'gt_images': images,
        'gt_sil_images': sil_images,
        'body_joints2d_gt': body_joints,
        'face_joints2d_gt': face_joints
    }

    return ground_truth


class dynamic_offsets_dataset(Dataset):
    def __init__(self, root_dir, idx_list, name, tmp_path,
                 img_size=(1024, 1024), use_normal=False):
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

        naked_vertice_uv = np.load(
            f'{self.tmp_path}/naked_vertice_uv/{self.name}/{str(self.idx_list[item]).zfill(4)}.npy')
        naked_vertice_uv = np.transpose(naked_vertice_uv, [2, 0, 1])
        naked_vertice_uv = numpy2tensor(naked_vertice_uv)

        # load_normal = False
        normal = []
        if self.use_normal == True:
            normal = np.load(
                f'{self.root_dir}/normal/{self.name}/{str(self.idx_list[item]).zfill(4)}_front.npz')[
                'arr_0']

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
            'naked_vertice_uv': naked_vertice_uv,
        }
        ground_truth = {
            'gt_images': img,
            'gt_sil_images': silhouette_img,
            'gt_normal_images': normal
        }

        return item, sample, ground_truth


# ----------------------------

class neural_rendering_train_dataset(Dataset):
    def __init__(self, img_dir, uv_dir, ref_dir, idx_list, name,
                img_size=(1024, 1024)):
        self.idx_list = idx_list
        self.img_dir, self.uv_dir, self.ref_dir = img_dir, uv_dir, ref_dir
        self.name = name
        self.img_size = img_size


    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):

        img = cv2.imread(f'{self.img_dir}/{self.name}/{self.idx_list[idx]}.png')
        img = cv2.resize(img, self.img_size)  # interpolation=cv2.INTER_AREA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        uv_map = np.load(f'{self.uv_dir}/{self.name}/{self.idx_list[idx]}.npz')['arr_0'] #
        # uv_map = cv2.resize(uv_map, shape) # better not resize

        ref_img = cv2.imread(f'{self.ref_dir}/{self.name}/ref_{self.idx_list[idx]}.png')
        ref_img = cv2.resize(ref_img, self.img_size) # interpolation=cv2.INTER_AREA
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        ref_img_mask = cv2.imread(f'{self.ref_dir}/{self.name}/mask_{self.idx_list[idx]}.png')[:,:,:1]
        ref_img = np.concatenate([ref_img, ref_img_mask], axis=2)

        img = numpy2tensor(img / 255.0).permute(2, 0, 1).contiguous()
        uv_map = numpy2tensor(uv_map).contiguous()

        ref_img = numpy2tensor(ref_img / 255.0).permute(2, 0, 1).contiguous()

        return img, uv_map, ref_img

class neural_rendering_test_dataset(Dataset):
    def __init__(self, img_dir, uv_dir, ref_dir, idx_list, name,
                img_size=(1024, 1024)):
        self.idx_list = idx_list

        self.img_dir, self.uv_dir, self.ref_dir = img_dir, uv_dir, ref_dir  # test_uvs
        self.name = name

        self.img_size = img_size

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        uv_map = np.load(f'{self.uv_dir}/{self.name}/{self.idx_list[idx]}.npz')['arr_0']
        uv_map = numpy2tensor(uv_map)

        ref_img = cv2.imread(f'{self.ref_dir}/{self.name}/ref_{self.idx_list[idx]}.png')
        ref_img = cv2.resize(ref_img, self.img_size)  # interpolation=cv2.INTER_AREA
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        ref_img_mask = cv2.imread(f'{self.ref_dir}/{self.name}/mask_{self.idx_list[idx]}.png')[:,:,:1]
        ref_img = np.concatenate([ref_img, ref_img_mask], axis=2)

        ref_img = numpy2tensor(ref_img / 255.0).permute(2, 0, 1).contiguous()

        return uv_map, ref_img, self.idx_list[idx]


class neural_rendering_finetune_dataset(Dataset):
    def __init__(self, img_dir, uv_dir, idx_list, name, img_size=(1024, 1024)):
        self.idx_list = idx_list

        self.uv_dir, self.img_dir = uv_dir, img_dir
        self.img_size = img_size
        self.name = name

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        shape = self.img_size

        img = cv2.imread(f'{self.img_dir}/{self.name}/render_{self.idx_list[item]}.png')

        img = cv2.resize(img, shape)  # interpolation=cv2.INTER_AREA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        uv_map = np.load(f'{self.uv_dir}/{self.name}/{self.idx_list[item]}.npz')['arr_0']

        img = numpy2tensor(img / 255.0).permute(2, 0, 1).contiguous()
        uv_map = numpy2tensor(uv_map).contiguous()

        return img, uv_map




if __name__ == '__main__':

    other_path = './results/'
    pose_save = np.load(f'{other_path}pose.npy')
    betas_save = np.load(f'{other_path}betas.npy')
    trans_save = np.load(f'{other_path}trans.npy')

    load_path = './results/diff_optiming/Body2D_2070_380/'

    pose = np.load(f'{load_path}pose.npy')
    betas = np.load(f'{load_path}betas.npy')
    trans= np.load(f'{load_path}trans.npy')

    sys.exit(0)

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    root_dir = '/mnt/8T/zh/vrc'
    name = 'Body2D_2070_380'

    length = len(os.listdir(f'{root_dir}/frames_mat/{name}'))
    length = length - length % 8
    ground_truth = load_all_data(root_dir=root_dir,
                                 name=name,
                                 length=length)
    for _, key in enumerate(ground_truth):
        ground_truth[key] = ground_truth[key].to(device)
    # apose_dataset = apose_train_dataset()
    pass