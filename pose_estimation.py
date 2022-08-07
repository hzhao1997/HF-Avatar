import numpy as np
import cv2
import os
import time
import pickle as pkl
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from geometry_model.network import apose_estimator_network
from geometry_model.lib import rodrigues, rodrigues_v
from geometry_model.loss import Joint2dLoss
from dataset.make_dataset import apose_train_dataset, apose_test_dataset
from utils.general import setup_seed, tensor2numpy, numpy2tensor

class AposeEstimator():
    def __init__(self, device, root_dir):
        self.device = device
        self.root_dir = root_dir

    def build_network(self, mode):
        self.model = apose_estimator_network(self.device, norm='batch', pose_encoding='2')
        self.model = self.model.to(self.device)
        if mode == 'train':
            self.model.train()
        elif mode == 'test':
            self.model.eval()

    def build_optimizer(self):
        self.optimizer = Adam([
            {'params': self.model.parameters(), 'lr': 1e-4},
        ])

    def build_dataloader(self, name, mode='train', length=0):

        if mode == 'train':
            self.view_num = 32
            self.length = length - length % self.view_num
            self.num = (int)(self.length / self.view_num)
            self.idx_list = [i for i in range(1, self.num + 1)]

            dataset = apose_train_dataset(root_dir=self.root_dir, name=name, idx_list=self.idx_list, view_num=self.view_num)
            self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)
            # dataset = Apose_train_dataset(root_dir=self.root_dir, name=self.name, idx_list=self.idx_list, view_num=self.view_num)
            # self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
        elif mode == 'test':
            self.view_num = 8
            self.length = length - length % self.view_num
            self.num = (int)(self.length / self.view_num)
            self.idx_list = [i for i in range(1, self.num + 1)]

            dataset = apose_test_dataset(root_dir=self.root_dir, name=name, length=self.length,
                                         idx_list=self.idx_list, view_num=self.view_num) # Apose_test_dataset
            self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    def build_loss_function(self):
        self.L1Loss = nn.L1Loss()
        self.MSELoss = nn.MSELoss()
        self.Joint2dLoss = Joint2dLoss()

    def forward(self, item, model_input, ground_truth):
        model_output = self.model(model_input['masks'][0],model_input['body_joints'] [0])
        # print(pose_pred.shape, pose_gt.shape)
        pose_loss = self.MSELoss(model_output['pose'], ground_truth['pose'])
        trans_loss = self.L1Loss(model_output['trans'], ground_truth['trans'])
        betas_loss = self.L1Loss(model_output['betas'], ground_truth['betas'])
        joints2d_loss = self.Joint2dLoss(model_output['perspective_body_joints_h'], ground_truth['body_joints'])


        loss = 10 * joints2d_loss + 10 * pose_loss + 1 * trans_loss + 0 * betas_loss

        print(f"step {item} joints2d_loss:{joints2d_loss} pose_loss: {pose_loss} trans_loss:{trans_loss} betas_loss:{betas_loss}")
        return loss, model_output['v']

    def train(self, name_list, outer_epoch=500, inner_epoch=1):
        # epoch = 500
        torch.set_grad_enabled(True)
        self.build_network(mode='train')
        self.build_optimizer()
        self.build_loss_function()

        # self.load_checkpoints()
        for epoch_idx in range(outer_epoch):
            print('--------------- Epoch: {} ---------------'.format(epoch_idx))
            for name in name_list:
                print(name)
                self.build_dataloader(name, mode='train')
                for _ in range(inner_epoch):
                    for step, (item, model_input, ground_truth) in enumerate(self.dataloader):
                        for key in model_input.keys():
                            model_input[key] = model_input[key].to(self.device)
                        for key in ground_truth.keys():
                            ground_truth[key] = ground_truth[key].to(self.device)

                        self.optimizer.zero_grad()

                        pose = rodrigues(pose.reshape([-1, 1, 3])).reshape([-1, 24, 3, 3]).unsqueeze(0) # axisangle to rotation matrix

                        loss, v = self.forward(item, model_input, ground_truth)

                        loss.backward()
                        self.optimizer.step()

            self.save_checkpoints()

    def test(self, name, save_path):
        length = len(os.listdir(self.root_dir + f'/frames_mat/{name}'))
        length = length - length % 8

        torch.set_grad_enabled(False)
        self.build_network(mode='test')
        self.build_dataloader(name, mode='test', length=length)

        self.load_checkpoints(path='./checkpoints/a_pose_estimation_h.pt')
        self.build_loss_function()
        pose_pred_save = np.zeros([self.length, 24, 3])
        pose_pred_matrix_save = np.zeros([self.length, 24, 3, 3])
        trans_pred_save = np.zeros([self.length, 3])
        betas_pred_save = np.zeros([1, 10])


        os.makedirs(save_path, exist_ok=True)


        with torch.no_grad():
            for step, (item, model_input) in enumerate(self.dataloader):
                for key in model_input.keys():
                    model_input[key] = model_input[key].to(self.device)

                model_output = self.model(model_input['masks'][0], model_input['pose_encodings'][0])
                # pose_pred_matrix, trans_pred, betas_pred, perspective_body_joints_h, v
                pose_pred_matrix = tensor2numpy(model_output['pose'][0])
                pose_pred_matrix_save[step:step + self.view_num * self.num:self.num] = pose_pred_matrix

                # pose_pred_2 = rodrigues_v(pose_pred_matrix.reshape([-1, 3, 3]))[0].reshape([self.view_num, 24, 3]).detach().cpu().numpy()
                pose_pred_matrix = pose_pred_matrix.reshape([-1, 3, 3])
                pose_pred = np.array([cv2.Rodrigues(pose_pred_matrix[i])[0] for i in range(pose_pred_matrix.shape[0]) ]).squeeze()
                pose_pred = pose_pred.reshape([-1, 24, 3])

                pose_pred_save[step:step + self.view_num * self.num:self.num] = pose_pred
                trans_pred_save[step:step + self.view_num * self.num:self.num] = tensor2numpy(model_output['trans'][0])
                betas_pred_save = tensor2numpy(model_output['betas'][0])

                # just for visualization
                if False:
                    for idx in range(model_output['v'].shape[1]):
                        # self.model.smpl.smpl.write_obj(v[0, idx], save_path + f'{str(step + idx * self.num).zfill(4)}.obj')
                        _pad = torch.zeros_like(model_input['body_joints'][0, idx, :, 2:])
                        self.model.smpl.smpl.write_obj(torch.cat([model_input['body_joints'][0, idx, :, :2], _pad], dim=1),
                                                            save_path + f'joints_{step + idx * self.num}_gt.obj', write_face=False)
                        self.model.smpl.smpl.write_obj(torch.cat([model_output['perspective_body_joints_h'][0, idx, :, :2], _pad], dim=1),
                                                        save_path + f'joints_{step + idx * self.num}_pred.obj', write_face=False)

        np.save(save_path + 'pose.npy', pose_pred_save)
        np.save(save_path + 'trans.npy', trans_pred_save)
        np.save(save_path + 'betas.npy', betas_pred_save)

    def load_checkpoints(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_checkpoints(self, path):
        torch.save(self.model.state_dict(), path)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' # args.device_id #
    setup_seed(20)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    root_dir = '/home/coder/self_rotate_data'

    import faulthandler
    faulthandler.enable()

    apose_estimator = AposeEstimator(device, root_dir)
    #
    # name_list = (os.listdir('/mnt/8T/zh/vrc/params'))[:100][::-1]
    # name_list.remove('Body2D_2046_inner_379')
    # random.shuffle(name_list)
    # # name_list = ['Body2D_2006_342']
    # apose_estimator.train(name_list, outer_epoch=500, inner_epoch=1)

    # -------------
    name = 'C0020'
    name_list = [name]
    # apose_estimator.train(name_list, outer_epoch=1, inner_epoch=10)
    apose_estimator.test(name)
    pass


