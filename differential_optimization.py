import os
import sys
import numpy as np
import cv2
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, LBFGS
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesUV,
    TexturesVertex
)
from geometry_model.network import diff_optimizer
from geometry_model.lib import rodrigues_v, rodrigues
from geometry_model.loss import LaplacianLoss, TemporalLoss, EDGELoss, Joint2dLoss, AposePriorLoss # , BatchGMMLoss
from dataset.make_dataset import diff_optimizer_dataset, get_all_joints_from_openpose, get_all_silhouettes
from utils.general import setup_seed, tensor2numpy
from pose_estimation import AposeEstimator

# from texture_model.loss import Get_sharpen


class DiffOptimRunner():
    def __init__(self, device, root_dir, name, stage='0', length=0, load_path=None, save_path=None):
        self.device = device
        self.root_dir = root_dir

        self.name = name

        self.length = length - length % 8
        # self.view_num = 8  # 8
        # self.num = (int)(self.length / self.view_num)  # * 8
        # self.interval_num = (int)(self.length / self.view_num)
        # self.idx_list = [i for i in range(1, self.num + 1)]
        self.img_size = 1024

        self.use_normal = False
        self.isupsample = False
        self.use_posematrix = False
        self.stage = stage # '1'

        self.load_path, self.save_path = load_path, save_path

        os.makedirs(self.save_path, exist_ok=True)

        # self.get_sharpen = Get_sharpen(kernel_name='n')

    def build_optimizer(self):

        if self.stage == '0':
            # self.params_lr = params_lr = 0.0005 # 1 # 0.0005
            self.original_lr = {'pose':0.0005, 'betas': 0.001, 'trans': 0.001}
            self.final_lr = {'pose':0.0001, 'betas': 0.001, 'trans': 0.001}
            self.turning_epoch = {'pose':600, 'betas': 1, 'trans': 1}
            self.optimizer = Adam(
                [
                    {'params': self.model.pose, 'lr': self.original_lr['pose']}, # params_lr
                    {'params': self.model.betas, 'lr': self.original_lr['betas']},  # 0.001
                    {'params': self.model.trans, 'lr': self.original_lr['trans'] }, # 0.001
                ]
            )
        elif self.stage == '1':
            self.original_lr = original_lr = 0.0005
            texture_lr = 0.02
            self.optimizer = Adam(
                [
                    {'params': self.model.pose, 'lr': original_lr},
                    {'params': self.model.betas, 'lr': original_lr},  # 0.0015
                    {'params': self.model.trans, 'lr': original_lr},
                    {'params': self.model.offsets, 'lr': original_lr},  # 0.0015
                    {'params': self.model.texture_parameters, 'lr': texture_lr},
                ]
            )

    def adjust_learning_rate(self, epoch):

        if self.stage == '0':
            lr = {}
            lr['pose'] = (self.final_lr['pose'] - self.original_lr['pose']) / self.turning_epoch['pose'] * epoch \
                 + self.original_lr['pose'] if epoch <= self.turning_epoch['pose'] else self.final_lr['pose']
            lr['betas']  = (self.final_lr['betas'] - self.original_lr['betas']) / self.turning_epoch['betas'] * epoch \
                 + self.original_lr['betas'] if epoch <= self.turning_epoch['betas'] else self.final_lr['betas']
            lr['trans']  = (self.final_lr['trans'] - self.original_lr['trans']) / self.turning_epoch['trans'] * epoch \
                 + self.original_lr['trans'] if epoch <= self.turning_epoch['trans'] else self.final_lr['trans']

            # for param_group in optimizer.param_groups[:3]:
            #     param_group['lr'] = lr
            self.optimizer.param_groups[0]['lr'] = lr['pose']
            self.optimizer.param_groups[1]['lr'] = lr['betas']
            self.optimizer.param_groups[2]['lr'] = lr['trans']
            if epoch % 20 == 0:
                print('pose_lr: {} betas_lr: {} trans_lr: {}'.format(lr['pose'], lr['betas'], lr['trans']) )
        elif self.stage == '1':
            if epoch == 0:
                lr = self.original_lr
            elif epoch <= 20:
                lr = self.original_lr * pow(0.95, epoch)  # * epoch
            elif epoch <= 1000:
                lr = 0.0001  # original_lr * 0.2
            for param_group in self.optimizer.param_groups[:4]:
                param_group['lr'] = lr

            if epoch % 2 == 0:
                print('-------------- learning rate: {} -------------- '.format(lr))
        # optimizer.param_groups[1]['lr'] = lr

    def build_loss_function(self):
        self.L1loss = nn.L1Loss()
        self.MSEloss = nn.MSELoss()
        self.LapLoss = LaplacianLoss(device=self.device, isupsample=self.isupsample, stage=self.stage)
        # SymLoss = SymmetryLoss()
        self.TempLoss = TemporalLoss()

        self.EdgeLoss = EDGELoss(device=self.device, isupsample=self.isupsample)
        self.Joint2dLoss = Joint2dLoss()
        self.AposePriorLoss = AposePriorLoss(device=self.device)

        # self.GmmLoss = BatchGMMLoss(device=self.device)

    def build_network(self):
        self.model = diff_optimizer(device=self.device, root_dir=self.root_dir,
                                           name=self.name, view_num=self.view_num,
                                           img_size=[self.img_size, self.img_size],
                                           f=[self.img_size, self.img_size],
                                           c=[self.img_size / 2, self.img_size / 2],
                                           r=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                                           mode=self.mode, length=self.length,
                                           isupsample=self.isupsample, stage=self.stage,
                                           use_posematrix=self.use_posematrix,
                                           )  # 'train'

    def build_dataloader(self):
        dr_dataset = diff_optimizer_dataset(root_dir=self.root_dir, idx_list=self.idx_list,
                                                      name=self.name, view_num=self.view_num,
                                                      length=self.length,
                                                      img_size=(self.img_size, self.img_size),
                                                      load_normal=self.use_normal,
                                                      )
        self.dataloader = DataLoader(dr_dataset, batch_size=1, shuffle=False, num_workers=4)

    # ------------------------------------------------
    # # optimize pose, shape and trans only
    def optimize_joints(self, stage = '0'):
        torch.set_grad_enabled(True)
        body_joints2d_gt, face_joints2d_gt  = get_all_joints_from_openpose(self.root_dir, self.name, self.length)

        body_joints2d_gt, face_joints2d_gt = body_joints2d_gt.to(self.device), face_joints2d_gt.to(self.device)
        epoch = 800 # 1600 #
        self.stage = stage # '0'
        self.mode = 'train'
        self.view_num = 8
        self.num = (int)(self.length / self.view_num)
        self.idx_list = [i for i in range(1, self.num + 1)]

        self.build_network()
        self.model.load_parameters(self.load_path)
        self.build_optimizer()
        self.build_loss_function()

        for epoch_idx in range(epoch):
            self.adjust_learning_rate(epoch_idx)

            self.optimizer.zero_grad()
            output = self.model.joints_forward_only()
            body_joints_loss = self.Joint2dLoss(output['perspective_body_joints_h'], body_joints2d_gt)
            face_joints_loss = self.Joint2dLoss(output['perspective_face_joints_h'], face_joints2d_gt)
            # gmm_loss = 0.0001 * self.GmmLoss(self.model.pose)

            a_pose_prior_loss = self.AposePriorLoss(output['pose'])
            # if self.use_posematrix:
            #     temporal_pose_loss = self.TempLoss(self.model.so_pose)  # ? model.pose
            # else:
            pose_matrix = rodrigues(output['pose'].view(-1, 1, 3)).reshape(output['pose'].shape[0], -1, 3, 3)
            temporal_pose_loss = self.TempLoss(pose_matrix)

            temporal_trans_loss = self.TempLoss(output['trans'])

            temporal_loss = 1 * temporal_pose_loss + temporal_trans_loss

            loss = 20 * body_joints_loss + 100 * face_joints_loss  + 0.003 * a_pose_prior_loss + 50 * temporal_loss

            loss.backward()
            self.optimizer.step()
            if epoch_idx % 20 == 0:
                print('epoch_idx {}: body_joints_loss {} face_joints_loss {} a_pose_prior_loss {} temporal_loss {}'.format(
                    epoch_idx, body_joints_loss.item(), face_joints_loss.item(), a_pose_prior_loss.item(), temporal_loss.item()))


        self.model.save_parameters(self.save_path)

        # just for visualization
        if False:
            with torch.no_grad():
                v = tensor2numpy(total_naked_vertices)
                step = 1
                _pad = torch.zeros_like(body_joints2d_gt[idx, :, 2:])
                for idx in range(0, v.shape[0], step):
                    self.model.smpl.smpl.write_obj(v[idx], self.mid_results_path + 'v_{}.obj'.format(int(idx / step)))
                    self.model.smpl.smpl.write_obj(torch.cat([body_joints2d_gt[idx, :, :2], _pad], dim=1),
                                                        self.mid_results_path + 'joints_{}_gt.obj'.format(int(idx / step)),
                                                        write_face=False)
                    self.model.smpl.smpl.write_obj(torch.cat([perspective_body_joints_h[idx, :, :2], _pad],dim=1),
                                                        self.mid_results_path + 'joints_{}_pred.obj'.format(int(idx / step)),
                                                        write_face=False)
                texture_size = 512
                texture = torch.from_numpy(np.ones([1, texture_size, texture_size, 3]).astype(np.float32) * 0.5).to(self.device)
                texture = TexturesUV(texture, self.model.fts, self.model.vts)
                for i in range(0, v.shape[0], step):
                    mesh = Meshes(total_naked_vertices[i:i + 1], self.model.fs)
                    mesh.textures = texture
                    mesh = mesh.to(self.device)

                    # image = self.simple_renderer(mesh)
                    image = self.model.renderer(mesh)
                    image = torch.flip(image, [2])

                    image = tensor2numpy(image)
                    # cv2.imwrite(self.mid_results_path + 'mesh_{}_pre.png'.format(i), image[0,:,:,:3] * 255)
                    cv2.imwrite(self.mid_results_path + 'sil_{}_pred.png'.format(i), image[0,:,:,3] * 255)
                    cv2.imwrite(self.mid_results_path + 'sil_{}_gt.png'.format(i), silhouettes_gt[i])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/8T/zh/vrc') # '/home/coder/vrc/'
    parser.add_argument('--name', type=str, default='Body2D_2061_507')  #       C0029 2040_01 Body2D_2070_380
    parser.add_argument('--device_id', type=str, default='2')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # import faulthandler
    # faulthandler.enable()

    print("deal with {}".format(args.name))
    setup_seed(20)
    method = '2'
    length = len(os.listdir(args.root_dir + f'/frames_mat/{args.name}'))
    length = length - length % 8
    print(length)


    # ------------------ A-pose estimation ------------------
    apose_estimator = AposeEstimator(device, args.root_dir)
    apose_estimator.test(args.name)

    trainer = DiffOptimRunner(device=device, root_dir=args.root_dir, name=args.name, length=length,
                       )
    # ------------------ optimize pose, trans, and shape ------------------
    trainer.optimize_joints(stage='0')
    trainer.test(stage='1')
    # ------------------ optimize pose, trans, shape, offsets and texture ------------------
    trainer.train(stage='1')

    # trainer.test()




    # trainer = Trainer(device=device, root_dir='/mnt/8T/zh/vrc/', name = 'Body2D_2035_375', dataset_name='vrc')




    pass
