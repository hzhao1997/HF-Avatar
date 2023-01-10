import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn

from dataset.make_dataset import (
    diff_optimizer_dataset, load_all_joints_from_openpose,
)
from geometry_model.lib import rodrigues
from geometry_model.loss import (  # , BatchGMMLoss
    AposePriorLoss, EDGELoss, Joint2dLoss, LaplacianLoss, TemporalLoss,
)
from geometry_model.network import diff_optimizer

from utils.general import setup_seed, tensor2numpy
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.make_dataset import load_all_data

class DiffOptimRunner():

    def __init__(self, device, root_dir, name, length=0,
                 # img_size=1024,
                 base_path=None):
        self.device = device
        self.root_dir = root_dir

        self.name = name
        self.length = length

        self.use_normal = False
        self.isupsample = False
        self.use_posematrix = False

        self.base_path = base_path
        self.mid_results_path = f'{base_path}{name}_mid_results/'
        self.final_results_path = f'{base_path}{name}_final_results/'
        os.makedirs(self.mid_results_path, exist_ok=True)
        os.makedirs(self.final_results_path, exist_ok=True)

    def build_optimizer(self):


        self.original_lr = {'pose': 0.0001, 'betas': 0.001, 'trans': 0.001, 'offsets': 0.0008}
        self.final_lr = {'pose': 0.0001, 'betas': 0.001, 'trans': 0.001, 'offsets': 0.0002}
        self.turning_epoch = {'pose': 50, 'betas': 50, 'trans': 50, 'offsets':50}
        texture_lr = 0.04  # 0.02
        self.optimizer = Adam([
            {
                'params': self.model.pose,
                'lr': self.original_lr['pose']
            },
            {
                'params': self.model.betas,
                'lr': self.original_lr['betas']
            },
            {
                'params': self.model.trans,
                'lr': self.original_lr['trans']
            },
            {
                'params': self.model.offsets,
                'lr': self.original_lr['offsets']
            },
            {
                'params': self.model.texture_parameters,
                'lr': texture_lr
            },
        ])


    def adjust_learning_rate(self, epoch):
        lr = {'pose': 0, 'betas': 0, 'trans': 0, 'offsets': 0}
        for _, key in enumerate(lr):
            lr[key] = \
                (self.final_lr[key] - self.original_lr[key]) / \
                self.turning_epoch[key] * epoch + self.original_lr[key] \
                if epoch <= self.turning_epoch[key] else self.final_lr[key]

        self.optimizer.param_groups[0]['lr'] = lr['pose']
        self.optimizer.param_groups[1]['lr'] = lr['betas']
        self.optimizer.param_groups[2]['lr'] = lr['trans']
        self.optimizer.param_groups[3]['lr'] = lr['offsets']
        if epoch % 2 == 0:
            print(f"pose_lr: {lr['pose']} betas_lr: {lr['betas']} "
                  f"trans_lr: {lr['trans']} offsets_lr: {lr['offsets']}")


    def build_loss_function(self):
        self.L1loss = nn.L1Loss()
        self.MSEloss = nn.MSELoss()
        self.LapLoss = LaplacianLoss(
            device=self.device, isupsample=self.isupsample, stage=self.stage)

        self.TempLoss = TemporalLoss()

        self.EdgeLoss = EDGELoss(
            device=self.device, isupsample=self.isupsample)
        self.Joint2dLoss = Joint2dLoss()
        self.AposePriorLoss = AposePriorLoss(device=self.device)

        # self.GmmLoss = BatchGMMLoss(device=self.device)

    def build_model(self):
        self.model = diff_optimizer(
            device=self.device,
            root_dir=self.root_dir,
            name=self.name,
            view_num=self.view_num,
            length=self.length,
            img_size=[self.img_size, self.img_size],
            f=[self.img_size, self.img_size],
            c=[self.img_size / 2, self.img_size / 2],
            r=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            t=[0., 0., 0.],
            mode=self.mode,
            stage=self.stage,
            isupsample=self.isupsample,
            use_posematrix=self.use_posematrix,
        )  # 'train'


    def build_dataloader(self):
        dataset = diff_optimizer_dataset(
            root_dir=self.root_dir,
            idx_list=self.idx_list,
            name=self.name,
            view_num=self.view_num,
            length=self.length,
            img_size=(self.img_size, self.img_size),
            load_normal=self.use_normal,
        )
        self.dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=4)

    def loss(self, output, groundtruth):
        body_joints_loss = self.Joint2dLoss(output['body_joints2d_pred'],
                                            groundtruth['body_joints2d_gt'])
        face_joints_loss = self.Joint2dLoss(output['face_joints2d_pred'],
                                            groundtruth['face_joints2d_gt'])

        # l1_loss = self.L1loss(pred_imgs, gt_images)
        # sil_loss = self.MSEloss(pred_sil_imgs, gt_sil_images)
        # sharpen_images =
        # self.get_sharpen(gt_images[0].permute(0, 3, 1, 2))
        # .permute(0, 2, 3, 1).unsqueeze(0)
        # gt_images = gt_images + sharpen_images

        # 224:224+576 is applied for non-square image
        l1_loss = self.L1loss(
            output['pred_imgs'],      # [:, :, :, 224:224 + 576, :]
            groundtruth['gt_images']) # [:, :, :, 224:224 + 576, :]
        sil_loss = self.MSEloss(
            output['pred_sil_imgs'],      # [:, :, :, 224:224 + 576]
            groundtruth['gt_sil_images']) # [:, :, :, 224:224 + 576]
        lap_loss = self.LapLoss(output['v_shaped'][0],
                                output['v_shaped_personal'][0])

        pose_matrix = rodrigues(output['pose'].view(-1, 1, 3)).reshape(output['pose'].shape[0], -1, 3, 3)
        temporal_pose_loss = self.TempLoss(pose_matrix)
        temporal_trans_loss = self.TempLoss(output['trans'])
        temporal_vertices_loss = self.TempLoss(output['total_vertices'])

        temporal_loss = \
            1 * temporal_pose_loss + temporal_trans_loss + \
            10 * temporal_vertices_loss
        # edge_loss = self.EdgeLoss(v_shaped_personal[0:1])

        loss = 20 * body_joints_loss + 10 * face_joints_loss + \
            50 * temporal_loss + 0.2 * l1_loss + 2 * sil_loss + 1200 * lap_loss

        loss_output = {
            'loss': loss,
            'body_joints_loss': body_joints_loss,
            'face_joints_loss': face_joints_loss,
            'l1_loss': l1_loss,
            'sil_loss': sil_loss,
            'lap_loss': lap_loss,
            'temporal_loss': temporal_loss,
        }

        return loss_output

    def forward(self, item, ground_truth):

        model_output = self.model.forward(item)
        loss_output = self.loss(model_output, ground_truth)

        self.visualization(item, model_output, ground_truth)
        if item % 20 == 0:
            print(f"step {item}: "
                  f"body_joints_loss {loss_output['body_joints_loss'].item()} "
                  f"face_joints_loss {loss_output['face_joints_loss'].item()} "
                  f"temporal_loss {loss_output['temporal_loss'].item()} "
                  f"sil_loss {loss_output['sil_loss'].item()} "
                  f"lap_loss {loss_output['lap_loss'].item()} "
                  f"l1_loss {loss_output['l1_loss'].item()} ")

        return loss_output['loss']


    # optimize pose, shape, trans, offsets and texture both
    def optimize(self, load_path, save_path, is_load_from_octopus=False, stage='1', epoch=100):
        torch.set_grad_enabled(True)
        self.stage = stage
        self.mode = 'train'
        # the number of view on one optimization
        self.view_num = 8
        self.interval_num = (int)(self.length / self.view_num)


        self.img_size = 1024
        self.build_model()
        self.model.load_parameters(load_path, is_load_from_octopus=is_load_from_octopus)
        self.build_optimizer()
        # self.build_dataloader()
        self.build_loss_function()

        all_ground_truth = load_all_data(root_dir=self.root_dir, name=self.name, length=self.length)

        for epoch_idx in range(epoch):
            print(f'-------------- Epoch: {epoch_idx} -------------')
            self.adjust_learning_rate(epoch_idx)

            for item in range(self.interval_num):
            # for step, (item, ground_truth) in enumerate(self.dataloader):
                ground_truth = {}
                for key in all_ground_truth.keys():
                    ground_truth[key] = \
                        all_ground_truth[key][item:item + self.interval_num * self.view_num:self.interval_num]
                for key in ground_truth.keys():
                    ground_truth[key] = ground_truth[key].to(self.device)
                    ground_truth[key] = ground_truth[key].unsqueeze(0)

                self.optimizer.zero_grad()

                loss = self.forward(item, ground_truth)
                loss.backward()
                self.optimizer.step()

            self.model.save_parameters(save_path=save_path, epoch_idx=epoch_idx)

            if epoch_idx % 10 == 0:
                self.model.write_tpose_obj(save_path)

    # ------------------------------------------

    def visualization(self, item, model_output, ground_truth):

        a = tensor2numpy(model_output['pred_imgs'])
        b = tensor2numpy(ground_truth['gt_images'])

        c = tensor2numpy(model_output['pred_sil_imgs'])
        d = tensor2numpy(ground_truth['gt_sil_images'])
        # e = tensor2numpy(v)

        for i in range(a.shape[1]):
            idx = (item) + i * self.interval_num # _idx + i * len(idx_list) #  #
            img = a[0, i, :, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{self.mid_results_path}img_{str(idx).zfill(4)}_pred.png', img * 255)

            img = b[0, i, :, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{self.mid_results_path}img_{str(idx).zfill(4)}_gt.png', img * 255)

            img = c[0, i, :, :]
            cv2.imwrite(f'{self.mid_results_path}sil_{str(idx).zfill(4)}_pred.png', img * 255)

            img = d[0, i, :, :]
            cv2.imwrite(f'{self.mid_results_path}sil_{str(idx).zfill(4)}_gt.png', img * 255)
            # self.model.smpl.smpl.write_obj(e[i],
            # self.mid_results_path + 'v_{}.obj'.format(i))

        # self.model.smpl.smpl.write_obj(v_shaped[0],
        #   self.mid_results_path + 'v_shaped.obj')
        # self.model.smpl.smpl.write_obj(v_shaped_personal[0],
        #   self.mid_results_path + 'v_shaped_personal.obj')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root_dir',
        type=str,
        default='/mnt/8T/zh/zte')
    #
    parser.add_argument('--name', type=str, default='2026_01')  # 2071_01 2026_01
    parser.add_argument('--device_id', type=str, default='2')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--img_size', type=int, default=1024)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # import faulthandler
    # faulthandler.enable()

    print(f'deal with {args.name}')
    setup_seed(20)
    method = '2'
    length = len(os.listdir(f'{args.root_dir}/frames_mat/{args.name}'))
    length = length - length % 8
    print(length)

    # tmp_path = './results'
    process_flow = ['0', '1', '2', '3']
    # process_flow = ['0', '1', '2']
    # process_flow = ['1', '2']
    # process_flow = ['2']
    # process_flow = ['3']
    # process_flow = ['0', '1', '3']
    os.makedirs(args.result_dir, exist_ok=True)
    # ------------------ A-pose estimation ------------------

    if '0' in process_flow:
        apose_estimator = AposeEstimator(
            device=device,
            root_dir=args.root_dir,
            img_size=(args.img_size, args.img_size),
        )
        apose_estimator.test(
            name=args.name,
            length=length,
            save_path=f'{args.result_dir}/params/{args.name}/'
        )

    runner = DiffOptimRunner(
        device=device,
        root_dir=args.root_dir,
        name=args.name,
        length=length,
        # img_size=args.img_size,
        base_path=f'{args.result_dir}/diff_optiming/')



    if '1' in process_flow:
        runner.optimize_all(
            # load_path=f'{args.result_dir}/params/{args.name}/',
            load_path=f'{args.result_dir}/diff_optiming/{args.name}/',
            # load_path=f'/mnt/8T/zh/vrc/params_of_octopus/{args.name}/',
            save_path=f'{args.result_dir}/diff_optiming/{args.name}/',
            is_load_from_octopus=False,
            # is_load_from_octopus=True,
            stage='1',
            epoch=70)
