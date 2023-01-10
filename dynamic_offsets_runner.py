import os
import numpy as np
import pickle as pkl
import cv2

import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from utils.general import setup_seed

from geometry_model.network import smpl_tpose_layer, rodrigues, dynamic_offsets_network
from geometry_model.loss import TemporalLoss, LaplacianLoss, EDGELoss
from dataset.make_dataset import dynamic_offsets_dataset
from texture_model.lib import Get_sharpen
from utils.general import numpy2tensor, tensor2numpy
from utils.data_generation import UVPositionalMapGenerator

from differential_optimization import DiffOptimRunner


class GeoTrainer():
    def __init__(self, device, root_dir, name, length, batch_size=1, tmp_path=None,
                 load_path=None, base_path='./results/dynamic_offsets/'):
        self.device = device
        self.root_dir = root_dir

        self.name = name
        self.length = length - length % 8
        self.idx_list = [i for i in range(1, self.length + 1)]

        self.batch_size = batch_size

        self.use_normal = True
        self.isupsample = True
        self.use_posematrix = False
        self.img_size = (1024, 1024)

        self.load_path = load_path
        self.base_path = base_path
        self.checkpoints_path = f'{base_path}{name}/'
        self.mid_results_path = f'{base_path}{name}_mid_results/'
        self.results_path = f'{base_path}{name}_final_results/'
        self.save_path = f'{base_path}{name}/'
        self.tmp_path = tmp_path

        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(self.mid_results_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)

        pass

    def build_optimizer(self):
        self.optimizer = Adam(
            [
                {'params': self.network.geometry_encoder.parameters(), 'lr': 0.0001},
                {'params': self.network.pose_encoder.parameters(), 'lr': 0.0001},
                {'params': self.network.shape_decoder.parameters(), 'lr': 0.0001},
                {'params': self.network.offsets_decoder.parameters(), 'lr': 0.0001},
                # {'params': do_network.normal_decoder.parameters(), 'lr': 0.0001},

                {'params': self.network.geometry_parameters, 'lr': 0.0001},
                {'params': self.network.texture_parameters, 'lr': 0.01},

            ]
        )

    def build_loss_function(self):
        self.L1loss = nn.L1Loss()
        self.MSEloss = nn.MSELoss()
        self.LapLoss = LaplacianLoss(device=self.device, isupsample=self.isupsample)
        # SymLoss = SymmetryLoss()
        self.TempLoss = TemporalLoss()

        self.EdgeLoss = EDGELoss(device=self.device, isupsample=self.isupsample)
        self.get_sharpen = Get_sharpen(kernel_name='n', amplitude=5)
        # self.Joint2dLoss = Joint2dLoss()
        # self.AposePriorLoss = AposePriorLoss(device=self.device)

    def build_network(self, mode):
        self.network = dynamic_offsets_network(device=self.device, name=self.name, batch_size=self.batch_size,
                                               isupsample=self.isupsample, use_posematrix=self.use_posematrix,
                                               mode=mode)
        self.network = self.network.to(self.device)
        if mode == 'train':
            self.network.train()
        elif mode == 'test':
            self.network.eval()


    def build_dataloader(self, mode):
        dataset = dynamic_offsets_dataset(root_dir=self.root_dir, idx_list=self.idx_list, name=self.name,
                                          img_size=self.img_size, use_normal=self.use_normal, tmp_path=self.tmp_path)
        if mode == 'train':
            self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        elif mode == 'test':
            self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def loss(self, output, groundtruth):
        # l1_loss = self.L1loss(output['pred_images'][:,:,224:224+576,:], groundtruth['gt_images'][:,:,224:224+576,:])
        # sil_loss = self.MSEloss(output['pred_sil_images'][:,:,224:224+576], groundtruth['gt_sil_images'][:,:,224:224+576])
        l1_loss = self.L1loss(output['pred_images'], groundtruth['gt_images'])
        sil_loss = self.MSEloss(output['pred_sil_images'], groundtruth['gt_sil_images'])

        lap_loss = 0.0
        for i in range(output['v_shaped'].shape[0]):
            lap_loss += self.LapLoss(output['v_shaped'][i], output['v_shaped_personal'][i])
        lap_loss = lap_loss if self.isupsample == False else lap_loss / output['v_shaped'].shape[0]

        if self.use_normal == True:
            gt_normal_imgs = groundtruth['gt_normal_images'].permute(0, 3, 1, 2)
            gt_normal_imgs = gt_normal_imgs + self.get_sharpen(gt_normal_imgs)
            gt_normal_imgs = gt_normal_imgs.permute(0, 2, 3, 1)

            gt_normal_imgs[torch.where(groundtruth['gt_sil_images'] == 0)] = torch.from_numpy(np.zeros(3).astype(np.float32)).cuda()
            gt_normal_imgs = F.normalize(gt_normal_imgs, dim=3)
            groundtruth['gt_normal_images'] = gt_normal_imgs
            normal_loss = - torch.mean(output['pred_normal_images'] * gt_normal_imgs)

        edge_loss = self.EdgeLoss(output['v_shaped_personal'])
        move_loss = torch.mean(torch.abs(output['v_shaped_personal'] - output['v_shaped']))

        loss = 2 * l1_loss + 20 * sil_loss + 1600 * 8 * lap_loss + 50 * normal_loss + 10 * edge_loss + 10 * move_loss
        loss_output = {
            'loss': loss,
            'l1_loss': l1_loss,
            'sil_loss': sil_loss,
            'lap_loss': lap_loss,
            'normal_loss': normal_loss,
            'edge_loss': edge_loss,
            'move_loss': move_loss,
        }
        return loss_output

    def forward(self, step, item, input, ground_truth):
        model_output = self.network.forward(item, input['naked_vertice_uv'])
        loss_output = self.loss(model_output, ground_truth)
        if step % 20 == 0:
            print(f"step {step}: l1_loss {loss_output['l1_loss'].item()} sil_loss {loss_output['sil_loss'].item()} "
                  f"lap_loss {loss_output['lap_loss'].item()} normal_loss {loss_output['normal_loss'].item()} "
                  f"edge_loss {loss_output['edge_loss'].item()} move_loss {loss_output['move_loss'].item()}" )
        if step % 80 == 0:
            self.visualization(model_output, ground_truth)

        return loss_output['loss']

    def train(self):
        torch.set_grad_enabled(True)
        epoch = 10
        mode = 'train'

        self.build_network(mode=mode)
        self.network.load_parameters(load_path=self.load_path)
        self.build_optimizer()
        self.build_dataloader(mode=mode)
        self.build_loss_function()

        for epoch_idx in range(epoch):
            print(f'--------------- Epoch: {epoch_idx} ---------------')
            for step, (item, input, ground_truth) in enumerate(self.dataloader):
                item = item.to(self.device)
                for key in input.keys():
                    input[key] = input[key].to(self.device)
                for key in ground_truth.keys():
                    ground_truth[key] = ground_truth[key].to(self.device)

                self.optimizer.zero_grad()
                loss = self.forward(step, item, input, ground_truth)
                loss.backward()
                self.optimizer.step()
            self.save_weights()
            # if epoch_idx % 10 == 0:
        self.save_weights()

    def test(self):
        torch.set_grad_enabled(False)
        mode = 'test'
        self.build_network(mode=mode)
        self.network.load_parameters(load_path=self.load_path)
        self.build_dataloader(mode=mode)
        self.load_weights()

        offsets = []
        for step, (item, input, ground_truth) in enumerate(self.dataloader):
            item = item.to(self.device)
            for key in input.keys():
                input[key] = input[key].to(self.device)
            for key in ground_truth.keys():
                ground_truth[key] = ground_truth[key].to(self.device)

            with torch.no_grad():
                model_output = self.network.forward(item, input['naked_vertice_uv'])



            _idx = tensor2numpy(item)
            o = tensor2numpy(model_output['offsets'])
            offsets.append(o)

            v = tensor2numpy(model_output['v'])

            mi = tensor2numpy(model_output['mesh_images'])
            ri = tensor2numpy(model_output['pred_imgs'])

            os.makedirs(f'{self.results_path }/obj', exist_ok=True)
            os.makedirs(f'{self.results_path }/img', exist_ok=True)
            write_obj = False
            if write_obj == True:
                for i in range(v.shape[0]):
                    self.network.write_v_obj(v=v[i], path=f'{self.results_path}/obj/{str(_idx[i] + 1).zfill(4)}.obj')
            if step == 0:
                self.network.write_v_obj(v=v[0], path=f'{self.results_path}/obj/{str(_idx[0] + 1).zfill(4)}.obj')
            for i in range(mi.shape[0]):
                img = mi[i, :, :, :3]
                cv2.imwrite(f'{self.results_path}/img/mesh_img_{str(_idx[i]+1).zfill(4)}.png', img * 255)
                img = ri[i, :, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{self.results_path}/img/render_img_{str(_idx[i]+1).zfill(4)}.png', img * 255)

        self.network.save_parameters(save_path=self.save_path)
        offsets = np.concatenate(offsets, axis=0)
        np.save(f'{self.save_path}offsets.npy', offsets)

    def visualization(self, model_output, ground_truth):
        a = tensor2numpy(model_output['pred_images'])
        b = tensor2numpy(ground_truth['gt_images'])

        c = tensor2numpy(model_output['pred_sil_images'])
        d = tensor2numpy(ground_truth['gt_sil_images'])

        # _idx = (int)(item.detach().cpu().numpy())
        for i in range(a.shape[0]):
            idx = i  # _idx + i * len(idx_list) #  #
            img = a[i, :, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{self.mid_results_path}img_{idx}_pred.png', img * 255)

            img = b[i, :, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{self.mid_results_path}img_{idx}_gt.png', img * 255)

            img = c[i, :, :]
            cv2.imwrite(f'{self.mid_results_path}sil_{idx}_pred.png', img * 255)

            img = d[i, :, :]
            cv2.imwrite(f'{self.mid_results_path}sil_{idx}_gt.png', img * 255)

            # self.network.write_v_obj(model_output['v'][i],  f'{self.mid_results_path}v_{i}.obj')
            # self.network.write_v_obj(model_output['v_shaped_personal'][i],
            #                        f'{self.mid_results_path}v_shaped_personal_{i}.obj')

        # self.network.write_v_obj(model_output['v_shaped'][0], self.mid_results_path + 'v_shaped.obj')

        if self.use_normal == True:
            m = tensor2numpy(model_output['pred_normal_images'])
            n = tensor2numpy(ground_truth['gt_normal_images'])
            for i in range(m.shape[0]):
                idx = i

                normal = m[i, :, :, :3]
                normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)  # sum = np.sum(np.square(normal), axis=2)
                normal_img = (normal * 0.5 + 0.5) * 255
                cv2.imwrite(f'{self.mid_results_path}normal_img_{idx}_pred.png', normal_img)  # img * 255

                normal = n[i, :, :, :3]
                normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)  # sum = np.sum(np.square(normal), axis=2)
                normal_img = (normal * 0.5 + 0.5) * 255
                cv2.imwrite(f'{self.mid_results_path}normal_img_{idx}_gt.png', normal_img)  # img * 255

    def save_weights(self, epoch_idx=None):
        save_path = f'{self.checkpoints_path}do_network_{epoch_idx}.pt' if epoch_idx is not None \
            else f'{self.checkpoints_path}do_network.pt'
        torch.save(self.network.state_dict(), save_path)


    def load_weights(self, epoch_idx=None):
        load_path = f'{self.checkpoints_path}do_network_{epoch_idx}.pt' if epoch_idx is not None \
            else f'{self.checkpoints_path}do_network.pt'
        self.network.load_state_dict(torch.load(load_path))

        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/8T/zh/vrc') # '/home/coder/vrc/'
    parser.add_argument('--name', type=str, default='Body2D_2061_507')  #  Body2D_2027_286  Body2D_2043_288  Body2D_2064_287 C0029 2040_01    Body2D_2070_380
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--tmp_path', type=str, default='./results')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print(f"deal with {args.name}")
    length = len(os.listdir(f'{args.root_dir}/frames_mat/{args.name}'))
    length = length - length % 8
    print(length)

    setup_seed(20)
    process_flow = [ '1', '2', '3']
    # process_flow = ['2', '3']
    # process_flow = ['3']
    tmp_path = args.tmp_path
    if '1' in process_flow:
        runner = DiffOptimRunner(device=device, root_dir=args.root_dir,
                                 name=args.name, length=length,
                                 base_path=f'{args.tmp_path}/diff_optiming/'
                            )
        runner.optimize(load_path=f'{args.root_dir}/params/{args.name}/', # f'{tmp_path}/params/{args.name}/',
                        save_path=f'{tmp_path}/diff_optiming/{args.name}/',
                        is_load_from_octopus=True,
                        epoch=30
        )
    if '2' in process_flow:
        # target_path = './results'
        uv_pos_generator = UVPositionalMapGenerator(src_data_path=f'{tmp_path}/diff_optiming/{args.name}')
        uv_pos_generator.generate_uv(target_uv_path=f'{tmp_path}/naked_vertice_uv/{args.name}')

    if '3' in process_flow:
        trainer = GeoTrainer(device=device, root_dir=args.root_dir, name=args.name,
                             length=length, batch_size=args.batch_size, tmp_path=tmp_path,
                             load_path=f'{tmp_path}/diff_optiming/{args.name}/',
                             base_path=f'{tmp_path}/dynamic_offsets/',)

        trainer.train()
        trainer.test()

