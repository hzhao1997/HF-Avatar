import numpy as np
import cv2
import os
import tqdm
import sys
import math
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from dataset.make_dataset import neural_rendering_train_dataset, neural_rendering_test_dataset, neural_rendering_finetune_dataset
from texture_model.network import neural_texture_network, neural_rendering_network, perceptual_network
from texture_model.lib import Get_sharpen, get_tex, save_tex


from utils.data_generation import UVMapGenerator
from utils.general import setup_seed, numpy2tensor, tensor2numpy

class BaseTrainer:
    def __init__(self, device, img_dir, uv_dir, ref_dir=None, name=None, length=None, batch_size=None, tmp_path='./results'):
        self.device = device
        self.img_dir, self.uv_dir, self.ref_dir = img_dir, uv_dir, ref_dir  # self.data_dir

        self.name = name
        self.batch_size = batch_size

        self.length = length
        self.img_size = (1024, 1024)
        self.batch_size = batch_size

        self.texturew = self.textureh = 2048
        self.texture_dim = 16

        # self.lr = 0.001

        self.checkpoint_path = './checkpoints'

        self.pretrained_checkpoint_path = self.checkpoint_path + '/renderer.pt'
        self.save_checkpoint_path = f'{tmp_path}/checkpoints' # self.checkpoint_path # + '/'

        self.coarse_texture_path = f'{tmp_path}/coarse_texture/'
        self.final_texture_path = f'{tmp_path}/final_texture/'
        self.render_path = f'{tmp_path}/render/{self.name}'

        os.makedirs(self.save_checkpoint_path, exist_ok=True)
        os.makedirs(self.coarse_texture_path, exist_ok=True)
        os.makedirs(self.final_texture_path, exist_ok=True)
        os.makedirs(self.render_path, exist_ok=True)

        self.build_loss_function()

    def build_optimizer(self):

        pass
    def build_loss_function(self):
        self.L1Loss = nn.L1Loss()
        self.MSELoss = nn.MSELoss()
        self.perceptualnet = perceptual_network().to(device)
        pass
    def build_network(self):
        pass

    def build_dataloader(self):
        pass

    def load_weights(self):
        pass

    def train(self):
        pass

class NeuralTextureTrainer(BaseTrainer):
    def __init__(self, device, img_dir, uv_dir, ref_dir, name, length, batch_size=1):
        super(NeuralTextureTrainer, self).__init__(device, img_dir, uv_dir, ref_dir, name, length, batch_size)

        self.build_network()
        self.build_optimizer()
        self.build_dataloader()

        # torch.save(self.model.texture.state_dict(), self.save_checkpoint_path + '/texture.pt'.format(self.name))

    def build_network(self):
        self.model = neural_texture_network(self.texturew, self.textureh, self.texture_dim)
        self.model = self.model.to(device)
        self.model.train()

    def build_optimizer(self):
        self.lr = 0.001
        for m in self.model.texture.parameters():
            m.requires_grad = True

        self.optimizer = Adam([
            {'params': self.model.texture.layer1, 'lr': self.lr}
        ])

    def build_dataloader(self):
        self.idx_list = ['{:04d}'.format(i) for i in range(1, self.length)]
        dataset = neural_rendering_train_dataset(self.img_dir, self.uv_dir, self.ref_dir,
                                                 self.idx_list, self.name, self.img_size)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def train(self):
        epoch = 10
        torch.set_grad_enabled(True)
        for epoch_idx in range(1, 1 + epoch):
            print(f'Epoch {epoch_idx}')
            for step, samples in enumerate(self.dataloader):
                images, uv_maps, _ = samples
                images = images.to(self.device)
                uv_maps = uv_maps.to(self.device)

                self.optimizer.zero_grad()
                RGB_texture = self.model(uv_maps)

                l1_loss1 = self.L1Loss(RGB_texture, images)
                loss = 10 * l1_loss1  # + 0 * l1_loss2  # + 0 * perceptualLoss

                loss.backward()
                self.optimizer.step()
                if step % 50 == 0:
                    print('step {}: l1_loss1 {} '.format(step, l1_loss1.item()))

            save_tex(self.model, self.coarse_texture_path, self.name)
        pass

class NeuralRenderingTrainer(BaseTrainer):
    def __init__(self, device, img_dir, uv_dir, ref_dir, name, length, batch_size=1):
        super(NeuralRenderingTrainer, self).__init__(device, img_dir, uv_dir, ref_dir, name, length, batch_size)


    def build_network(self, mode='train'):
        self.model = neural_rendering_network(self.texturew, self.textureh, self.texture_dim,
                                              norm='instance')
        self.model = self.model.to(device)
        if mode == 'train':
            self.model.train()
        elif mode == 'test':
            self.model.eval()

    def build_optimizer(self):
        self.lr = 0.001
        for m in self.model.renderer.parameters():
            m.requires_grad = True
        for m in self.model.texture.parameters():
            m.requires_grad = True  # False

        for idx in range(3):
            for m in self.model.texture.textures[idx].parameters():
                m.requires_grad = False

        self.optimizer = Adam([
            {'params': self.model.texture.layer1, 'lr': self.lr},
            {'params': self.model.renderer.parameters(), 'lr': 0.1 * self.lr},
        ])

    def build_dataloader(self, mode = 'train'):
        if mode == 'train':
            self.idx_list = ['{:04d}'.format(i) for i in range(1, self.length)]
            dataset = neural_rendering_train_dataset(self.img_dir, self.uv_dir, self.ref_dir, self.idx_list, self.name,
                                                     self.img_size)
            self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        elif mode == 'test':
            self.idx_list = ['{:04d}'.format(i) for i in range(1, 360)]
            dataset = neural_rendering_test_dataset(self.img_dir, self.uv_dir, self.ref_dir, self.idx_list, self.name,
                                                    self.img_size)
            self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    def load_weights(self, mode='train'):
        if mode == 'train':
            tex = get_tex(path=self.coarse_texture_path, name=self.name).cuda()

            pretrained_dict = torch.load(os.path.join(self.checkpoint_path, 'texture.pt'))
            pretrained_dict['textures.0.layer1'] = torch.unsqueeze(tex[:, 0, :, :], axis=0)  # texture.textures.0.layer4
            pretrained_dict['textures.1.layer1'] = torch.unsqueeze(tex[:, 1, :, :], axis=0)
            pretrained_dict['textures.2.layer1'] = torch.unsqueeze(tex[:, 2, :, :], axis=0)
            self.model.texture.load_state_dict(pretrained_dict)

            if os.path.exists(self.pretrained_checkpoint_path):
                self.model.renderer.load_state_dict(torch.load(self.pretrained_checkpoint_path))
        elif mode == 'test':
            self.model.texture.load_state_dict(torch.load(f'{self.save_checkpoint_path}/texture_{self.name}.pt'))
            self.model.renderer.load_state_dict(torch.load(f'{self.save_checkpoint_path}/renderer_{self.name}.pt'))

    def train(self):
        epoch = 10
        torch.set_grad_enabled(True)
        self.build_network(mode='train')
        self.build_dataloader(mode='train')
        self.load_weights(mode='train')
        self.build_optimizer()
        for epoch_idx in range(1, 1 + epoch):
            print(f'Epoch {epoch_idx}')
            for step, samples in enumerate(self.dataloader):

                images, uv_maps, ref_img = samples
                images, uv_maps, ref_img = images.to(self.device), uv_maps.to(self.device), ref_img.to(self.device)

                self.optimizer.zero_grad()

                sampled_texture, preds = self.model(uv_maps, ref_img)

                l1_loss1 = self.L1Loss(sampled_texture, images)
                l1_loss2 = self.L1Loss(preds, images)

                img_featuremaps = self.perceptualnet(images)
                pred_featuremaps = self.perceptualnet(preds)
                perceptualLoss = 0.0
                for s in range(len(img_featuremaps)):
                    perceptualLoss += self.L1Loss(pred_featuremaps[s], img_featuremaps[s])

                loss = 0 * l1_loss1 + 10 * l1_loss2 + 3 * perceptualLoss

                loss.backward()
                self.optimizer.step()
                if step % 20 == 0:
                    print('step {}: l1_loss1 {} l1_loss2 {} precept_loss {}'.format(step,
                        l1_loss1.item(), l1_loss2.item(), perceptualLoss.item()))

            torch.save(self.model.texture.state_dict(),  f'{self.save_checkpoint_path}/texture_{self.name}.pt')
            torch.save(self.model.renderer.state_dict(), f'{self.save_checkpoint_path}/renderer_{self.name}.pt')

        pass

    def test(self):
        torch.set_grad_enabled(False)
        self.build_network(mode='test')
        self.build_dataloader(mode='test')
        self.load_weights(mode='test')
        with torch.no_grad():
            for samples in tqdm.tqdm(self.dataloader):

                uv_maps, ref_img, idxs = samples
                uv_maps, ref_img = uv_maps.to(self.device), ref_img.to(self.device)
                sampled_texture, preds = self.model(uv_maps, ref_img)  #


                for i in range(len(idxs)):
                    rgb_image = np.clip(np.transpose(tensor2numpy(sampled_texture[i]), [1, 2, 0]), 0, 1)
                    image = np.clip(np.transpose(tensor2numpy(preds[i]), [1, 2, 0]), 0, 1)

                    cv2.imwrite(self.render_path + f'/sampled_texture_{idxs[i]}.png', rgb_image[:, :, ::-1] * 255)
                    cv2.imwrite(self.render_path + f'/render_{idxs[i]}.png', image[:, :, ::-1] * 255)
        pass

class TexOptimizer(BaseTrainer):
    def __init__(self, device, img_dir, uv_dir, name, batch_size=1):
        super(TexOptimizer, self).__init__(device, img_dir, uv_dir, name=name, batch_size=batch_size)
        self.build_network()
        self.build_optimizer()
        self.build_dataloader()

        self.load_weights()
        self.get_sharpen = Get_sharpen(kernel_name='n')

    def build_network(self):

        texturew, textureh = self.texturew * 2, self.textureh * 2

        self.model = neural_texture_network(texturew, textureh)
        self.model = self.model.to(device)
        self.model.train()

    def build_optimizer(self):
        for m in self.model.texture.parameters():
            m.requires_grad = False  # False
        for idx in range(3):
            for m in self.model.texture.textures[idx].parameters():
                m.requires_grad = True

        self.lr = 0.005

        self.optimizer = Adam([
            {'params': self.model.texture.layer1, 'lr': self.lr},  # * 0.5 args.lr
        ])

    def build_dataloader(self):
        img_size = (self.img_size[0] * 2, self.img_size[1] * 2)

        self.idx_list = ['{:04d}'.format(i) for i in range(1, 360)]
        dataset = neural_rendering_finetune_dataset(self.img_dir, self.uv_dir, self.idx_list,
                                                    self.name, img_size=img_size,)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,num_workers=4)  #


    def load_weights(self):
        texture_size = self.texturew
        tex = get_tex(self.coarse_texture_path, name, shape=((int)(texture_size / 4), (int)(texture_size / 4))).to(device)
        pretrained_dict = torch.load(self.checkpoint_path + '/texture_512.pt')
        # ----------------------------
        pretrained_dict['textures.0.layer1'] = torch.unsqueeze(tex[:, 0, :, :], dim=0)  # texture.textures.0.layer4
        pretrained_dict['textures.1.layer1'] = torch.unsqueeze(tex[:, 1, :, :], dim=0)
        pretrained_dict['textures.2.layer1'] = torch.unsqueeze(tex[:, 2, :, :], dim=0)
        self.model.texture.load_state_dict(pretrained_dict)

        pass

    def adjust_learning_rate(self, optimizer, epoch, original_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        final_lr = 0.001
        base = 1.1
        mid_epoch = math.log( original_lr / final_lr, base )
        if epoch == 1:
            lr = original_lr
        elif epoch <= mid_epoch :
            lr = original_lr * (base ** (-epoch))  # * epoch
        elif epoch <= 1000:
            lr = final_lr  # original_lr * 0.2

        for param_group in optimizer.param_groups[:4]:
            param_group['lr'] = lr
        print('learning rate: ', lr)


    def optimize(self):
        epoch = 10  # 10
        torch.set_grad_enabled(True)
        for epoch_idx in range(1, 1 + epoch):
            print('Epoch {}'.format(epoch_idx))
            step = 0
            self.adjust_learning_rate(self.optimizer, epoch_idx, self.lr)

            for samples in self.dataloader:
                images, uv_maps = samples
                images = images.to(device)
                uv_maps = uv_maps.to(device)
                # ref_img = ref_img.cuda()
                # face_mask = face_mask.cuda()
                self.optimizer.zero_grad()
                RGB_texture = self.model(uv_maps, )
                loss = torch.mean(torch.abs(images + self.get_sharpen(images) - (RGB_texture)))

                loss.backward()
                self.optimizer.step()
                step += images.shape[0]
                if step % 50 == 0:
                    print('step {}: l1_loss1 {} '.format(step, loss.item()))

            save_tex(self.model, path=self.final_texture_path, name=self.name)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/8T/zh/vrc')
    parser.add_argument('--name', type=str, default='Body2D_2037_344')
    #   Body2D_2010_378 Body2D_2061_507   Body2D_2041_308  Body2D_2031_313 Body2D_2070_380
    parser.add_argument('--device_id', type=str, default='2')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id # '2'
    setup_seed(20)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")


    root_dir = args.root_dir # '/mnt/8T/zh/vrc/'
    name = args.name # 'Body2D_2070_380' # 'Body2D_2040_499'
    print(name)
    #

    length = len(os.listdir(f'{root_dir}/frames_mat/{name}'))
    length = length - length % 8
    process_flow = ['0', '1', '2', '3', '4']
    # process_flow = ['1', '2', '3', '4']
    # process_flow = ['2', '3', '4']
    # process_flow = ['3', '4']
    # process_flow = ['4']
    tmp_path = './results'
    src_path = f'{tmp_path}/dynamic_offsets/'  # new_diff_rendering
    img_path = f'{root_dir}/frames_mat/'
    uv_path, test_uv_path, double_test_uv_path = f'{tmp_path}/uvs/', f'{tmp_path}/test_uvs/', f'{tmp_path}/double_test_uvs/'
    ref_path, test_ref_path = f'{tmp_path}/ref/', f'{tmp_path}/test_ref/'
    render_path = f'{tmp_path}/render/'
    if '0' in process_flow:
        uv_generator = UVMapGenerator(device=device, root_dir=root_dir, src_data_path=src_path + name, name=name)
        uv_generator.generate_uv(target_uv_path=uv_path + name, target_ref_path=ref_path + name,
                                 pose_seq='src', img_size=1024, generate_ref=True)
        uv_generator.generate_uv(target_uv_path=test_uv_path + name, target_ref_path=test_ref_path+name,
                                 pose_seq='self_rotate', img_size=1024, generate_ref=True)
        uv_generator.generate_uv(target_uv_path=double_test_uv_path + name,
                                 pose_seq='self_rotate', img_size=2048, generate_ref=False)

    if '1' in process_flow:
        neural_texture_model = NeuralTextureTrainer(device, img_dir=img_path,
                                                    uv_dir=uv_path, ref_dir=ref_path, length=length, name=name)
        neural_texture_model.train()
    if '2' in process_flow:
        neural_rendering_model_train = NeuralRenderingTrainer(device, img_dir=img_path ,
                                                              uv_dir=uv_path, ref_dir=ref_path, length=length, name=name)
        neural_rendering_model_train.train()
    if '3' in process_flow:
        neural_rendering_model_test = NeuralRenderingTrainer(device, img_dir=None,
                                                             uv_dir=test_uv_path, ref_dir=test_ref_path, length=length, name=name)
        neural_rendering_model_test.test()
    if '4' in process_flow:
        tex_optimizer_model = TexOptimizer(device, img_dir=render_path, uv_dir=double_test_uv_path,
                                           name=name, batch_size=4)
        tex_optimizer_model.optimize()

