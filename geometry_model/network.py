import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl

from geometry_model.lib import (
    NormalShader, cal_perspective_matrix, get_f_ft_vt, get_regressor,
    rodrigues, schmidt_orthogonalization, write_obj,
)
from geometry_model.loss import LaplacianLoss
from smpl_model.batch_smpl_torch import SMPLModel
from utils.general import numpy2tensor, tensor2numpy
from pytorch3d.renderer import (
    MeshRasterizer, MeshRenderer, PointLights, RasterizationSettings,
    SoftPhongShader, TexturesUV,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes





class smpl_tpose_layer(nn.Module):

    def __init__(self,
                 device,
                 isupsample=False,
                 use_posematrix=False,
                 model_path=None):
        super(smpl_tpose_layer, self).__init__()

        model_path = './assets/smpl/neutral_smpl.pkl' if isupsample is False \
            else './assets/upsmpl/neutral_smpl.pkl'
        self.smpl = SMPLModel(
            device=device,
            model_path=model_path,
            use_posematrix=use_posematrix)

    def forward(self, betas, pose, trans, offets):
        self.vertices = self.smpl(betas, pose, trans, offets)

        return self.vertices, self.smpl.v_shaped, \
            self.smpl.v_shaped_personal, self.smpl.v_offsets


def build_differential_renderer(device, r, t, K, f, c, img_size):
    R = numpy2tensor(np.expand_dims(np.array(r), axis=0))
    T = numpy2tensor(np.expand_dims(np.array(t), axis=0))  # 0.0,0.2,2.3
    if K is not None:
        K = numpy2tensor(np.expand_dims(np.array(K), axis=0))

    width = img_size[0]  # 1080.0
    height = img_size[1]  # 1080.0

    # self.f = img_size[0]
    focal = numpy2tensor(np.expand_dims(np.array([f[0], f[1]]), axis=0))
    center = numpy2tensor(np.expand_dims(np.array([c[0], c[1]]), axis=0))
    Size = numpy2tensor(np.expand_dims(np.array([width, height]), axis=0))

    if K is not None:
        cam = PerspectiveCameras(
            device=device,
            R=R, T=T, K=K,
            focal_length=focal,
            principal_point=center,
            image_size=Size,
            # in_ndc=False
        )
    else:
        cam = PerspectiveCameras(
            device=device,
            R=R, T=T,
            focal_length=focal,
            principal_point=center,
            image_size=Size,
            # in_ndc=False
        )

    # ------------ ImgShader ------------
    # Define the settings for rasterization and shading.
    blend_params = BlendParams(1e-9, 1e-9, (0.0, 0.0, 0.0))
    cam_ = cam

    lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
    raster_settings_soft = RasterizationSettings(
        image_size=width,
        blur_radius=0,
        faces_per_pixel=2,  #
    )
    # soft rasterizer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cam_, raster_settings=raster_settings_soft),
        shader=SoftPhongShader(
            device=device,
            cameras=cam_,
            lights=lights,
            blend_params=blend_params))
    raster_settings_soft_normal = RasterizationSettings(
        image_size=width,
        blur_radius=0,
        faces_per_pixel=5,
    )

    normal_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cam_, raster_settings=raster_settings_soft_normal),
        shader=NormalShader(device=device, blend_params=blend_params))

    # if mode == 'test':
    focal = numpy2tensor(
        np.expand_dims(np.array([f[0] / 2, f[1] / 2]), axis=0))
    center = numpy2tensor(
        np.expand_dims(np.array([c[0] / 2, c[1] / 2]), axis=0))
    Size = numpy2tensor(
        np.expand_dims(np.array([width / 2, height / 2]), axis=0))
    test_cam = PerspectiveCameras(
        device=device,
        R=R,
        T=T,
        focal_length=focal,
        principal_point=center,
        image_size=Size)  #
    raster_settings_soft_mesh = RasterizationSettings(
        image_size=int(width / 2),
        blur_radius=0,
        faces_per_pixel=5,
    )
    lights_mesh = PointLights(device=device, location=[[0.0, 0.0, 10.0]])
    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=test_cam, raster_settings=raster_settings_soft_mesh),
        shader=SoftPhongShader(
            device=device,
            cameras=test_cam,
            blend_params=blend_params,
            lights=lights_mesh))

    return renderer, normal_renderer, mesh_renderer


class diff_optimizer(nn.Module):

    def __init__(
        self,
        device,
        root_dir,
        name,
        batch_size=1,
        view_num=8,
        length=None,
        img_size=[1024, 1024],
        r=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        t=[0., 0., 0.],
        K=None,
        f=[1024, 1024],
        c=[512, 512],
        mode='train',
        stage='1',
        isupsample=False,
        use_normal=False,
        use_posematrix=False,
    ):
        super(diff_optimizer, self).__init__()

        self.device = device
        self.root_dir = root_dir
        self.name = name

        self.batch_size = batch_size

        self.length = length
        self.view_num = view_num
        self.interval_num = (int)(self.length / self.view_num) \
            if self.view_num is not None else 0

        self.mode = mode
        self.stage = stage

        self.isupsample = isupsample
        self.use_normal = use_normal
        self.use_posematrix = use_posematrix  # True

        self.build_basefiles()

        self.renderer, self.normal_renderer, self.mesh_renderer = \
            build_differential_renderer(device, r, t, K, f, c, img_size)

        self.smpl = smpl_tpose_layer(
            device=device,
            isupsample=isupsample,
            use_posematrix=use_posematrix)
        self.perspective_matrix = cal_perspective_matrix(
            f=[1024, 1024],
            c=[512, 512],
            w=1024,
            h=1024,
        ).to(self.device)

        self.body25_reg_tensor, self.face_reg_tensor = get_regressor()
        self.body25_reg_tensor, self.face_reg_tensor = \
            self.body25_reg_tensor.to(self.device),\
            self.face_reg_tensor.to(self.device)

    def build_basefiles(self):
        f, ft, vt = get_f_ft_vt(self.isupsample)

        self.fs = numpy2tensor(np.expand_dims(f, axis=0), np.long). \
            repeat([self.batch_size, 1, 1]).to(self.device)
        self.fts = numpy2tensor(np.expand_dims(ft, axis=0), np.long). \
            repeat([self.batch_size, 1, 1]).to(self.device)
        self.vts = numpy2tensor(np.expand_dims(vt, axis=0), np.float32). \
            repeat([self.batch_size, 1, 1]).to(self.device)

    def forward(self, item):
        # -------------------
        offsets = 1 * self.offsets

        # -------------------
        self.total_vertices, _, _, _ = \
            self.smpl(self.betas.repeat(self.pose.shape[0], 1), self.pose,
                      self.trans,
                      offsets.repeat(self.pose.shape[0], 1, 1))
        total_naked_vertices, _, _, _ = \
            self.smpl(self.betas.repeat(self.pose.shape[0], 1), self.pose,
                      self.trans,
                      torch.zeros_like(offsets).
                      repeat(self.pose.shape[0], 1, 1))

        # -------------------
        pose = self.pose[item:item +
                         self.interval_num * self.view_num:self.interval_num]
        # --------------------
        trans = self.trans[item:item +
                           self.interval_num * self.view_num:self.interval_num]
        betas = self.betas
        # [item: item + self.interval_num * 8 : self.interval_num]
        pose = pose.contiguous()
        self.vertices, self.v_shaped, self.v_shaped_personal, self.v_offsets =\
            self.smpl(betas.repeat(self.view_num, 1), pose,
                      trans,
                      offsets.repeat(self.view_num, 1, 1))

        naked_v, _, _, _ = \
            self.smpl(betas.repeat(self.view_num, 1), pose,
                      trans,
                      torch.zeros_like(offsets).repeat(self.view_num, 1, 1))

        body_joints = torch.tensordot(
            self.body25_reg_tensor, naked_v, dims=([1], [1])).transpose(0, 1)
        face_joints = torch.tensordot(
            self.face_reg_tensor, naked_v, dims=([1], [1])).transpose(0, 1)

        body_joints_h = \
            torch.cat([body_joints, torch.ones_like(body_joints[..., -1:])],
                      dim=2)
        perspective_body_joints = \
            torch.matmul(body_joints_h, self.perspective_matrix)
        face_joints_h = \
            torch.cat([face_joints, torch.ones_like(face_joints[..., -1:])],
                      dim=2)
        perspective_face_joints = \
            torch.matmul(face_joints_h, self.perspective_matrix)

        body_joints2d_pred = perspective_body_joints /\
            perspective_body_joints[..., -1:]
        face_joints2d_pred = perspective_face_joints /\
            perspective_face_joints[..., -1:]

        body_joints2d_pred, face_joints2d_pred = \
            body_joints2d_pred.unsqueeze(0), \
            face_joints2d_pred.unsqueeze(0)

        v = self.vertices
        # --------------------

        mesh = Meshes(self.vertices,
                      self.fs.repeat(self.vertices.shape[0], 1, 1))

        texture = TexturesUV(
            self.texture_parameters.repeat(self.vertices.shape[0], 1, 1, 1),
            self.fts.repeat(self.vertices.shape[0], 1, 1),
            self.vts.repeat(self.vertices.shape[0], 1, 1))

        mesh.textures = texture
        images = self.renderer(mesh)
        images = images.unsqueeze(0)
        images = torch.flip(images, [3])
        normal_images = []

        if self.mode == 'train':
            output = {
                'pred_imgs': images[:, :, :, :, :3],
                'pred_sil_imgs': images[:, :, :, :, 3],
                'v_shaped': self.v_shaped,
                'v_shaped_personal': self.v_shaped_personal,
                'body_joints2d_pred': body_joints2d_pred,
                'face_joints2d_pred': face_joints2d_pred,
                'v': v,
                'naked_v': naked_v,
                'pred_normal_imgs': normal_images,
                'pose': self.pose,
                'trans': self.trans,
                'betas': self.betas,
                'total_vertices': self.total_vertices
            }
            return output
        elif self.mode == 'test':
            mesh = Meshes(self.vertices,
                          self.fs.repeat(self.vertices.shape[0], 1, 1))

            texture = TexturesUV(
                self.vis_texture.repeat(self.vertices.shape[0], 1, 1, 1),
                self.fts.repeat(self.vertices.shape[0], 1, 1),
                self.vts.repeat(self.vertices.shape[0], 1, 1))
            mesh.textures = texture
            mesh_images = self.mesh_renderer(mesh)
            mesh_images = mesh_images.unsqueeze(0)
            mesh_images = torch.flip(mesh_images, [3])

            output = {
                'pred_imgs': images[:, :, :, :, :3],
                'pred_sil_imgs': images[:, :, :, :, 3],
                'v_shaped': self.v_shaped,
                'v_shaped_personal': self.v_shaped_personal,
                'body_joints2d_pred': body_joints2d_pred,
                'face_joints2d_pred': face_joints2d_pred,
                'v': v,
                'naked_v': naked_v,
                'mesh_images': mesh_images[:, :, :, :, :3]
            }
            return output

    def joints_forward_only(self):
        offsets = 1 * self.offsets
        # betas = self.betas.detach()  # torch.zeros_like().cuda()
        # betas[:, :4] = self.betas[:, :4]
        betas = 1 * self.betas

        self.total_vertices, _, _, _ = \
            self.smpl(betas.repeat(self.pose.shape[0], 1), self.pose,
                      self.trans, offsets.repeat(self.pose.shape[0], 1, 1))
        total_naked_vertices, _, _, _ = \
            self.smpl(
                betas.repeat(self.pose.shape[0], 1), self.pose,
                self.trans,
                torch.zeros_like(offsets).repeat(self.pose.shape[0], 1, 1))

        body_joints = torch.tensordot(
            self.body25_reg_tensor, total_naked_vertices,
            dims=([1], [1])).transpose(0, 1)
        face_joints = torch.tensordot(
            self.face_reg_tensor, total_naked_vertices,
            dims=([1], [1])).transpose(0, 1)

        body_joints_h = torch.cat(
            [body_joints, torch.ones_like(body_joints[..., -1:])], dim=2)
        face_joints_h = torch.cat(
            [face_joints, torch.ones_like(face_joints[..., -1:])], dim=2)

        perspective_body_joints = torch.matmul(body_joints_h,
                                               self.perspective_matrix)
        perspective_face_joints = torch.matmul(face_joints_h,
                                               self.perspective_matrix)

        body_joints2d_pred = perspective_body_joints / \
            perspective_body_joints[..., -1:]
        face_joints2d_pred = perspective_face_joints / \
            perspective_face_joints[..., -1:]

        output = {
            'body_joints2d_pred': body_joints2d_pred,
            'face_joints2d_pred': face_joints2d_pred,
            'total_naked_vertices': total_naked_vertices,
            'pose': self.pose,
            'trans': self.trans,
        }
        return output

    def load_parameters(self, load_path, is_load_from_octopus = False):
        # ------------------------------------------- texture
        texture_size = 512  # 1024

        texture = numpy2tensor(
            np.ones([1, texture_size, texture_size, 3]) * 0.5).to(self.device)
        # 0.5 is set to be initial value

        self.texture_parameters = nn.Parameter(texture, requires_grad=True)
        self.vis_texture = numpy2tensor(
            np.ones([1, texture_size, texture_size, 3])).to(self.device)

        # ------------------------------------------- geometry

        mean_a_pose = np.load('./assets/mean_a_pose.npy')
        mean_a_pose[:, :3] = 0.

        self.mean_a_pose = \
            numpy2tensor(mean_a_pose.reshape([-1, 3])).to(self.device)



        if is_load_from_octopus:
            with open(f"{load_path}param.pkl",'rb') as file:
                params_data = pkl.load(file, encoding='latin1')
            pose = numpy2tensor(params_data['poses'][:self.length]).to(self.device)  # .cuda()
            betas = numpy2tensor(params_data['betas'][:1]).to(self.device)
            trans = numpy2tensor(params_data['trans'][:self.length]).to(self.device)

            # offset = numpy2tensor(params_data['offsets'][:1]).to(self.device)
            offset = numpy2tensor(np.zeros([1, 6890, 3])).to(self.device)
        else:
            pose = numpy2tensor(np.load(f'{load_path}pose.npy')).to(self.device)
            betas = numpy2tensor(np.load(f'{load_path}betas.npy')).to(self.device)
            trans = numpy2tensor(np.load(f'{load_path}trans.npy')).to(self.device)

            # offset = numpy2tensor(np.load(f'{load_path}offsets.npy')).to(self.device)
            offset = numpy2tensor(np.zeros([1, 6890, 3])).to(self.device)

        if self.use_posematrix is True:
            pose = rodrigues(pose.view(-1, 1, 3)) \
                .reshape(pose.shape[0], -1, 3, 3)

        self.pose = nn.Parameter(pose, requires_grad=True)
        self.betas = nn.Parameter(betas, requires_grad=True)
        self.trans = nn.Parameter(trans, requires_grad=True)
        self.offsets = nn.Parameter(offset, requires_grad=True)

    def save_parameters(self, save_path,
                        to_save_parameters:list=['pose', 'betas', 'trans', 'offsets'],
                        epoch_idx=None):

        save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        pose = tensor2numpy(self.so_pose) if self.use_posematrix \
            else tensor2numpy(self.pose)
        betas = tensor2numpy(self.betas)
        trans = tensor2numpy(self.trans)


        if 'pose' in to_save_parameters:
            np.save(f'{save_path}pose.npy', pose)
        if 'betas' in to_save_parameters:
            np.save(f'{save_path}betas.npy', betas)
        if 'trans' in to_save_parameters:
            np.save(f'{save_path}trans.npy', trans)


    def write_tpose_obj(self, save_path):
        write_obj(
            vs=self.v_shaped_personal[0],
            vt=self.vts[0],
            fs=self.fs[0],
            ft=self.fts[0],
            path=f'{save_path}vis.obj',
            write_mtl=True)


class downsample(nn.Module):
    def __init__(self, in_ch, out_ch, norm = 'instance'):
        super(downsample, self).__init__()
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d(out_ch)
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d(out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
            norm_layer,
            nn.LeakyReLU(0.2, inplace=True)
        )
        pass

    def forward(self, x):
        y = self.conv(x)
        return y

class upsample(nn.Module):
    def __init__(self, in_ch, out_ch, concat=True, final=False, norm='instance'):
        super(upsample, self).__init__()
        self.concat = concat
        self.final = final

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d(out_ch)
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d(out_ch)
        if self.final:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=1, padding=1, output_padding=0),
                # nn.Upsample(mode='bilinear', scale_factor=2),
                # nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(out_ch), #nn.InstanceNorm2d(out_ch),
                nn.Tanh()
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=1, padding=1, output_padding=0),
                # nn.Upsample(mode='bilinear', scale_factor=2),
                # nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
                norm_layer, # nn.BatchNorm2d(out_ch),
                # nn.LeakyReLU(0.2, inplace=True)
                nn.ReLU(True)
            )
        pass

    def forward(self, x1, x2):
        if self.concat:
            x1 = torch.cat((x2, x1), dim=1)
        y = self.conv(x1)
        return y

        pass

class shape_encoder_unet(nn.Module):
    def __init__(self, input_channels, output_channels, norm='instance'):
        super(shape_encoder_unet, self).__init__()

        self.down1 = downsample(input_channels, 64, norm=norm)
        self.down2 = downsample(64, 128, norm=norm)
        self.down3 = downsample(128, 256, norm=norm)
        self.down4 = downsample(256, 512, norm=norm)
        self.down5 = downsample(512, 512, norm=norm)

        self.up1 = upsample(512, 512, concat=False, norm=norm)
        self.up2 = upsample(1024, 512, norm=norm)
        self.up3 = upsample(768, 256, norm=norm)
        self.up4 = upsample(384, 128, norm=norm)
        self.up5 = upsample(128, output_channels, concat=False, final=True, norm=norm)
        pass
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        y = self.up1(x5, None)
        y = self.up2(y, x4)
        y = self.up3(y, x3)
        y = self.up4(y, x2)
        y = self.up5(y, None)
        return y

class add_coords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(add_coords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret

class dynamic_offsets_network(nn.Module): #
    def __init__(self, device, name, batch_size=1, mode='train', isupsample=True, use_posematrix=False,
                dataset_len=0, img_size = [1024, 1024],
                 r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], t = [0., 0., 0.],
                 K = None, f = [1024, 1024], c = [512, 512],):
        super(dynamic_offsets_network, self).__init__()

        self.device = device
        self.mode = mode
        self.name = name

        self.norm = 'batch' # 'instance'
        self.batch_size = batch_size
        self.dataset_len = 0

        self.isupsample = isupsample

        self.use_posematrix = use_posematrix

        self.build_basefiles()
        # self.load_parameters()

        self.renderer, self.normal_renderer, self.mesh_renderer  = \
            build_differential_renderer(device, r, t, K, f, c, img_size)
        self.smpl = smpl_tpose_layer(device=self.device, isupsample=self.isupsample, use_posematrix=self.use_posematrix)

        self.build_network()

    def build_basefiles(self):
        f, ft, vt = get_f_ft_vt(self.isupsample)

        self.fs = numpy2tensor(np.expand_dims(f, axis=0), np.long).repeat([1, 1, 1]).to(self.device)
        self.fts = numpy2tensor(np.expand_dims(ft, axis=0), np.long).repeat([1, 1, 1]).to(self.device)
        self.vts = numpy2tensor(np.expand_dims(vt, axis=0), np.float32).repeat([1, 1, 1]).to(self.device)

        # -------------------------------
        texture_size = 512
        texture = torch.from_numpy(np.ones([1, texture_size, texture_size, 3]).astype(np.float32) * 0.5).to(self.device)
        self.texture_parameters = nn.Parameter(texture, requires_grad=True)

        # self.texture = TexturesUV(self.texture_parameters, self.fts, self.vts) # .repeat(batch_size, 1, 1, 1)

        self.vis_texture = numpy2tensor(np.ones([1, texture_size, texture_size, 3])).to(self.device)

    def load_parameters(self, load_path):
         # = f'./results/diff_optiming/{self.name}/'

        self.pose = numpy2tensor(np.load(load_path + 'pose.npy')).to(self.device)
        self.betas = numpy2tensor(np.load(load_path + 'betas.npy')).to(self.device)
        self.trans = numpy2tensor(np.load(load_path + 'trans.npy')).to(self.device)
        # self.mean_offset = torch.from_numpy(np.load(_load_path + 'offsets_{}.npy'.format(name)).astype(np.float32)).cuda()
        self.mean_offset = None
        self.perspective_matrix = cal_perspective_matrix().to(self.device)


        vertex_label = LaplacianLoss(device=self.device, isupsample=self.isupsample).v_ids
        self.hand_index = np.concatenate([vertex_label['left_hand'], vertex_label['right_hand']], axis=0)

    def build_network(self):
        # main networks
        self.pose_encoder = shape_encoder_unet(3, 64, norm=self.norm) # _UNet(3, 128, norm='instance')

        self.geometry_parameters = nn.Parameter(numpy2tensor(np.zeros([1, 64, 128, 128])).to(self.device), requires_grad=True)

        if self.norm == 'batch':
            norm_layer = nn.BatchNorm2d(64)
        elif self.norm == 'instance':
            norm_layer = nn.InstanceNorm2d(64)
        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1, 2),
            norm_layer,
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),
            norm_layer,
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),
            # nn.InstanceNorm2d(64),
            nn.Tanh(),
        )

        self.add_coords = add_coords()

        sample_point_path = './assets/smpl/sampled_points.npy' if self.isupsample == False \
            else './assets/upsmpl/sampled_points.npy'
        sample_point = np.expand_dims(np.expand_dims(np.load(sample_point_path), axis=0), axis=0)
        sample_point = sample_point * 2 - 1
        self.sample_point = torch.from_numpy(sample_point).to(self.device)


        self.skips = [3]
        self.shape_decoder = nn.ModuleList(
            [nn.Linear(64 + 64 + 2, 256)] +
            [nn.Linear(256, 256) if i not in self.skips else nn.Linear(256 + 64 + 64 + 2, 256) for i in range(8 - 2)] # +
        )

        self.offsets_decoder = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3), nn.Tanh())
        self.normal_decoder = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3), nn.Tanh())

        # init_weights(self.pose_encoder, 'kaiming')
        # init_weights(self.shape_decoder, 'kaiming')
        # init_weights(self.offsets_decoder, 'kaiming')
        # init_weights(self.normal_decoder, 'kaiming')

        # ---------------------


    def forward(self, item, naked_vertice_uv):

        pose_feature = self.pose_encoder(naked_vertice_uv)

        # geometry_feature = self.geometry_encoder(self.geometry_parameters) #
        geometry_feature = self.geometry_parameters.repeat(pose_feature.shape[0], 1, 1, 1)

        shape_feature = torch.cat([pose_feature, geometry_feature], dim=1)

        shape_feature = self.add_coords(shape_feature)

        # if self.isupsample == False:
        sampled_shape_feature = F.grid_sample(shape_feature, self.sample_point.repeat(pose_feature.shape[0], 1, 1, 1)) # .repeat(pose_feature.shape[0], 1, 1, 1)

        input_feature = sampled_shape_feature.squeeze(2).permute(0, 2, 1)
        h = input_feature
        for i, l in enumerate(self.shape_decoder):
            h = self.shape_decoder[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_feature, h], dim=-1)

        offsets = self.offsets_decoder(h) / 10.0
        normal = self.normal_decoder(h)

        mask_hand = False
        if mask_hand:
            offsets[:,self.hand_index,:] = torch.from_numpy(np.array([0., 0. ,0.]).astype(np.float32)).cuda()
        # normal_length = torch.sqrt(torch.sum(torch.square(normal), dim=2)).unsqueeze(dim=-1).repeat(1,1,3)
        unit_normal = F.normalize(normal, dim=2) # normal / normal_length

        pose = self.pose.index_select(dim=0, index=item)
        trans = self.trans.index_select(dim=0, index=item)
        betas = self.betas.repeat(pose.shape[0], 1)  # .index_select(dim=0, index=item)

        vertices, v_shaped, v_shaped_personal, v_offsets = self.smpl(betas, pose, trans, offsets)
        mean_v_shaped_personal = None
        # mean_vertices, _, mean_v_shaped_personal, _ = self.smpl(betas, pose, trans, self.mean_offset)
        # v, v_shaped, v_shaped_personal = self.smpl(self.betas[i:i + 1], self.pose[i:i + 1], self.trans[i:i + 1], offsets)


        # ---------------------
        v = vertices

        mesh = Meshes(v, self.fs.repeat(v.shape[0], 1, 1))
        texture = TexturesUV(self.texture_parameters.repeat(v.shape[0], 1, 1, 1),
                             self.fts.repeat(v.shape[0], 1, 1),
                             self.vts.repeat(v.shape[0], 1, 1))
        mesh.textures = texture
        images = self.renderer(mesh)
        normal_images = self.normal_renderer(mesh)
        images = torch.flip(images, [2])
        normal_images = torch.flip(normal_images, [2])

        mesh_images = []
        if self.mode == 'test':
            texture = TexturesUV(self.vis_texture.repeat(v.shape[0], 1, 1, 1),
                                 self.fts.repeat(v.shape[0], 1, 1),
                                 self.vts.repeat(v.shape[0], 1, 1))
            mesh.textures = texture
            mesh_images = self.mesh_renderer(mesh)
            mesh_images = torch.flip(mesh_images, [2])

        # return offsets, pred_imgs, pred_sil_imgs, v_shaped, v_shaped_personal
        if self.mode == 'train':
            output = {
                'pred_images':images[:,:,:,:3],
                'pred_sil_images':images[:,:,:,3],
                'v_shaped':v_shaped,
                'v_shaped_personal':v_shaped_personal,
                'mean_v_shaped_personal':mean_v_shaped_personal,
                'v':v,
                'offsets':offsets,
                'mean_offset': self.mean_offset,
                'pred_normal_images':normal_images,

            }
            return output
            # return images[:, :, :, :3], images[:, :, :, 3], v_shaped, v_shaped_personal, mean_v_shaped_personal,\
            #        v, offsets, self.mean_offset, normal_images
        elif self.mode == 'test':
            output = {
                'pred_imgs': images[:, :, :, :3],
                'pred_sil_imgs': images[:, :, :, 3],
                'v_shaped': v_shaped,
                'v_shaped_personal': v_shaped_personal,
                'v_offsets':v_offsets,
                'v':v,
                'offsets':offsets,
                'mesh_images':mesh_images
            }
            return output
            # return images[:, :, :, :3], images[:, :, :, 3], v_shaped, v_shaped_personal, v_offsets, \
            #        v, offsets, mesh_images

    def write_v_obj(self, v, path):
        write_obj(vs=v, fs=self.fs[0], path=path)

    def save_parameters(self, save_path):

        pose = tensor2numpy(self.so_pose) if self.use_posematrix else tensor2numpy(self.pose)
        betas = tensor2numpy(self.betas)
        trans = tensor2numpy(self.trans)

        np.save(save_path + 'pose.npy', pose)
        np.save(save_path + 'betas.npy', betas)
        np.save(save_path + 'trans.npy', trans)
