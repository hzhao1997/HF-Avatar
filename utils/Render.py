import os
import torch
# from skimage.io import imread

from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
)

import numpy as np
import cv2
import torch.nn as nn
from utils.general import setup_seed, numpy2tensor, tensor2numpy

class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images / 2.0

class Render:
    def __init__(self, img_size, batch_size, device, r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t = [0., 0., 0.],
                 K = None, f = [1024, 1024], c = [512, 512], tex_path = None, bg_color=(0.0, 0.0, 0.0), isupsample=False):
        R = numpy2tensor(np.expand_dims(np.array(r), axis=0))
        T = numpy2tensor(np.expand_dims(np.array(t), axis=0)) # 0.0,0.2,2.3
        if K is not None:
            K = numpy2tensor(np.expand_dims(np.array(K), axis=0))

        width = img_size[0]  # 1080.0
        height = img_size[1]  # 1080.0

        # self.f = img_size[0]
        focal = numpy2tensor(np.expand_dims(np.array([f[0], f[1]]),axis=0))
        center = numpy2tensor(np.expand_dims(np.array([c[0], c[1]]),axis=0))
        Size = numpy2tensor(np.expand_dims(np.array([width, height]),axis=0)) # width, height

        if K is not None:
            cam = PerspectiveCameras(device=device, R=R, T=T, K=K, focal_length=focal, principal_point=center, image_size=Size)  # focal_length=focal, principal_point=center, image_size=Size
        else:
            cam = PerspectiveCameras(device=device, R=R, T=T, focal_length=focal, principal_point=center, image_size=Size)
        # Define the settings for rasterization and shading.
        raster_settings = RasterizationSettings(
            image_size=int(width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Place a point light in front of the object.
        # if light_position is not None:
        #    lights = PointLights(device=device, location=[light_position]) # [10.0, 10.0, 10.0]
        # self.lights = lights

        blend_params = BlendParams(1e-9, 1e-9, bg_color)
        cam_ = cam  # cameras

        self.uv_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cam_,  # cam,#cameras,#
                raster_settings=raster_settings
            ),

            shader=SimpleShader(
                device=device,
                blend_params=blend_params
            )
        )

        if isupsample == False:
            f = np.loadtxt('./assets/smpl_f_ft_vt/smpl_f.txt')
            ft = np.loadtxt('./assets/smpl_f_ft_vt/smpl_ft.txt')
            vt = np.loadtxt('./assets/smpl_f_ft_vt/smpl_vt.txt')
        else:
            f = np.loadtxt('./assets/upsmpl_f_ft_vt/smpl_f.txt')
            ft = np.loadtxt('./assets/upsmpl_f_ft_vt/smpl_ft.txt')
            vt = np.loadtxt('./assets/upsmpl_f_ft_vt/smpl_vt.txt')

        self.device = device
        self.batch_size = batch_size

        self.fs = numpy2tensor(np.expand_dims(f, axis=0), np.long).repeat([batch_size, 1, 1]).to(self.device)
        self.fts = numpy2tensor(np.expand_dims(ft, axis=0), np.long).repeat([batch_size, 1, 1]).to(self.device)
        self.vts = numpy2tensor(np.expand_dims(vt, axis=0), np.float32).repeat([batch_size, 1, 1]).to(self.device)

        if tex_path is not None:
            tex = np.expand_dims(cv2.imread(tex_path ), axis=0)
            texture_image = torch.from_numpy(tex / 255.0).type(torch.float).to(self.device) * 2.0
            self.texture = TexturesUV(texture_image, self.fts, self.vts)

        self.cnt = 0

    def load_texture(self, tex):
        # tex = np.expand_dims(cv2.imread(tex_path ), axis=0)

        tex = tex.permute(0,2,3,1)  # NxCxHxW -> NxHxWxC
        tex = tex[:, :, :, [2, 1, 0]]  # RGB -> BGR
        texture_image = tex.type(torch.float).to(self.device) * 2.0  # torch.from_numpy(tex/255.0)
        self.texture = TexturesUV(texture_image, self.fts, self.vts)

    def load(self, _tex):
        tex = _tex.clone()
        tex = tex[:, :, :, [2, 1, 0]]
        texture_image = tex.type(torch.float).to(self.device) * 2.0
        self.texture = TexturesUV(texture_image, self.fts, self.vts)


    # get uv img
    def get_uv_img(self, verts, output_path=None):
        vert = verts.clone()

        vert[:, :, 2] = vert[:, :, 2] * (-1.0)

        mesh = Meshes(vert, self.fs)
        mesh.textures = self.texture
        mesh = mesh.to(self.device)

        # image = self.renderer(mesh)  # image = self.renderer(mesh) # , lights=self.lights
        image = self.uv_renderer(mesh)

        uv = []
        # for idx in range(1):
        img = tensor2numpy(image)[0,:,:,:3]

        img = cv2.flip(img, 1)
        if output_path is not None:
            os.makedirs(f"./results/{output_path}", exist_ok=True)
            cv2.imwrite(f"./results/{output_path}/uv_renderd_{self.cnt + 1}.png", img * 255)
        else:
            cv2.imwrite(f"./results/vis_uv/uv_renderd_{self.cnt + 1}.png", img * 255)

        img = img[:,:,::-1]
        img = img[:,:,0:2]
        uv.append(img)
        # np.save()
        self.cnt = self.cnt + 1

        return uv



