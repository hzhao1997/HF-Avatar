import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from utils.general import init_weights

class perceptual_network(nn.Module):
    def __init__(self, opt=None):
        super(perceptual_network, self).__init__()
        block = torchvision.models.vgg16(pretrained=True).features[:].eval()
        for p in block:
            p.requires_grad = False
        self.block = block # torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self.opt = opt

    def _forward(self, x):
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(self.opt.render_size, self.opt.render_size), align_corners=False)
        for block in self.block:
            x = block(x)
        return x


    def forward(self, x):

        seq = []
        seq_idx = ['3', '8', '15', '22']
        #x = (x - self.mean) / self.std
        #x = self.transform(x, mode='bilinear', size=(512, 512), align_corners=False)

        for name, block in self.block._modules.items():
            x = block(x)
            if(name in seq_idx):
                seq.append(x)

        return seq


class single_layer_texture(nn.Module):
    def __init__(self, W, H):
        super(single_layer_texture, self).__init__()
        self.layer1 = nn.Parameter(torch.FloatTensor(1, 1, W // 8, H // 8).zero_())

    def forward(self, x):
        batch = x.shape[0]
        x = x * 2.0 - 1.0
        y = F.grid_sample(self.layer1.repeat(batch,1,1,1), x, )  # align_corners=True
        return y


class Texture(nn.Module):
    def __init__(self, W, H, feature_num):
        super(Texture, self).__init__()
        self.feature_num = feature_num

        self.layer1 = nn.ParameterList()

        self.textures = nn.ModuleList([single_layer_texture(W, H) for i in range(feature_num)])
        for i in range(self.feature_num):
            self.layer1.append(self.textures[i].layer1)


    def forward(self, x):
        y_i = []
        for i in range(self.feature_num):
            y = self.textures[i](x)
            y_i.append(y)
        y = torch.cat(tuple(y_i), dim=1)
        return y

class neural_texture_network(nn.Module):
    def __init__(self, W=2048, H=2048, feature_num=16):
        super(neural_texture_network, self).__init__()
        self.feature_num = feature_num

        self.texture = Texture(W, H, feature_num)

    def forward(self, uv_map):
        x = self.texture(uv_map)
        x = torch.clamp(x, 0.0, 1.0)
        return x[:, 0:3, :, :]

# --------------------------------

class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch'):
        super(down, self).__init__()
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d(out_ch)
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d(out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            norm_layer,
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, concat=True, final=False, norm='batch'):
        super(up, self).__init__()
        self.concat = concat
        self.final = final
        if norm == 'batch':
            norm_layer = nn.BatchNorm2d(out_ch)
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d(out_ch)
        if self.final:
            self.conv = nn.Sequential(
                # nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(out_ch), #nn.InstanceNorm2d(out_ch),
                nn.Tanh()
            )
        else:
            self.conv = nn.Sequential(
                # nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
                norm_layer, # nn.BatchNorm2d(out_ch),
                # nn.LeakyReLU(0.2, inplace=True)
                nn.ReLU(True)
            )

    def forward(self, x1, x2):
        if self.concat:
            x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv(x1)
        return x1

class unet(nn.Module):
    def __init__(self, input_channels, output_channels, norm='batch'):
        super(unet, self).__init__()
        self.down1 = down(input_channels, 64, norm=norm)
        self.down2 = down(64, 128, norm=norm)
        self.down3 = down(128, 256, norm=norm)
        self.down4 = down(256, 512, norm=norm)
        self.down5 = down(512, 512, norm=norm)

        self.up1 = up(512, 512, concat=False, norm=norm)
        self.up2 = up(1024, 512, norm=norm)
        self.up3 = up(768, 256, norm=norm)
        self.up4 = up(384, 128, norm=norm)
        self.up5 = up(128, output_channels, concat=False, final=True, norm=norm)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, None)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, None)

        return x

class ref_concat_unet(nn.Module):
    def __init__(self, input_channels, output_channels, norm='batch'):
        super(ref_concat_unet, self).__init__()

        self.down1 = down(input_channels, 64, norm=norm)
        self.down2 = down(64, 128, norm=norm)
        self.down3 = down(128, 256, norm=norm)
        self.down4 = down(256, 512, norm=norm)
        self.down5 = down(512, 512, norm=norm)

        self.up1 = up(1024, 512, concat=False, norm=norm)
        self.up2 = up(1024, 512, norm=norm)
        self.up3 = up(768, 256, norm=norm)
        self.up4 = up(384, 128, norm=norm)
        self.up5 = up(128, output_channels, concat=False, final=True, norm=norm)

        self.ref_encode1 = down(4, 64, norm=norm)
        self.ref_encode2 = down(64, 128, norm=norm)
        self.ref_encode3 = down(128, 256, norm=norm)
        self.ref_encode4 = down(256, 512, norm=norm)
        self.ref_encode5 = down(512, 512, norm=norm)

    def forward(self, x, ref):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        ref_latent_code = self.ref_encode5(self.ref_encode4(
            self.ref_encode3(self.ref_encode2(self.ref_encode1(ref)))))
        fused_code = torch.cat([x5, ref_latent_code], dim=1)
        # fused_code = adaptive_instance_normalization(x5, ref_latent_code)

        x = self.up1(fused_code, None)
        # x = self.up1(x5, None)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, None)
        return x


class neural_rendering_network(nn.Module):
    def __init__(self,  W=2048, H=2048, feature_num=16, norm='batch'):
        super(neural_rendering_network, self).__init__()
        self.feature_num = feature_num

        self.texture = Texture(W, H, feature_num)

        self.renderer = ref_concat_unet(feature_num, 3, norm=norm)  # norm=norm

        init_weights(self.renderer, 'kaiming')

    def forward(self, uv_map, ref):
        x = self.texture(uv_map)
        x = torch.clamp(x, 0.0, 1.0)

        y = self.renderer(x, ref)
        y = torch.clamp(y, 0.0, 1.0)

        return x[:, 0:3, :, :], y
