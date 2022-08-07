import numpy as np
import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F



def get_gaussian_kernal(ksize, sigma=None):
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    kernal = np.zeros((ksize, ksize))
    for i in range(ksize):
        for j in range(ksize):
            x = i - (ksize - 1) / 2
            y = j - (ksize - 1) / 2
            kernal[i][j] = 1.0 / (2 * np.pi * pow(sigma, 2) ) * np.exp( - ( pow(x, 2) + pow(y, 2)  ) / ( 2 * pow(sigma, 2)  ) )

    kernal /= np.sum(kernal)

    return kernal

def get_sharpen_kernel(ksize, sigma=None):

    kernal = get_gaussian_kernal(ksize, sigma)

    k = np.zeros((ksize, ksize))
    m = (int)((ksize-1)/2)
    k[m][m] = 1

    kernal = k - kernal
    kernal = kernal * 1.5

    kernal = k + kernal
    return kernal

def get_sharpen_bias_kernel(ksize, sigma=None, amplitude=1.5):

    kernal = get_gaussian_kernal(ksize, sigma)

    k = np.zeros((ksize, ksize))
    m = (int)((ksize-1)/2)
    k[m][m] = 1

    kernal = k - kernal
    kernal = kernal * amplitude # 3.0

    # kernal = k + kernal
    return kernal

class Get_sharpen(nn.Module):
    def __init__(self, kernel_name='n', amplitude = 1.5):
        super(Get_sharpen, self).__init__()

        if kernel_name == 'c':
            kernel = [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]
            ksize = 3
            self.kernel_size = ksize
            self.pad_size = (int)((self.kernel_size-1)/2)
        # kernel = [
        #     [0, 1, 0],
        #     [1, -4, 1],
        #     [0, 1, 0]
        # ]
        elif kernel_name == 'n':
            ksize = 17
            self.kernel_size = ksize
            self.pad_size = (int)((self.kernel_size-1)/2)
            # kernel = get_gaussian_kernal(ksize)
            # kernel = get_sharpen_kernel(ksize)
            kernel = get_sharpen_bias_kernel(ksize, amplitude=amplitude)

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()


    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]

        x0 = F.conv2d(x0.unsqueeze(1), self.weight, padding=self.pad_size)
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=self.pad_size)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=self.pad_size)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

def get_tex(path, name, shape=(256, 256), dilation=False):
    # tex = cv2.imread("./texture/{}.jpg".format(name))
    if path is not None:
        tex = cv2.imread(path + '{}.png'.format(name))
    else:
        tex = cv2.imread("./texture/{}.png".format(name))


    # if dilation == True:
    #     tex = process_tex(tex)

    tex = tex.astype(np.float32)
    tex = cv2.resize(tex, shape)
    tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
    tex = np.transpose(tex, [2,0,1])
    tex = tex / 255.0
    tex = torch.from_numpy(np.expand_dims(tex, axis=0))

    return tex

def save_tex(model, path=None, name=None, texture_dim = 16):
    os.makedirs(path, exist_ok=True)

    texture1 = [model.texture.textures[idx].layer1[0] for idx in range(texture_dim)]  # model.texture.layer4[idx][0]
    texture1 = torch.cat(texture1, dim=0)
    texture1_img = texture1[:3, :, :].detach().cpu().numpy()
    texture1_img = np.transpose(texture1_img, [1, 2, 0])
    if path is not None:
        cv2.imwrite(path + '{}.png'.format(name), texture1_img[:, :, ::-1] * 255)
    else:
        cv2.imwrite("./texture/{}.png".format(name), texture1_img[:, :, ::-1] * 255)

