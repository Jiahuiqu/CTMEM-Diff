import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
import cv2
from torch.utils.data import Dataset
import os
from scipy.io import loadmat


class Latent(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential()


class Blur(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding_mode=None, padding=0):
        super(Blur, self).__init__()
        # factory_kwargs = {'device': device, 'dtype': dtype}

        self.padding_mode = padding_mode if padding_mode else 'replicate'
        self.chan = out_channels
        self.weight = nn.Parameter(torch.rand(kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding
        self.stride = stride if stride else 1

    def forward(self, x):
        weight = self.weight.unsqueeze(0).unsqueeze(0).repeat(self.chan, 1, 1, 1)
        return F.conv2d(F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode=self.padding_mode),
                        weight=weight, stride=self.stride, groups=self.chan)


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.downsample(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.nonlinear = nn.GELU()
        if in_chan != out_chan:
            self.conv_shortcut = nn.Conv2d(in_chan, out_chan, kernel_size=1, padding=0)

    def forward(self, x):
        h = self.conv1(x)
        h = self.nonlinear(h)
        h = self.conv2(h)

        if self.in_chan != self.out_chan:
            x = self.conv_shortcut(x)
        return x + h


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.first_layer = ResidualBlock(3, 4)
        self.mid_layer = ResidualBlock(4, 16)
        self.last_layer = ResidualBlock(16, 64)
        self.out_layer = ResidualBlock(64, 64)
        self.down = Downsample()

    def forward(self, x):
        h = self.first_layer(x)
        h = self.mid_layer(h)
        h = self.down(h)
        h = self.last_layer(h)
        h = self.down(h)
        h = self.out_layer(h)

        return h


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class LatentBlur(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2)) + x2
        x4 = self.act(self.conv4(x3)) + x1
        return x4


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = ResidualBlock(64, 64)
        self.mid_layer = ResidualBlock(64, 16)
        self.last_layer = ResidualBlock(16, 4)
        self.out_layer = ResidualBlock(4, 3)
        self.up1 = Upsample(64)
        self.up2 = Upsample(16)

    def forward(self, x):
        h = self.first_layer(x)
        h = self.up1(h)
        h = self.mid_layer(h)
        h = self.up2(h)
        h = self.last_layer(h)
        h = self.out_layer(h)
        return h


class Kernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.latent = LatentBlur()
    def forward(self, x):
        return self.decoder(self.latent(self.encoder(x)))


def psnr(img1, img2):
    p = np.array([])
    for sample in range(img2.shape[0]):
        for band in range(img2.shape[1]):
            mse = np.mean((img1[sample][band] - img2[sample][band]) ** 2)
            if mse < 1.0e-10:
                return 100
            p = np.append(p, 20 * math.log10(np.max(img2[sample][band]) / math.sqrt(mse)))
    return np.mean(p)


def SAM(x_true, x_pred):
    """calculate SAM method in code"""
    dot_sum = np.sum(x_true * x_pred, axis=1)
    norm_true = np.linalg.norm(x_true, axis=1)
    norm_pred = np.linalg.norm(x_pred, axis=1)
    res = np.arccos(dot_sum / norm_pred / norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0
    return np.mean(res) * 180 / np.pi


class DataLoad(Dataset):

    def __init__(self, root1, root2, mode):
        super(DataLoad, self).__init__()
        self.root1 = root1
        self.root2 = root2
        self.mode = mode

        if self.mode == "train":
            self.gtHS = os.listdir(os.path.join(self.root1, "train"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0]))
            self.HRMS = os.listdir(os.path.join(self.root2, "train", "gtHS"))
            self.HRMS.sort(key=lambda x: int(x.split(".")[0]))
            # self.LRMS = os.listdir(os.path.join(self.root, "train", "LRMS"))

        elif self.mode == "test":
            self.gtHS = os.listdir(os.path.join(self.root1, "test"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0]))
            self.HRMS = os.listdir(os.path.join(self.root2, "test", "gtHS"))
            self.HRMS.sort(key=lambda x: int(x.split(".")[0]))

    def __len__(self):
        return len(self.gtHS)

    def __getitem__(self, index):
        gt_HS, HR_MS = self.gtHS[index], self.HRMS[index]
        data_ref = loadmat(os.path.join(self.root1, self.mode, gt_HS))['lrMS'].reshape(3, 160, 160)
        # data_LRHS = loadmat(os.path.join(self.root2, self.mode, "LRHS", LR_HS))['LRHS'].reshape(102, 160, 160)
        data_HRMS = loadmat(os.path.join(self.root2, self.mode, "gtHS", HR_MS))['patch'].transpose(2, 0, 1)
        # data_LRMS = loadmat(os.path.join(self.root, self.mode, "LRMS", HR_MS))['LRMS'].reshape(4, 40, 40)
        return data_HRMS, data_ref
