""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self,
                 latent_channel,
                 latent_dim,
                 hidden_dim,
                 text_channel,
                 text_dim
                 ):
        super(AttnBlock, self).__init__()
        self.latent_channel = latent_channel
        self.latent_dim = latent_dim
        self.latent_dim_dim = latent_dim * latent_dim
        self.hidden_dim = hidden_dim
        self.text_channel = text_channel
        self.text_dim = text_dim

        self.text_in = nn.Conv2d(in_channels=self.text_channel, out_channels=self.latent_channel, kernel_size=1)

        self.latent_q = nn.Linear(self.latent_dim_dim, hidden_dim)
        self.text_k = nn.Linear(text_dim, hidden_dim)
        self.text_v = nn.Linear(text_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, self.latent_dim_dim)

    def forward(self, latent, text):
        text = text.reshape(text.shape[0], 8, 32, 48)
        text = self.text_in(text)
        text = torch.flatten(text, start_dim=2)

        latent = torch.flatten(latent, start_dim=2)

        latent_q = self.latent_q(latent)
        text_k = self.text_k(text)
        text_v = self.text_v(text)

        weight = torch.matmul(latent_q, text_k.transpose(1, 2))
        weight = torch.softmax(weight, dim=-1)

        out = torch.matmul(weight, text_v)
        out = self.out(out)
        out = out.reshape(latent.shape[0], self.latent_channel, self.latent_dim, self.latent_dim)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)


        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
