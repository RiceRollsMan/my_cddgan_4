import torch
import torch.nn as nn
from model.unet_parts import *

vgg_cfgs = [2, 2, 'M', 4, 4, 'M', 8, 8, 8, 'M', 8, 8, 8, 'M']


def vgg_block(in_channels):
    main = []
    for layer in vgg_cfgs:
        if layer == 'M':
            main += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
            main += [conv2d, nn.ReLU(True)]
            in_channels = layer
    return nn.Sequential(*main)


class UNet(nn.Module):
    def __init__(self, in_channels, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))

        self.ca1 = (AttnBlock(64, 64, 1024, 8, 384))
        self.down1 = (Down(64, 128))

        self.ca2 = (AttnBlock(128, 32, 512, 8, 384))
        self.down2 = (Down(128, 256))

        self.ca3 = (AttnBlock(256, 16, 256, 8, 384))
        self.down3 = (Down(256, 512))

        self.ca4 = (AttnBlock(512, 8, 256, 8, 384))
        self.down4 = (Down(512, 1024))

        self.ca5 = (AttnBlock(1024, 4, 256, 8, 384))

        self.up1 = (Up(1024, 512, bilinear))
        self.ca6 = (AttnBlock(512, 8, 256, 8, 384))

        self.up2 = (Up(512, 256, bilinear))
        self.ca7 = (AttnBlock(256, 16, 256, 8, 384))

        self.up3 = (Up(256, 128, bilinear))
        self.ca8 = (AttnBlock(128, 32, 512, 8, 384))

        self.up4 = (Up(128, 64, bilinear))
        self.ca9 = (AttnBlock(64, 64, 1024, 8, 384))

        self.up5 = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, latent, text_features):
        latent1 = self.inc(latent)
        latent2 = self.ca1(latent1, text_features)

        ds1 = self.down1(latent2)
        latent3 = self.ca2(ds1, text_features)

        ds2 = self.down2(latent3)
        latent4 = self.ca3(ds2, text_features)

        ds3 = self.down3(latent4)
        latent5 = self.ca4(ds3, text_features)

        ds4 = self.down4(latent5)
        latent6 = self.ca5(ds4, text_features)

        x = self.up1(latent6, latent5)
        x = self.ca6(x, text_features)

        x = self.up2(x, latent4)
        x = self.ca7(x, text_features)

        x = self.up3(x, latent3)
        x = self.ca8(x, text_features)

        x = self.up4(x, latent2)
        x = self.ca9(x, text_features)

        x = self.up5(x)

        return x


class TextFeatureVGG(nn.Module):
    def __init__(self, in_channels, init_weight=False):
        super().__init__()
        self.main = vgg_block(in_channels=in_channels)

    def forward(self, text):
        # b, 1, 128, 768
        text_features = self.main(text)
        # b, 8, 8, 48
        text_features_reshape = torch.flatten(text_features, start_dim=2)
        # b, 8, 384
        return text_features_reshape


class latent_decoder(nn.Module):
    def __init__(self, latent_channel, latent_dim, img_channel, img_size):
        super().__init__()
        self.latent_channel = latent_channel
        self.latent_dim = latent_dim
        self.img_channel = img_channel
        self.img_size = img_size

        self.double_conv = nn.Sequential(
            nn.Conv2d(self.latent_channel, self.img_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.img_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.img_channel, self.img_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.img_channel),
            nn.ReLU(inplace=True)
        )
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, latent):
        latent = self.double_conv(latent)
        img = self.decode(latent)
        return img


class T2IGenerator(nn.Module):
    def __init__(self, modelConfig):
        super().__init__()
        self.label_channel = modelConfig['label_channel']
        self.latent_channel = modelConfig['latent_channel']
        self.latent_dim = modelConfig['latent_dim']
        self.img_size = modelConfig['img_size']
        self.img_channel = modelConfig['img_channel']

        self.textFeatures = TextFeatureVGG(in_channels=self.label_channel)
        self.unet = UNet(self.latent_channel)

        self.decoder = latent_decoder(self.latent_channel, self.latent_dim, self.img_channel, self.img_size)

    def forward(self, text, latent):
        text_features = self.textFeatures(text)

        unet_out = self.unet(latent, text_features)

        img = self.decoder(unet_out)
        return img

# modelConfig = {
#     "state": "train",  # train or test
#     "epoch": 200,
#     "batch_size": 32,
#     "dataset": 'xray',
#     # img
#     "img_size": 256,
#     "img_channel": 3,
#     # label
#     "label_channel": 1,
#     "label_features_channel": 8,
#     "label_seq_length": 512,
#     "label_embedding_dim": 768,
#     # latent
#     "latent_channel": 4,
#     "latent_dim": 64,
#     # time_steps
#     "num_time_steps": 4,
#     # optimizer
#     "lr_g": 5e-5,
#     "lr_d": 5e-5,
#     "beta_min": 0.1,
#     "beta_max": 20,
#     "use_geometric": False,
#     # file
#     'dataset_root': 'D:/Code/data/Xray/Xray',
#     "save_weight_dir": "./Checkpoints_xray/",
#     "save_img": "./save_images",
#     "device": "cuda",
#     "training_load_weight": None,
#     "labels": "no acute cardiopulmonary disease . the heart , pulmonary and mediastinum are within normal "
#               "limits . there is no pleural effusion or pneumothorax . there is no focal air space opacity to "
#               "suggest a pneumonia .",
#     "test_load_weight": "ckpt_199_.pt", }
#
# a = T2IGenerator(modelConfig)
# t = torch.randn([32, 1, 512, 768])
# print(a(t).shape)
