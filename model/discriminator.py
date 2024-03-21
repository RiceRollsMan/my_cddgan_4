import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0')

vgg_cfgs = [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 16, 16, 16, 'M']


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


class T2IDiscriminator(nn.Module):
    def __init__(self, labels_channel=1, img_channel=3, modelConfig=None):
        super(T2IDiscriminator, self).__init__()
        self.text_ch = nn.Conv2d(in_channels=labels_channel, out_channels=img_channel, kernel_size=1)
        self.text_dim_two = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.text_up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.get_score = vgg_block(in_channels=img_channel+1)

        self.classifier = nn.Sequential(
            nn.Linear(16 * 16 * 16, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
        )

    def forward(self, image, text):
        # text = self.text_ch(text)
        text = self.text_dim_two(text)
        text = self.text_up_sample(text)

        combine_tensor = torch.cat((text, image), dim=1)
        combine_tensor = self.get_score(combine_tensor)
        combine_tensor = torch.flatten(combine_tensor, start_dim=1)

        combine_tensor = self.classifier(combine_tensor)
        return combine_tensor

# text = torch.randn([1, 1, 512, 768])
# img = torch.randn([1, 3, 256, 256])
#
# u = T2IDiscriminator(1, 3)
# print(u(img, text))
