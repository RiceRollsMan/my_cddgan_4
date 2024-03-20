import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
import torch.nn.functional as F

from dataset_prep.xray import Xray
from model.discriminator import T2IDiscriminator
from model.generator import T2IGenerator


def train(modelConfig):
    device = torch.device(modelConfig["device"])
    train_transform = transforms.Compose([
        transforms.Resize(modelConfig['img_size']),
        transforms.CenterCrop(modelConfig['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Xray(root=modelConfig['dataset_root'], target_transforms=train_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=modelConfig['batch_size'],
        shuffle=True,
    )

    generator = T2IGenerator(modelConfig).to(device)
    optimizerG = torch.optim.Adam(generator.parameters(), lr=modelConfig['lr_g'], weight_decay=1e-4)

    discriminator = T2IDiscriminator().to(device)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=modelConfig['lr_d'], weight_decay=1e-4)

    for i in range(modelConfig['epoch']):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataloader:
            for (images, labels) in tqdmDataloader:
                real_data = images.to(device)
                labels = labels.to(device)

                # train generator
                optimizerG.zero_grad()
                x_0_predict = generator(labels)

                output = discriminator(x_0_predict, labels)
                errG = F.softplus(-output)
                errG = errG.mean()

                errG.backward()
                optimizerG.step()

                # train discriminator
                optimizerD.zero_grad()

                x_0_predict = generator(labels)

                D_real = discriminator(real_data, labels)
                errD_real = F.softplus(-D_real)
                errD_real = errD_real.mean()
                errD_real.backward(retain_graph=True)

                output = discriminator(x_0_predict, labels)
                errD_fake = F.softplus(output)
                errD_fake = errD_fake.mean()
                errD_fake.backward()

                errD = errD_fake + errD_real
                optimizerD.step()

                tqdmDataloader.set_postfix(ordered_dict={
                    "epoch": i,
                    "loss_G": errG.data,
                    "loss_D_real": errD_real.data,
                    "loss_D_fake": errD_fake.data,
                    "loss_D": errD
                })

        if i > 150:
            torch.save(discriminator.state_dict(), os.path.join(
                modelConfig['save_weight_dir'], 'discriminator/ckpt_' + str(i) + "_.pt"
            ))
            torch.save(generator.state_dict(), os.path.join(
                modelConfig['save_weight_dir'], 'generator/ckpt_' + str(i) + "_.pt"
            ))
