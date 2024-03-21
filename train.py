import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
import torch.nn.functional as F

from dataset_prep.xray import Xray
from model.discriminator import T2IDiscriminator
from model.generator import T2IGenerator
from torch.utils.tensorboard import SummaryWriter


def train(modelConfig):
    tb_writer = SummaryWriter(log_dir='runs/xray_2')

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

    init_text = torch.zeros((1, 1, 128, 768), device=device)
    init_latent = torch.zeros((1, 4, 64, 64), device=device)
    tb_writer.add_graph(generator, [init_text, init_latent])

    discriminator = T2IDiscriminator().to(device)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=modelConfig['lr_d'], weight_decay=1e-4)

    for i in range(modelConfig['epoch']):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataloader:
            for (images, labels) in tqdmDataloader:
                real_data = images.to(device)
                labels = labels.to(device)
                latent = torch.randn([labels.shape[0],
                                      modelConfig['latent_channel'],
                                      modelConfig['latent_dim'],
                                      modelConfig['latent_dim']]).cuda()

                # train generator
                optimizerG.zero_grad()
                x_0_predict = generator(labels, latent)

                output = discriminator(x_0_predict, labels)
                errG = F.softplus(-output)
                errG = errG.mean()

                errG.backward()
                optimizerG.step()

                # train discriminator
                optimizerD.zero_grad()

                x_0_predict = generator(labels, latent)

                D_real = discriminator(real_data, labels)
                errD_real = F.softplus(-D_real)
                errD_real = errD_real.mean()
                # errD_real.backward(retain_graph=True)

                output = discriminator(x_0_predict, labels)
                errD_fake = F.softplus(output)
                errD_fake = errD_fake.mean()
                # errD_fake.backward()

                errD = errD_fake + errD_real
                errD.backward()
                optimizerD.step()

                tqdmDataloader.set_postfix(ordered_dict={
                    "epoch": i,
                    "loss_G": errG.data,
                    "loss_D_real": errD_real.data,
                    "loss_D_fake": errD_fake.data,
                    "loss_D": errD
                })

            tb_writer.add_scalar("errG", scalar_value=errG.data, global_step=i)
            tb_writer.add_scalar("errD", errD.data, i)
            tb_writer.add_scalar("errD_fake", errD_fake.data, i)
            tb_writer.add_scalar("errD_real", errD_real.data, i)

            torch.save(discriminator.state_dict(), os.path.join(
                modelConfig['save_weight_dir'], 'discriminator/ckpt_' + str(i) + "_.pt"
            ))
            torch.save(generator.state_dict(), os.path.join(
                modelConfig['save_weight_dir'], 'generator/ckpt_' + str(i) + "_.pt"
            ))

        # if i % 10 == 0:
        #     expand = np.load("test_label.npy").reshape([1, 1, 128, 768])
        #     latent = torch.randn([1, 4, 64, 64])
        #     test_img = generator(expand, latent)
        #     img_PIL = np.array(test_img.reshape([3, 256, 256]))
        #
        #     tb_writer.add_image('i', img_PIL, 10, dataformats="HWC")

    tb_writer.close()
