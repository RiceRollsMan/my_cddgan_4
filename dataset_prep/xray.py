import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import torchvision.transforms as transforms
import numpy as np


class Xray(Dataset):
    def __init__(self, root, train=True, target_transforms=None):
        self.transforms = target_transforms
        self.train = train
        self.json_path = os.path.join(root, 'img2text.json')
        self.label = os.path.join(root, 'bert_textencoder')
        with open(self.json_path) as f:
            d = json.load(f)
        if self.train:
            img_path = os.path.join(root, 'images\\train_img')
            label_path = os.path.join(self.label, 'train')
            data = d.get('train')
        else:
            img_path = os.path.join(root, 'images\\test_img')
            label_path = os.path.join(self.label, 'test')
            data = d.get('test')
        self.data = data
        self.length = len(data)
        self.img_path = img_path
        self.label_path = label_path

    def __getitem__(self, idx):
        img_name = self.data[idx]['filename']
        img_id = self.data[idx]['id']
        img_label = self.getBertTextEncoder(img_id)
        img = os.path.join(self.img_path, img_name)
        img = Image.open(img).convert('RGB')
        if self.transforms is None:
            self.transforms = transforms.ToTensor()
        img = self.transforms(img)
        return img, img_label

    def __len__(self):
        return len(os.listdir(self.img_path))

    def getBertTextEncoder(self, id):
        save_dir = os.path.join(f"{self.label_path}\\{id}")
        BertTextEncoder = np.load(f'{save_dir}.npy')
        return BertTextEncoder

# root = 'D:\\Code\\data\\Xray\\Xray'
# xray = Xray(root)
# img, img_label, img_name = xray[0]
# print(img_name)
# img_pil = transforms.ToPILImage()(img)
# img_pil.show()


