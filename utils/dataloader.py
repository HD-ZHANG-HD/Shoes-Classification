import os
from PIL import Image
from PIL import ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch


from copy import deepcopy
import math
import numpy as np
import os
import random


class ShoesDataset(data.Dataset):
    def __init__(self, image_root, trainsize, dir, mode, id_path=None, nsample=None) -> None:
        self.trainsize = trainsize
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()
        if mode == 'train_l' and nsample is not None:
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]
            random.shuffle(self.ids)

        self.images = [image_root + os.sep +
                       line for line in self.ids]
        self.cls = [dir(f.split(os.sep)[-2]) for f in os.listdir(image_root)
                    if f.endswith('.jpg') or f.endswith('.png')]
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.toTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        cls = self.cls[index]
        image = self.transform(image)
        return image, cls

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.size


def get_loader(cfg):
    dataset = ShoesDataset(cfg['image_root'], cfg['trainsize'], cfg['dir'])
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg['batchsize'],
                            shuffle=cfg['shuffle'],
                            num_workers=cfg['num_workers'],
                            pin_memory=cfg['pin_memory'])
    return dataloader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [
            image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [
            gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
