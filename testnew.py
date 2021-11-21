import numpy as np
import matplotlib.pyplot as plt

import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from unet import Unet as unet
from utils import cvtColor, preprocess_input, resize_image


import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from dataset import LiverDataset2
import torch.nn.functional as F

from mIou import *
import os
import cv2
import PIL.Image as Image
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class U_net():
    _defaults = {
        "model_path": 'D:\\project\\weights.pth',
        "num_classes": 3,
        "input_shape": [256, 256],
        "blend": False,
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 3:
            self.colors = [(0, 0, 1), (0, 1, 0), (1, 1, 1)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # ---------------------------------------------------#
        #   获得模型
        # ---------------------------------------------------#
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        self.net = unet(3,3)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def test(self):
        x_transforms = transforms.Compose([
            transforms.ToTensor(),  # -> [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        ])

        # mask只需要转换为tensor
        y_transforms = transforms.ToTensor()
        liver_dataset = LiverDataset2(r"D:\project\testnew\data", transform=x_transforms,
                                      target_transform=y_transforms)
        dataloaders = DataLoader(liver_dataset, batch_size=1)
        plt.ion()  # 开启动态模式
        correct = 0
        total = 0
        with torch.no_grad():
            i = 0  # 验证集中第i张图
            miou_total = 0
            num = len(dataloaders)  # 验证集图片的总数
            print('num=%d' % num)
            dice_total = 0
            for x, target in dataloaders:
                x = x.to(device)
                target = target.to(device)
                y = self.net(x)
                pr = F.softmax(y.permute(1, 2, 0), dim=-1).cpu().numpy()
                pr = pr.argmax(axis=-1)
                seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
                for c in range(self.num_classes):
                    seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
                    seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
                    seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')
                image = Image.fromarray(np.uint8(seg_img))
                plt.imshow(image)
                plt.show()
