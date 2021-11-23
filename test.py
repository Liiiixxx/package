import numpy as np
import matplotlib.pyplot as plt

import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from dataset import LiverDataset3

from mIou import *
import os
import cv2
import PIL.Image as Image
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colormap = [(0,0,255),(0,255,0),(255,0,0)]
def test():
    model = Unet(3, 3).to(device)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))  # 载入训练好的模型
    model.eval()
    with torch.no_grad():

        image_g_path = 'D:\\project\\testnew\\whole1'
        dirs = os.listdir(image_g_path)
        for i in range(len(dirs)):
            file = dirs[i]
            img_path = 'D:\\project\\testnew\\whole1\\' + file + '\\image.png'
            mask_path = 'D:\\project\\testnew\\whole1\\' + file + '\\mask.png'
            prd_savepath = 'D:\\project\\testnew\\whole1\\' + file + '\\prd.png'
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            h,w,_ = img.shape
            r = 256
            stride = 256  # 3/4
            padding_h = (h//stride + 1) * stride
            padding_w = (w//stride + 1) * stride
            print(padding_h)
            padding_img = np.zeros((int(padding_h), int(padding_w), 3), dtype=np.uint8)
            padding_mask = np.zeros((int(padding_h), int(padding_w), 3), dtype=np.uint8)
            padding_img[0:h, 0:w, :] = img[:, :, :]
            padding_mask[0:h, 0:w, :] = mask[:, :, :]
            # padding_img = padding_img.astype("float") / 255.0
            # padding_img = np.array(padding_img)
            whole = np.zeros((int(padding_h), int(padding_w),3),dtype=np.uint8)
            for i in range(int(padding_h // stride)):
                for j in range(int(padding_w // stride)):
                    crop = padding_img[i * stride:i * stride + r, j * stride:j * stride + r]
                    print(crop.shape)
                    ch, cw,_ = crop.shape
                    if ch != 256 or cw != 256:
                        print('invalid size!')
                        continue
                    mask_crop = padding_mask[i * stride:i * stride + r, j * stride:j * stride + r]
                    tensor_type = LiverDataset3(crop,mask_crop,transform=x_transforms,
                                 target_transform=y_transforms)

                    for img_tensor,tar in tensor_type:

                        img_tensor = img_tensor.to(device)
                        tar = tar.to(device)
                        print(img_tensor.shape)
                        print(tar.shape)

                        img_tensor = img_tensor.unsqueeze(0)
                        tar = tar.unsqueeze(0) # 没有dataloader
                        pred = model(img_tensor)
                        # pred = torch.squeeze(pred)
                        _, predicted = torch.max(pred.data, dim=1)
                        predicted = torch.squeeze(predicted).cpu().numpy() # （256,256）
                        tar = torch.squeeze(tar).cpu().numpy()
                        dice = Dice(predicted, tar)
                        label2rgb = np.zeros((256,256,3),dtype=np.uint8)
                        for c in range(3):
                            label2rgb[:, :, 0] += ((predicted[:, :] == c) * (colormap[c][0])).astype('uint8')
                            label2rgb[:, :, 1] += ((predicted[:, :] == c) * (colormap[c][1])).astype('uint8')
                            label2rgb[:, :, 2] += ((predicted[:, :] == c) * (colormap[c][2])).astype('uint8')
                        # a,b = predicted.shape
                        # count = 0
                        # for x in range(a):
                        #     for y in range(b):
                        #         count+=1
                        #         if(predicted[x][y]==2):
                        #             label2rgb[x][y] = [0,0,255]
                        #         if (predicted[x][y] == 1):
                        #             label2rgb[x][y] = [0, 255, 0]
                        #         if(predicted[x][y]==0):
                        #             label2rgb[x][y ] = [255,0,0]

                        whole[i * stride:i * stride + r, j * stride:j * stride + r] = label2rgb[:, :]
                        label2rgb = np.array(label2rgb)
                        label2rgb = Image.fromarray(label2rgb)
                        plt.imshow(label2rgb)
                        plt.pause(1)
                        plt.show
            whole = np.array(whole)
            whole = Image.fromarray(whole)
            plt.imshow(whole)
            plt.show
            Image.save(prd_savepath,whole)


def Dice(inp, target, eps=1):
    # 抹平了，弄成一维的
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    # 计算交集中的数量
    overlap = np.sum(input_flatten * target_flatten)
    # 返回值，让值在0和1之间波动
    return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)


if __name__ == "__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = None

    # 参数解析器,用来解析从终端读取的命令
    parse = argparse.ArgumentParser()
    # parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train or test", default="train")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckp", type=str, help="D:\project\weights.pth")
    args = parse.parse_args()
    args.ckp = r"D:\project\weights.pth"
    test()