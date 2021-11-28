import numpy as np
import matplotlib.pyplot as plt

import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from dataset import LiverDataset2

from mIou import *
import os
import cv2
import PIL.Image as Image
# 是否使用cuda
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



def get_data(i):
    import dataset
    imgs = dataset.make_dataset(r"D:\\project\\testnew\\data")
    imgx = []
    imgy = []
    for img in imgs:
        imgx.append(img[0])
        imgy.append(img[1])
    return imgx[i], imgy[i]


def train_model(model, criterion, optimizer, dataload, num_epochs=1):
    epoch_loss_group = []
    epoch_group = []
    for epoch in range(num_epochs):
        epoch_group.append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        print(dt_size)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            predicted = torch.squeeze(predicted).cpu().numpy()
            # print(labels.shape)
            print(predicted.shape)
            # print(outputs)
            #
            # print(predicted)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/dt_size))
        epoch_loss_group.append(epoch_loss/dt_size)
    plt.plot(epoch_group,epoch_loss_group)
    plt.title("Loss")
    plt.xlabel('loss vs. epoches')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("D:\\project\\loss.jpg")
    torch.save(model.state_dict(), r'D:\\project\\weights.pth')
    return model


# 训练模型
def train():
    model = Unet(3, 3).to(device)
    batch_size = args.batch_size
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset(r"D:\project\trainnew\b",r"D:\project\trainnew\f",r"D:\project\trainnew\t",
                                 transform=x_transforms,target_transform=y_transforms)
    # liver_dataset = LiverDataset(r"C:\Users\Lsx\Desktop\train\b",r"C:\Users\Lsx\Desktop\train\f",r"C:\Users\Lsx\Desktop\train\t",
    #                              transform=x_transforms,target_transform=y_transforms)
    # liver_dataset = LiverDataset2(r"D:\project\traindata",
    #                              transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders)



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

    y_transforms = None


    # 参数解析器,用来解析从终端读取的命令
    parse = argparse.ArgumentParser()
    # parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train or test", default="train")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckp", type=str, help="D:\project\weights.pth")
    args = parse.parse_args()

    # train
    train()






