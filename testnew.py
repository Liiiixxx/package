import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
from mIou import *
import os
import cv2
import PIL.Image as Image
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colormap = [(255,255,255),(0,255,0),(255,0,0)]
writer = SummaryWriter('./runs/examples_test')

def colormaptolabel(group):
    colormap2label = np.zeros([len(group),len(group)])
    i=0
    colormap2label = colormap2label
    for colorrgb in group:
        colormap2label[i] = ((colorrgb[i][0]*2 + colorrgb[i][1])*1 + colorrgb[i][2])/510
        i+=1
    colormap2label = torch.from_numpy(colormap2label)
    colormap2label = colormap2label.long()
    return colormap2label

# 传入的是256*256 里面的值为0,1,2  分别计算每一类的dice
def Dice(inp, target, eps=1):
    # 抹平了，弄成一维的
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    # 计算交集中的数量
    r = len(input_flatten)
    overlap_num = 0
    for i in range(r):
        overlap_num += (input_flatten[i] == target_flatten[i])
    return overlap_num/(r*r)

def DivideBlockValidIndex(numBlock, i, j):
    a = i != 0
    b = j != 0
    c = i != numBlock[0] - 1
    d = j != numBlock[1] - 1
    return a, b, c, d

def DivideImg(numBlock,sizeBlock,widthOverlap):
    blockInfo = np.zeros((int(numBlock[0]), int(numBlock[1]), 14), dtype = np.float32)
    for i in range(0, int(numBlock[0])):
        for j in range(0, int(numBlock[1])):
            a, b,c,d= DivideBlockValidIndex(numBlock,i, j)
            tempSartInd = np.array([a * i * (sizeBlock[0] - widthOverlap[0]), b * j * (sizeBlock[1] - widthOverlap[1])])
            tempEndInd = tempSartInd + sizeBlock -1  # 第0个像素到第959个像素
            tempValidSartInd = np.array([a * widthOverlap[0]/2., b * widthOverlap[1]/2.]) # ???
            tempValidEndInd = sizeBlock - np.array([c * widthOverlap[0]/2., d * widthOverlap[1]/2.]) - 1
            m = sizeBlock - widthOverlap/2
            n = widthOverlap/2

            # tempValidSartIndWhole = tempSartInd +  tempValidSartInd
            temp_finalstartInd = np.array([a*(int(m[0])+int(m[0]-n[0])*(i-1)), b*(int(m[1])+int(m[1]-n[1])*(j-1))])
            temp_finalendInd = temp_finalstartInd + np.array([int(m[0])-a*c*n[0],int(m[1])-b*d*n[1]])
            blockInfo[i, j, :] = np.concatenate((np.array([i, j]), tempSartInd, tempEndInd, tempValidSartInd, tempValidEndInd, temp_finalstartInd, temp_finalendInd))
    return blockInfo

def get_patch(coordinate, img):
    r = 256
    patch = img[coordinate[0]:coordinate[0]+r, coordinate[1]:coordinate[1]+r]
    return patch
# 保存每次预测的结果，四维


def test():
    model = Unet(3, 3).to(device)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))  # 载入训练好的模型
    model.eval()
    with torch.no_grad():
        image_g_path = 'D:\\project\\testnew\\whole1'
        dirs = os.listdir(image_g_path)
        dice_total = 0
        for h in range(len(dirs)):
            file = dirs[h]
            img_path = 'D:\\project\\testnew\\whole1\\' + file + '\\image.png'
            mask_path = 'D:\\project\\testnew\\whole1\\' + file + '\\mask.png'
            prd_savepath = 'D:\\project\\testnew\\whole1\\' + file + '\\prd.png'
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            sizeImg = np.shape(img)[0:2]  # 长宽
            sizeImg = np.array(sizeImg, dtype=np.float32)
            widthOverlap = np.array((64,64), dtype=np.float32)
            sizeBlock = np.array((256,256), dtype=np.float32)
            size = (int(sizeBlock[0]), int(sizeBlock[0]))
            numBlock = np.ceil((sizeImg - sizeBlock) / (sizeBlock - widthOverlap) + 1)
            shapeTemp = (int(numBlock[0]), int(numBlock[1]))
            sizeImgNew = (numBlock - 1) * (sizeBlock - widthOverlap) + sizeBlock

            imgPad = np.pad(img, ((0, int(sizeImgNew[0] - sizeImg[0])), (0, int(sizeImgNew[1] - sizeImg[1])),(0,0)),'reflect')  # 对称填充
            maskpad = np.pad(mask, ((0, int(sizeImgNew[0] - sizeImg[0])), (0, int(sizeImgNew[1] - sizeImg[1])),(0,0)),'reflect')
            n = int(numBlock[0]*numBlock[1])
            imgT = np.zeros((n, int(sizeBlock[0]), int(sizeBlock[1]), 3), dtype=np.float32)
            whole = np.zeros((int(sizeImgNew[0]), int(sizeImgNew[1]), 3), dtype=np.uint8)
            blockInfo = DivideImg(numBlock,sizeBlock,widthOverlap)

            for k in range(n):
                i,j = np.unravel_index(k, shapeTemp)
                d1 = blockInfo[i, j, 2: 4]  # 2,3  tempstartInd
                # blockInfo (num[0],num[1],12)
                position = (int(d1[0]), int(d1[1]))
                crop = get_patch(position, imgPad)  # 得到小块的图
                mask_crop = get_patch(position,maskpad)
                mask_num_label_tensor = colormaptolabel(mask_crop)  # (256,256)
                img_tensor = torch.from_numpy(crop.transpose((2, 0, 1)))
                img_tensor = img_tensor.float().div(255)
                img_tensor = img_tensor.to(device)  # torch.Size([3, 256, 256])
                mask_num_label_tensor = mask_num_label_tensor.to(device)  # torch.Size([256, 256])
                img_tensor = img_tensor.unsqueeze(0)
                tar = mask_num_label_tensor.unsqueeze(0)  # 没有dataloader
                pred = model(img_tensor)
                _, predicted = torch.max(pred.data, dim=1)
                predicted = torch.squeeze(predicted).cpu().numpy()  # （256,256）
                tar = torch.squeeze(tar).cpu().numpy()

                # 得到的索引值和标签类别值一致（对应）吗
                dice = Dice(predicted, tar)
                dice_total += dice
                print('%d:dice=%f' % (k, dice))
                writer.add_scalar('dice', dice, global_step=k)

                label2rgb = np.zeros((256, 256, 3), dtype=np.uint8)
                for c in range(3):
                    label2rgb[:, :, 0] += ((predicted[:, :] == c) * (colormap[c][0])).astype('uint8')
                    label2rgb[:, :, 1] += ((predicted[:, :] == c) * (colormap[c][1])).astype('uint8')
                    label2rgb[:, :, 2] += ((predicted[:, :] == c) * (colormap[c][2])).astype('uint8')
                imgT[k, :, :, :] = label2rgb
            imgPredictList = [[0] * int(numBlock[1]) for i in range(int(numBlock[0]))]
            for k in range(n):
                i, j = np.unravel_index(k, shapeTemp)
                c1 = blockInfo[i, j, 6: 8]  # i,j不为0时，120 120 tempValidSartInd
                c2 = blockInfo[i, j, 8: 10]  # tempValidEndInd 重叠部分的中点 840,840
                imgProb = imgT[k, :, :, :]
                imgProbValid = imgProb[int(c1[0]): int(c2[0] + 1), int(c1[1]): int(c2[1] + 1), :]
                c3 = blockInfo[i,j,10:12]
                c4 = blockInfo[i,j,12:14]
                whole[int(c3[0]):int(c4[0]),int(c3[1]):int(c4[1])] = imgProbValid[:,:]

            whole = Image.fromarray(whole)
            Image.Image.save(whole, fp=prd_savepath)
        writer.close()
        print('Dice=%f' % (dice_total / n))






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