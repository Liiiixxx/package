import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import torch

def make_dataset(root):
    imgs = []
    n = len(os.listdir(root))//2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
    for i in range(int(n)):
        img = os.path.join(root, "%d.png" % i)
        mask = os.path.join(root, "%d_mask.png" % i)
        imgs.append((img, mask))
    return imgs

# 构建从rgb到类别的映射
def colormaptolabel(group):
    # print(len(group))
    colormap2label = np.zeros([len(group),len(group)])
    i=0
    colormap2label = colormap2label
    for colorrgb in group:
        colormap2label[i] = ((colorrgb[i][0]*2 + colorrgb[i][1])*1 + colorrgb[i][2])/510
        i+=1
    colormap2label = torch.from_numpy(colormap2label)
    colormap2label = colormap2label.long()
    return colormap2label


class LiverDataset(data.Dataset):
    def __init__(self, root1,root2,root3, transform=None, target_transform=None):
        imgs1 = make_dataset(root1)
        imgs2 = make_dataset(root2)
        imgs3 = make_dataset(root3)
        imgs = np.concatenate((imgs1, imgs2,imgs3))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        origin_x = Image.open(x_path)
        origin_x = np.array(origin_x)
        origin_y = Image.open(y_path)
        origin_y = np.array(origin_y)
        new_y = colormaptolabel(origin_y)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        # if self.target_transform is not None:
        #     img_y = self.target_transform(new_y)

        return img_x, new_y


    def __len__(self):
        return len(self.imgs)



class LiverDataset2(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        # imgs = np.concatenate((imgs1, imgs2,imgs3))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        origin_x = Image.open(x_path)
        origin_x = np.array(origin_x)
        origin_y = Image.open(y_path)
        origin_y = np.array(origin_y)
        new_y = colormaptolabel(origin_y)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        # if self.target_transform is not None:
        #     img_y = self.target_transform(new_y)

        return img_x, new_y

    def __len__(self):
        return len(self.imgs)


