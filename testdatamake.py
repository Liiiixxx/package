import os, sys
import cv2
import random
import numpy as np
# 打开文件
path = "D:\\project\\whole\\testmade"
dirs = os.listdir( path )
a =0
for i in range(len(dirs)):
    file = dirs[i]
    img_path =  'D:\\project\\whole\\testmade\\' + file +'\\image.png'
    mask_path = 'D:\\project\\whole\\testmade\\' + file +'\\mask.png'
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    img_x, img_y = img.shape[0:2]
    r = 256
    count = 0
    x_num = int(img_x/r)
    y_num = int(img_y/r)
    for m in range(0,x_num):
        for j in range(0,y_num):
            f_mask_patch = mask[(m*r):(m*r + r), (j*r):(j*r + r)]
            f_img_patch = img[(i*r):(i*r + r), (j*r):(j*r + r)]
            cv2.imwrite('D:\\project\\testnew\\try\\' + str(count + a) + '.png', f_img_patch)
            cv2.imwrite('D:\\project\\testnew\\try\\' + str(count + a) + '_mask.png', f_mask_patch)
            count+=1

    a += count