import os, sys
import cv2
import random
import numpy as np
# 打开文件
path = "D:\\project\\whole\\trainmade"
dirs = os.listdir( path )
for i in range(len(dirs)):
    file = dirs[i]
    img_path =  'D:\\project\\whole\\trainmade\\' + file +'\\image.png'
    mask_path = 'D:\\project\\whole\\trainmade\\' + file +'\\mask.png'
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    img_x, img_y = img.shape[0:2]
    r = 256
    count = 0
    num = 1500
    while 1:
        y = random.randint(r, img_y - r)
        x = random.randint(r, img_x - r)
        if mask[x,y][2]==255:
            f_mask_patch = mask[(x):(x + r), (y):(y + r)]
            f_img_patch = img[(x):(x + r), (y):(y + r)]
            cv2.imwrite('D:\\project\\trainnew\\f\\' + str(count + i * num) + '.png', f_img_patch)
            cv2.imwrite('D:\\project\\trainnew\\f\\' + str(count + i * num) + '_mask.png', f_mask_patch)
            count+=1
        if count==num:
            break

    count = 0
    num = 1500
    while 1:
        y = random.randint(r, img_y - r)
        x = random.randint(r, img_x - r)
        if mask[x,y][1]==255:
            t_mask_patch = mask[(x):(x + r), (y):(y + r)]
            t_img_patch = img[(x):(x + r), (y):(y + r)]
            cv2.imwrite('D:\\project\\trainnew\\t\\' + str(count + i * num) + '.png', t_img_patch)
            cv2.imwrite('D:\\project\\trainnew\\t\\' + str(count + i * num) + '_mask.png', t_mask_patch)
            count+=1
        if count==num:
            break


    count = 0
    num = 1500
    while 1:
        y = random.randint(r, img_y - r)
        x = random.randint(r, img_x - r)
        m = np.logical_and(mask[x,y][1]==0,mask[x,y][2]==0)
        if m:
            b_mask_patch = mask[(x):(x + r), (y):(y + r)]
            b_img_patch = img[(x):(x + r), (y):(y + r)]
            cv2.imwrite('D:\\project\\trainnew\\b\\' + str(count + i * num) + '.png', b_img_patch)
            cv2.imwrite('D:\\project\\trainnew\\b\\' + str(count + i * num) + '_mask.png', b_mask_patch)
            count+=1
        if count==num:
            break

