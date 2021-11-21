import os, sys
import cv2
import PIL.Image as Image
import random
import numpy as np

def image_compose(IMAGE_COLUMN,IMAGE_ROW,IMAGE_SIZE,IMAGES_PATH,image_names,IMAGE_SAVE_PATH):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH+'\\' + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
            to_image.save(IMAGE_SAVE_PATH)


path = "D:\\project\\whole\\testmade"
dirs = os.listdir( path )
a =0
IMAGES_PATH = "D:\\project\\testnew\\try"
dirs2 = os.listdir(IMAGES_PATH)
dirs2.sort(key= lambda x:int(x[:-9]))
a = 0
for i in range(len(dirs)):
    file = dirs[i]
    mask_path = 'D:\\project\\whole\\testmade\\' + file +'\\mask.png'
    mask = cv2.imread(mask_path)
    img_x, img_y = mask.shape[0:2]

    r = 256
    count = 0
    x_num = int(img_x/r)
    y_num = int(img_y/r)
    pic_num = x_num*y_num
    image_names = []
    for img_num in range(a,a+pic_num):
        image_names.append(dirs2[img_num])
        count +=1
    a += count
    image_save_path = "D:\\project\\whole_img_" + str(i) + ".png"
    # image_compose(x_num, y_num, 256, IMAGES_PATH, image_names, image_save_path)
    to_image = Image.new('RGB', (y_num * r, x_num * r))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, x_num + 1):
        for x in range(1, y_num + 1):
            from_image = Image.open(IMAGES_PATH + '\\' + image_names[y_num * (y - 1) + x - 1]).resize(
                (r, r), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * r, (y - 1) * r))
            to_image.save(image_save_path)
    i += 1



