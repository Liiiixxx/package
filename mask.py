import numpy as np
import cv2
import openslide
import os

def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def Getmask(file1_path,file2_path,sli_path,level,img_save):
    slide = openslide.open_slide(sli_path)
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(img_save,img)
    csv_lines1 = open(file1_path, "r").readlines()
    csv_lines2 = open(file2_path, "r").readlines()
    b = [x[0] for x in csv_lines2]
    a = []
    csv_lines1new = []
    for i in range(len(b)):
        if (b[i] == 'f'):
            a.append(b[i])
            csv_lines1new.append(csv_lines1[i])
    for i in range(len(b)):
        if (b[i] == 'T'):
            a.append(b[i])
            csv_lines1new.append(csv_lines1[i])
    for i in range(len(b)):
        if (b[i] == 'S'):
            a.append(b[i])
            csv_lines1new.append(csv_lines1[i])
    mask = np.zeros(img.shape, dtype=np.uint8)
    j = 0
    for line in csv_lines1new:
        if (a[j] == 'f'):
            color = (0, 0, 255)
        if (a[j] == 'T'):
            color = (0, 255, 0)
        if (a[j] == 'S'):
            color = (0, 0, 255)
        j += 1
        contour = []
        # con = []
        s = line.strip()  ##去除末尾的/n符号
        ind = s.find('[')  ##返回字符在母串中的下标位置, [ 在数据的每行开头
        # strROI = s[:ind - 1].split(',')
        strContour = s[ind + 1:-1].split('Point:')[1:]
        for elem in strContour:
            elem = elem.split(',')
            x = int(np.round(float(elem[0]))/4)# level3 为什么是32正常
            y = int(np.round(float(elem[1]))/4)
            contour.append((x, y))  # 就可以转变成数组了

        pl = [np.stack(contour)]
        cv2.fillPoly(mask, pl, color)
    # cv_show('mask',mask)
    return mask
level = 1
path = "D:\\project\\whole"
dirs = os.listdir( path )
for i in dirs:
    file1_path = 'D:\\project\\whole\\'+i+'\\file1.csv'
    file2_path = 'D:\\project\\whole\\'+i +'\\file2.csv'
    sli_path = 'D:\\project\\IHC\\QuPath--20210823\\data2\\'+i+'.svs'
    img_savepath = 'D:\\project\\whole\\'+i +'\\image.png'
    mask = Getmask(file1_path, file2_path, sli_path, level, img_savepath)
    mask_savepath = 'D:\\project\\whole\\'+i+ '\\mask.png'
    cv2.imwrite(mask_savepath,mask)

#
# file1_path = 'D:\\project\\whole\\6826_2011555_ki67_level0\\file1.csv'
# file2_path = 'D:\\project\\whole\\6826_2011555_ki67_level0\\file2.csv'
# sli_path = 'D:\\project\\IHC\\QuPath--20210823\\data2\\6826_2011555_ki67.svs'
# img_savepath = 'D:\\project\\whole\\6826_2011555_ki67_level0\\image.png'
#
# mask = Getmask(file1_path,file2_path,sli_path,level,img_savepath)
# # cv_show('m',mask)
# cv2.imwrite('D:\\project\\whole\\6826_2011555_ki67_level0\\mask.png',mask)
#
#
