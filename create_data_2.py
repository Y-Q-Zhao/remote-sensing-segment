#2019/07/29
'''
生成数据，针对样本少的类别进行扩充
'''

from libtiff import TIFF
import numpy as np
import cv2
import random
from tqdm import tqdm
import os
import random

#-----------------------------------------------------------------------------------------------------------------------
im_w=256                #小图大小
im_h=256
X_width=7200            #大图尺寸
X_height=6800
img_each=2000           #每张大图产生的小图数
aug_rate=0.5            #进行数据增强的概率
colormap = [0, 179, 146, 164, 156, 92, 140, 148, 117, 73, 83, 161, 60, 23, 192, 180]
namespace = ['GF2_PMS1__20150212_L1A0000647768-MSS1',
                 'GF2_PMS1__20150902_L1A0001015649-MSS1',
                 'GF2_PMS1__20151203_L1A0001217916-MSS1',
                 'GF2_PMS1__20160327_L1A0001491417-MSS1',
                 'GF2_PMS1__20160816_L1A0001765570-MSS1',
                 'GF2_PMS1__20160827_L1A0001793003-MSS1',
                 'GF2_PMS2__20160225_L1A0001433318-MSS2',
                 'GF2_PMS2__20160510_L1A0001573999-MSS2']

lab_bmp_path = 'E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
train_path = 'E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
seg_result_path = 'E:\\competition\\segment\\data\\special_seg\\'
#-----------------------------------------------------------------------------------------------------------------------
def read_images(add):
    tif=TIFF.open(add,mode='r')
    image=tif.read_image()
    return image
def save_to_bmp(image,filename):
    '''
    将图像一bmp格式保存到目标文件夹中
    '''
    ad=filename + '.bmp'
    image=image.astype(np.uint8)
    cv2.imwrite(ad,image)
#-----------------------------------------------------------------------------------------------------------------------
def ordinary_gen():
    '''
    顺序分割和随机随机分割
    '''
    for order in range(16):
        name = namespace[order]

        add_img_tif = train_path + name + '.tif'
        add_lab_tif = train_path + name + '_label.tif'
        add_lab_bmp = lab_bmp_path + name + '_label.bmp'

        image = read_images(add_img_tif)
        label = read_images(add_lab_tif)
        print(label.shape)
        lab_gray = cv2.imread(add_lab_bmp, cv2.IMREAD_GRAYSCALE)
        print(lab_gray.shape)

        # ##顺序分割部分
        index = 0
        account = 0
        for i in range(26):
            for j in range(28):
                save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],
                            seg_result_path + 'NIR\\' + 'tif' + str(order) + '_ori_' + str(index))
                save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],
                            seg_result_path + 'RGB\\' + 'tif' + str(order) + '_ori_' + str(index))
                save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],
                            seg_result_path + 'LABEL\\' + 'tif' + str(order) + '_ori_' + str(index))
                index += 1
                account += 1

        ##随机分割部分
        index = 0
        while account < img_each:
            random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
            random_height = random.randint(0, X_height - im_h - 1)
            ran_image = image[random_height: random_height + im_h, random_width: random_width + im_w, :]
            ran_lab = label[random_height: random_height + im_h, random_width: random_width + im_w]
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif'+str(order)+'_ran_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' +  'tif'+str(order)+'_ran_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' +  'tif'+str(order)+'_ran_'  + str(index))
            index += 1
            account += 1

def special_gen():
    axis = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for order in tqdm(range(8)):
        name=namespace[order]

        img_gray=cv2.imread(lab_bmp_path+name+'_label.bmp',cv2.IMREAD_GRAYSCALE)

        step=10
        for i in range((X_height-128)//step):
            for j in range((X_width-128)//step):
                for k in range(16):
                    if img_gray[i*10,j*10]==colormap[k] and i*10-128>=0 and i*10+128<=6800 and j*10-128>=0 and j*10+128<7200:
                        axis[k].append((order,i,j))

    index=0
    for k in [5,6,10,11,12,13,14,15]:
        list=axis[k]
        random.shuffle(list)
        for i in range(1000):
            name=namespace[list[i][0]]
            h=list[i][1]
            w=list[i][2]

            add_img_tif = train_path + name + '.tif'
            add_lab_tif = train_path + name + '_label.tif'

            image = read_images(add_img_tif)
            label = read_images(add_lab_tif)

            aug_img=image[h*10-128:h*10+128,w*10-128:w*10+128,:]
            aug_lab=label[h*10-128:h*10+128,w*10-128:w*10+128,:]

            save_to_bmp(aug_img[:, :, 0], seg_result_path +   'NIR\\' + 'kind' + str(k) + '_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'kind' + str(k) + '_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path          + 'LABEL\\' + 'kind' + str(k) + '_spe_' + str(index))

            index+=1

if __name__ == '__main__':
    special_gen()
