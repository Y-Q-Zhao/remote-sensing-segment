#2019/07/18
#读取tiff图像数据,并将图片分割，保存成bmp格式
#分割训练集和验证集的图片，并进行图像增强，获得20000张数据
#zhaoyiqun

import numpy as np
from libtiff import TIFF
import cv2
import os
import random

im_w=256                #小图大小
im_h=256
X_width=7200            #大图尺寸
X_height=6800
img_each=2000           #每张大图产生的小图数
aug_rate=0.5            #进行数据增强的概率


add = 'D:\\academic\\competition\\segment\\data\\data_with_num\\whole_dataset2\\'
#分割后图像存储文件地址，该文件夹下需要建立3个文件夹：train(RGB,NIR,LABEL),validation(RGB,NIR,LABEL),test
file_train='D:\\academic\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train'
#训练数据集存储地址
file_val  ='D:\\academic\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\val\\val'
#验证数据集存储地址

if not os.path.exists(add):
    os.makedirs(add)
    os.makedirs(add+'NIR')
    os.makedirs(add+'RGB')
    os.makedirs(add+'LABEL')


def read_images(add_tif,single_channel=True):
    '''
    读取tiff图像，tiff图像为4维，分别为NIR,R,G,B,尺寸为(6800,7200,4)
    '''
    tif=TIFF.open(add_tif,mode='r')
    image=tif.read_image()
    print(image.shape)
    channel1=image[:,:,0]#NIR
    channel2=image[:,:,1]#R
    channel3=image[:,:,2]#G
    channel4=image[:,:,3]#B

    if single_channel:
        return channel1,channel2,channel3,channel4
    else:
        return image

def read_labels(add_lab,single_channel=True):
    '''
    读取label图像，label图像为3维，分别为R,G,B,尺寸为(6800,7200,3)
    '''
    tif=TIFF.open(add_lab,mode='r')
    label=tif.read_image()
    print(label.shape)
    channel1=label[:,:,0]#R
    channel2=label[:,:,1]#G
    channel3=label[:,:,2]#B

    if single_channel:
        return channel1,channel2,channel3
    else:
        return label

def save_to_tiff(image,filename,data_kind='train'):
    add='/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/segmented_image/'+data_kind+'/'+filename+'.tif'
    tif=TIFF.open(add,mode='w')
    tif.write_image(image[:,:,0:3],compression=None)
    tif.write_image(image[:,:,2],compression=None,write_rgb=True)
    tif.write_image(image[:,:,3],compression=None,write_rgb=True)
    tif.close()

def save_to_bmp(image,filename):
    '''
    将图像一bmp格式保存到目标文件夹中
    '''
    ad=add+filename + '.bmp'
    image=image.astype(np.uint8)
    cv2.imwrite(ad,image)

########################################################################################################################
def gamma_transform(img, gamma):
    '''
    伽马校正
    '''
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    '''
    随机伽马校正
    '''
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    '''
    旋转
    '''
    M_rotate = cv2.getRotationMatrix2D((im_w / 2, im_h / 2), angle, 1)  # 获得仿射变换矩阵
    xb = cv2.warpAffine(xb, M_rotate, (im_w, im_h))  # 进行仿射变化
    yb = cv2.warpAffine(yb, M_rotate, (im_w, im_h))
    return xb, yb


def blur(img):
    '''
    均值滤波
    '''
    img = cv2.blur(img, (3, 3));
    return img


def add_noise(img):
    '''
    加噪声
    '''
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb
########################################################################################################################


def segmentation_image():
    '''
    分割遥感图像
    '''
    dir_list_train=os.listdir(file_train)
    dir_list_val = os.listdir(file_val  )

    index=0
    for filename in dir_list_train:
        if len(filename.split('_'))==5:
            name=filename.split('.')[0]
            add=file_train+'\\'+filename
            add_lab=file_train+'\\'+name+'_label.tif'
            # print('the tiff image address:',add)
            image=read_images(add,single_channel=False)
            label=read_labels(add_lab,single_channel=False)

            account=0
            for i in range(26):
                for j in range(28):
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],
                                'NIR\\' + str(index))
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],
                                'RGB\\' + str(index))
                    save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],
                                'LABEL\\' + str(index))
                    index+=1
                    account+=1

            while account<img_each:
                random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
                random_height = random.randint(0, X_height - im_h - 1)
                ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
                ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
                if np.random.random() < aug_rate:
                    aug_img, aug_lab=data_augment(ran_image,ran_lab)
                else:
                    aug_img=ran_image
                    aug_lab=ran_lab
                save_to_bmp(aug_img[:,:,0],'NIR\\' + str(index))
                save_to_bmp(aug_img[:,:,1:4],'RGB\\' + str(index))
                save_to_bmp(aug_lab,'LABEL\\' + str(index))
                index+=1
                account+=1

            print(filename+'    is done !')

    for filename in dir_list_val:
        if len(filename.split('_'))== 5:
            name=filename.split('.')[0]
            add=file_val+'\\'+filename
            add_lab=file_val+'\\'+name+'_label.tif'
            image=read_images(add,single_channel=False)
            label=read_labels(add_lab,single_channel=False)


            account=0
            for i in range(26):
                for j in range(28):
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],
                                'NIR\\' + str(index))
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],
                                'RGB\\' + str(index))
                    save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],
                                'LABEL\\'+ str(index))
                    index+=1
                    account+=1


            while account<img_each:
                random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
                random_height = random.randint(0, X_height - im_h - 1)
                ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
                ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
                if np.random.random() < aug_rate:
                    aug_img, aug_lab = data_augment(ran_image, ran_lab)
                else:
                    aug_img = ran_image
                    aug_lab = ran_lab
                save_to_bmp(aug_img[:,:,0],'NIR\\' + str(index))
                save_to_bmp(aug_img[:,:,1:4],'RGB\\' + str(index))
                save_to_bmp(aug_lab,'LABEL\\' + str(index))
                index+=1
                account+=1


            print(filename + '    is done !')

    print('total image:',index)


if __name__ == '__main__':
    # read_images('D:\\code\\GF2_PMS1__20150212_L1A0000647768-MSS1.tif')
    # read_all_data()
    segmentation_image()
