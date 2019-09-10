# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:33:27 2019

@author: Administrator
"""


import matplotlib
matplotlib.use("Agg")
#import matplotlib.pyplot as plt

import numpy as np
from keras.models import load_model
#from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input,Dropout
#from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
#from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.preprocessing import LabelEncoder
#from PIL import Image
#import matplotlib.pyplot as plt
import cv2
#import random
import os
#from tqdm import tqdm

#from keras import backend as K
#import tensorflow as tf
#from keras.layers.core import Dropout, Lambda
#from keras.layers.merge import concatenate
from libtiff import TIFF





filepath = 'E:\\test\\'
modelpath= 'E:\\data_segmentated\\model\\'                      ############################

img_w=256
img_h=256
image_size=256
# 有一个为背景
n_label=15+1

classes=[0., 179., 146., 164., 156., 92., 140., 148., 117., 73., 83., 161., 60., 23., 192., 180.]
labelencoder = LabelEncoder()  # 标签标准化
labelencoder.fit(classes)



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
#由于网络的输入是是RGB,NIR,所以得调整次序
    image[:,:,0]=channel2
    image[:,:,1]=channel3
    image[:,:,2]=channel4
    image[:,:,3]=channel1
    if single_channel:
        return channel1,channel2,channel3,channel4
    else:
        return image
def save_to_tiff(image,filepath):
    tif=TIFF.open(filepath,mode='w')
    tif.write_image(image[:,:,0:3],compression=None,write_rgb=True)
    tif.close()
def gray2rgb(lab_gray):         #lab_gray必须为单通道
    '''transform data from gray to rgb'''
    dict={117:[  0,200,  0],        #水田
          164:[150,250,  0],        #水浇地
          179:[150,200,150],        #旱耕地
           83:[200,  0,200],        #园地
           92:[150,  0,250],        #乔木林地
          180:[150,150,250],        #灌木林地
          146:[250,200,  0],        #天然草地
          140:[200,200,  0],        #人工草地
           23:[200,  0,  0],        #工业用地
           73:[250,  0,150],        #城市住宅
          156:[200,150,150],        #城镇住宅
          161:[250,150,150],        #交通运输
           60:[  0,  0,200],        #河流
          148:[  0,150,200],        #湖泊
          192:[  0,200,250],        #坑塘
            0:[  0,  0,  0]}        #其他类别
    lab_h,lab_w=lab_gray.shape
    lab_rgb=np.zeros((lab_h,lab_w,3))
    for i in range(lab_h):
        for j in range(lab_w):
            lab_rgb[i,j,:]=np.array(dict[lab_gray[i,j]])

    return np.uint8(lab_rgb)
def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model=load_model(modelpath+'unet_1_3.h5')  # 载入模型                                   ########################################
    stride = 64
    pad_len = np.int8((256 - stride) / 2)
    TEST_SET = os.listdir(filepath + 'test_image\\')  ########################################

    nextpath = 'test_result'  ########################################
    if not os.path.exists(filepath + nextpath):
        os.makedirs(filepath + nextpath)

    # print(TEST_SET)
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        image = read_images(filepath + 'test_image\\' + path,
                                      single_channel=False)  #######################################
        h, w, _ = image.shape
        # padding_h = (h // stride + 2) * stride
        # padding_w = (w // stride + 2) * stride
        padding_h = (h // stride + 1) * stride + pad_len * 2
        padding_w = (w // stride + 1) * stride + pad_len * 2

        padding_img = np.zeros((padding_h, padding_w, 4), dtype=np.uint8)
        padding_img[pad_len:h + pad_len, pad_len:w + pad_len, :] = image[:, :, :]  # padding图像原位置有图像的地方用原图像填充，没有的用0填充
        # padding_img = padding_img.astype("float") / 255.0  # 归一化
        padding_img = img_to_array(padding_img)  # 转换成数组（可能会使矩阵发生转置）
        print('src:', padding_img.shape)
        mask_whole = np.zeros((padding_h - pad_len * 2, padding_w - pad_len * 2), dtype=np.uint8)  # 标签
        for i in range(h // stride + 1):
            for j in range(w // stride + 1):
                crop_tif = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size, :]
                # print(crop_tif.shape,i,j)
                crop = np.zeros((image_size, image_size, 4))
                crop[:, :, 0:3] = crop_tif[:, :, 1:4]
                crop[:, :, 3] = crop_tif[:, :, 0]
                # _, ch, cw = crop.shape
                ch, cw, _ = crop.shape
                if ch != 256 or cw != 256:
                    print('invalid size!,crop size:', ch, cw, i, j)
                    continue

                crop = np.expand_dims(crop, axis=0)
                pred = model.predict(crop)
                pred = np.squeeze(pred)
                pred = np.argmax(pred, axis=1)
                pred = labelencoder.inverse_transform(pred)
                pred = pred.reshape((256, 256)).astype(np.uint8)
                mask_whole[i * stride:i * stride + stride, j * stride:j * stride + stride] = pred[pad_len:pad_len + stride,pad_len:pad_len + stride]

        cv2.imwrite(filepath + nextpath + '\\gray_' + str(n + 1) + '.bmp',
                    mask_whole[0:h, 0:w])  ########################
        rgb = gray2rgb(mask_whole[0:h, 0:w])
        cv2.imwrite(filepath + nextpath + '\\rgb_' + str(n + 1) + '.bmp', rgb)  ########################
        save_to_tiff(rgb, filepath + nextpath + '\\tif_' + str(n + 1) + '.tif')  ########################

if __name__ == '__main__':
    predict()