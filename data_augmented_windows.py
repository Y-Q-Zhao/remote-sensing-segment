#coding=utf-8
#数据增强
#在生成一个个batch
import cv2

import os
import numpy as np

#from libtiff import TIFF

img_w = 256  
img_h = 256  
filepath = 'E:\\data_segmentated\\'
train_RGB_path=filepath+'train\\RGB'
train_NIR_path=filepath+'train\\NIR'
train_LABEL_path=filepath+'train\\LABEL'
image_sets=os.listdir(train_RGB_path)
#image_sets = ['1.png','2.png','3.png','4.png','5.png']
# np.power(x1,n)求的是x1的n次方
#图像灰度的伽玛变换
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    
# 图像的旋转操作
def rotate(xb,nb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    nb = cv2.warpAffine(nb, M_rotate, (img_w, img_h))
    return xb,nb,yb
    #因为要对原图和标记图片做相同的处理


#进行均值滤波
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    num=np.random.randint(0,256)#随机生成点噪声的个数
    for i in range(num): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255 #255是白噪声
    return img
    
    
def data_augment(xb,nb,yb):
    if np.random.random() < 0.25:
        xb,nb,yb = rotate(xb,nb,yb,90)
    if np.random.random() < 0.25:
        xb,nb,yb = rotate(xb,nb,yb,180)
    if np.random.random() < 0.25:
        xb,nb,yb = rotate(xb,nb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        nb = cv2.flip(nb, 1)
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        nb = random_gamma_transform(nb,1.0)
        
        
    if np.random.random() < 0.25:
        xb = blur(xb)
        nb = blur(nb)
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        nb = add_noise(nb)
        
        
    return xb,nb,yb

#------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
#import matplotlib.pyplot as plt
#import argparse
#import numpy as np
#from keras.models import Sequential
#from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
#from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
#from PIL import Image
#import matplotlib.pyplot as plt
#import cv2
#import random
import os
n_label=15+1
IMG_CHANNELS=4
classes=[0., 179., 146., 164., 156., 92., 140., 148., 117., 73., 83., 161., 60., 23., 192., 180.]
labelencoder = LabelEncoder()  # 标签标准化
labelencoder.fit(classes)

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读入灰度图片
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0  # 归一化
    return img
#-------------------------------------------
# data for training
def generateTrainData(batch_size,data=[]):
    while True:
        train_data=[]
        train_label=[]
        batch=0
        for i in (range(len(data))):
            url=data[i]
            batch+=1

            #读取RGB图片和NIR图片，并合并
            tif_img = np.zeros((img_w, img_h, IMG_CHANNELS))
            img_rgb=load_img(filepath+'train\\RGB\\'+url)                                                              #####修改路径
            img_nir=load_img(filepath+'train\\NIR\\'+url)
            label=load_img(filepath+'train\\LABEL\\'+url,grayscale=True)      #####修改 
            img_rgb,img_nir,label=data_augment(img_rgb,img_nir,label)  
            label=img_to_array(label).reshape((img_w*img_h,))                                        #####修改路径
            tif_img[:,:,0:3]=img_rgb
            tif_img[:, :, 3]=img_nir[:,:,0]
            img=tif_img
            img=img_to_array(img)
            tif_img=np.zeros((img_w,img_h,IMG_CHANNELS))
            img=img_to_array(img)            #将图片转化成数组
            train_data.append(img)

            
            
            # print label.shape
            train_label.append(label)
            if batch % batch_size==0:
                #print 'get enough bacth!\n'
                train_data=np.array(train_data)       #batch_size*img_w*img_h*3
                #标签转化成one-hot
                train_label=np.array(train_label).flatten()       #降成一维向量
                train_label=labelencoder.transform(train_label)   #转换成连续的数值型变量
                train_label=to_categorical(train_label, num_classes=n_label)          #将类别向量转换为二进制（只有0和1）的矩阵类型表示
                train_label=train_label.reshape((batch_size,img_w, img_h,n_label))

                yield (train_data,train_label)  #输出数据和标签
                train_data = []
                train_label = []
                batch = 0


def generateValidData(batch_size, data=[]):
    while True:
        valid_data =[]
        valid_label=[]
        batch=0
        for i in (range(len(data))):
            url=data[i]
            batch+=1

            # 读取RGB图片和NIR图片，并合并
            tif_img=np.zeros((img_w, img_h, IMG_CHANNELS))
            img_rgb=load_img(filepath+'validation\\RGB\\'+url)                                                         #####修改
            img_nir=load_img(filepath+'validation\\NIR\\'+url)
            label=load_img(filepath+'validation\\LABEL\\'+url, grayscale=True)
            img_rgb,img_nir,label=data_augment(img_rgb,img_nir,label)  
            label=img_to_array(label).reshape((img_w*img_h,))                                        #####修改路径
                             #####修改
            label=img_to_array(label).reshape((img_w*img_h,))
                                                                    #####修改
            tif_img[:,:,0:3]=img_rgb
            tif_img[:, :, 3]=img_nir[:,:,0]
            img=tif_img
            img=img_to_array(img)
            tif_img=np.zeros((img_w,img_h,IMG_CHANNELS))
            img=img_to_array(img)  # 将图片转化成数组
            valid_data.append(img)

            # print label.shape
            valid_label.append(label)
            if batch%batch_size==0:
                # print 'get enough bacth!\n'
                valid_data=np.array(valid_data)  # batch_size*img_w*img_h*3
                # 标签转化成one-hot
                valid_label=np.array(valid_label).flatten()  # 降成一维向量
                valid_label=labelencoder.transform(valid_label)  # 转换成连续的数值型变量
                valid_label=to_categorical(valid_label,num_classes=n_label)  # 将类别向量转换为二进制（只有0和1）的矩阵类型表示
                valid_label=valid_label.reshape((batch_size,img_w,img_h,n_label))

                yield (valid_data,valid_label)  # 输出数据和标签
                valid_data=[]
                valid_label=[]
                batch=0


