# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:52:01 2019

@author: skyliuhc
"""

import cv2
#import random
import numpy as np
import os
#import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
from libtiff import TIFF
from keras import backend as K
import tensorflow as tf


image_size = 256
n_label=15+1
classes=[0., 179., 146., 164., 156., 92., 140., 148., 117., 73., 83., 161., 60., 23., 192., 180.] 
#-----------------------------------------------------
modelpath='E:\\data_segmentated\\model\\unet.h5'
filepath='E:\\test\\'   #存放test文件的路径
#-----------------------------------------------------
labelencoder = LabelEncoder()  
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

    if single_channel:
        return channel1,channel2,channel3,channel4
    else:
        return image
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, n_label)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

    
def predict(modelpath):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    
    model = load_model(modelpath,custom_objects={'mean_iou': mean_iou})
  # 载入模型                      #####修改路径
    stride=256#拼接的步长，改小可以减少拼接的痕迹
    TEST_SET=os.listdir(filepath)                                      #####修改路径
    print(TEST_SET)
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        # load the image
        # image = cv2.imread('./test/' + path)  # 读取图片
        image=read_images(filepath+path,single_channel=False)#####修改路径
        h, w, _ = image.shape
        padding_h = (h // stride + 1) * stride
        padding_w = (w // stride + 1) * stride
        padding_img = np.zeros((padding_h, padding_w, 4), dtype=np.uint8)
        padding_img[0:h, 0:w, :] = image[:, :, :]  # padding图像原位置有图像的地方用原图像填充，没有的用0填充
        padding_img = padding_img.astype("float") / 255.0  # 归一化
        padding_img = img_to_array(padding_img)  # 转换成数组（可能会使矩阵发生转置）
        print('src:', padding_img.shape)
        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)  # 标签
        for i in range(padding_h // stride):
            for j in range(padding_w // stride):
                # crop = padding_img[:, i * stride:i * stride + image_size, j * stride:j * stride + image_size]
                crop = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size,:]
                # _, ch, cw = crop.shape
                ch, cw, _ = crop.shape
                if ch != 256 or cw != 256:
                    print('invalid size!')
                    continue
                print('crop.size before',crop.shape)
                crop = np.expand_dims(crop, axis=0)
                print('crop.size after',crop.shape)
                # print 'crop:',crop.shape
                pred = model.predict(crop)  # 预测的是类别，打印出来的值就是类别号
                pred=np.squeeze(pred)
                print(pred.shape)
                pred=pred.reshape(256*256,16)
                print(pred.shape)
                pred = np.argmax(pred,axis=1)
#                print(pred)
                print('类别',pred.shape)
#                pred=pred.reshape(256,256)
                print(pred)
                pred = labelencoder.inverse_transform(pred)
                # print (np.unique(pred))
                pred = pred.reshape((256, 256)).astype(np.uint8)
                # print 'pred:',pred.shape
                mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred[:, :]

        test_result_path=filepath+'\\test_result\\';#创建存放测试结果的路劲
        if not os.path.exists(test_result_path ):
            os.mkdir(test_result_path )
            
        cv2.imwrite(test_result_path+str(n + 1) + '.png', mask_whole[0:h, 0:w])#####修改路径
        
    

    
if __name__ == '__main__':
    
    predict(modelpath)



