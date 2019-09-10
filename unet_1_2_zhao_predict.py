
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras.models import Sequential,Model,load_model
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input,Dropout
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import os
from tqdm import tqdm

from keras import backend as K
import tensorflow as tf
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from libtiff import TIFF

import read_data
import trans_label


filepath = 'D:\\academic\\competition\\segment\\data\\data_with_num\\'
modelpath= 'D:\\academic\\competition\\segment\\code\\semantic_segmentation\\model\\'                       ############################

img_w=256
img_h=256
image_size=256
# 有一个为背景
n_label=15+1

classes=[0., 179., 146., 164., 156., 92., 140., 148., 117., 73., 83., 161., 60., 23., 192., 180.]
labelencoder = LabelEncoder()  # 标签标准化
labelencoder.fit(classes)


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

def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model=load_model(modelpath+'unet_z_2.h5',custom_objects={'mean_iou': mean_iou})  # 载入模型                                   ########################################
    stride = 64
    pad_len = np.int8((256 - stride) / 2)
    TEST_SET = os.listdir(filepath + 'test\\image')  ########################################

    nextpath = 'test\\test_result15'  ########################################
    if not os.path.exists(filepath + nextpath):
        os.makedirs(filepath + nextpath)

    # print(TEST_SET)
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        image = read_data.read_images(filepath + 'test\\image\\' + path,
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
        rgb = trans_label.gray2rgb(mask_whole[0:h, 0:w])
        cv2.imwrite(filepath + nextpath + '\\rgb_' + str(n + 1) + '.bmp', rgb)  ########################
        trans_label.save_to_tiff(rgb, filepath + nextpath + '\\tif_' + str(n + 1) + '.tif')  ########################

if __name__ == '__main__':
    predict()