# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:26:46 2019

@author: Administrator
"""

#2019/7/20
#刘杭成

# 数据增强

# 在生成一个个batch

import cv2

import os

import numpy as np

import random

import warnings

import matplotlib.pyplot as plt

from keras.models import Model

from keras.layers import Input,Reshape

from keras.layers.core import Dropout

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

import tensorflow as tf



# ------------------------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelEncoder

n_label = 15 + 1

IMG_CHANNELS = 4

classes = [0., 179., 146., 164., 156., 92., 140., 148., 117., 73., 83., 161., 60., 23., 192., 180.]

labelencoder = LabelEncoder()  # 标签标准化

labelencoder.fit(classes)

def rand_split_data(datalist):
#    print(datalist)
    np.random.shuffle(datalist)
#    print(datalist)
    train_list=[]
    valid_list=[]
    for i in range(18000):
        train_list.append(datalist[i])
    for j in range(2000):
        valid_list.append(datalist[18000+j])
    return train_list,valid_list

def load_img(path, grayscale=False):
    if grayscale:

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读入灰度图片

    else:

        img = cv2.imread(path)

        # img = np.array(img, dtype="float") / 255.0  # 归一化

    return img


# -------------------------------------------

# data for training

def generateTrainData(batch_size, data=[]):
    while True:

        train_data = []

        train_label = []

        batch = 0

        for i in (range(len(data))):

            url = data[i]

            batch += 1

            # 读取RGB图片和NIR图片，并合并

            tif_img = np.zeros((img_w, img_h, IMG_CHANNELS))

            img_rgb = load_img(filepath + '\\RGB\\' + url)  #####修改路径

            img_nir = load_img(filepath + '\\NIR\\' + url)

            label = load_img(filepath + '\\LABEL\\' + url, grayscale=True)  #####修改

            # img_rgb, img_nir, label = data_augment(img_rgb, img_nir, label)

            label = img_to_array(label).reshape((img_w * img_h,))  #####修改路径

            tif_img[:, :, 0:3] = img_rgb

            tif_img[:, :, 3] = img_nir[:, :, 0]

            img = tif_img

            img = img_to_array(img)

            # tif_img = np.zeros((img_w, img_h, IMG_CHANNELS))

            img = img_to_array(img)  # 将图片转化成数组

            train_data.append(img)      # (n*(img_w,img_h,4))

            train_label.append(label)   # (n*(img_w*img_h))

            if batch % batch_size == 0:
                # print 'get enough bacth!\n'

                train_data = np.array(train_data)  # (batch_size,img_w,img_h,4)

                # 标签转化成one-hot

                train_label = np.array(train_label).flatten()  # 降成一维向量 (n*img_w*img_h)

                train_label = labelencoder.transform(train_label)  # 转换成连续的数值型变量    (n*img_w*img_h)

                train_label = to_categorical(train_label, num_classes=n_label)  # 将类别向量转换为二进制（只有0和1）的矩阵类型表示 (n*img_w*img_h,n_label)

                # train_label = train_label.reshape((batch_size, img_w, img_h, n_label))
                train_label = train_label.reshape((batch_size, img_w*img_h, n_label))

                yield (train_data, train_label)  # 输出数据和标签

                train_data = []

                train_label = []

                batch = 0


def generateValidData(batch_size, data=[]):
    while True:

        valid_data = []

        valid_label = []

        batch = 0

        for i in (range(len(data))):

            url = data[i]

            batch += 1

            # 读取RGB图片和NIR图片，并合并

            tif_img = np.zeros((img_w, img_h, IMG_CHANNELS))

            img_rgb = load_img(filepath + '\\RGB\\' + url)  #####修改

            img_nir = load_img(filepath + '\\NIR\\' + url)

            label = load_img(filepath + '\\LABEL\\' + url, grayscale=True)

            # img_rgb, img_nir, label = data_augment(img_rgb, img_nir, label)

            label = img_to_array(label).reshape((img_w * img_h,))  #####修改路径

            #####修改

            #####修改

            tif_img[:, :, 0:3] = img_rgb

            tif_img[:, :, 3] = img_nir[:, :, 0]

            img = tif_img

            img = img_to_array(img)

            # tif_img = np.zeros((img_w, img_h, IMG_CHANNELS))

            img = img_to_array(img)  # 将图片转化成数组

            valid_data.append(img)

            # print label.shape

            valid_label.append(label)

            if batch % batch_size == 0:
                # print 'get enough bacth!\n'

                valid_data = np.array(valid_data)  # batch_size*img_w*img_h*3

                # 标签转化成one-hot

                valid_label = np.array(valid_label).flatten()  # 降成一维向量

                valid_label = labelencoder.transform(valid_label)  # 转换成连续的数值型变量

                valid_label = to_categorical(valid_label, num_classes=n_label)  # 将类别向量转换为二进制（只有0和1）的矩阵类型表示

                # valid_label = valid_label.reshape((batch_size, img_w, img_h, n_label))
                valid_label = valid_label.reshape((batch_size, img_w*img_h, n_label))

                yield (valid_data, valid_label)  # 输出数据和标签

                valid_data = []

                valid_label = []

                batch = 0


# --------------------------------------------------------------


# Set some parameters

img_w = 256

img_h = 256

IMG_CHANNELS = 4

filepath = 'E:\\newdataset'

# TRAIN_PATH=next(os.walk(filepath+'train' ))                                             #####修改数据集路径

modelpath = 'E:\\data_segmentated\\model\\'  #####修改模型路径

resultpath = 'E:\\data_segmentated\\result\\'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42

random.seed = seed

np.random.seed = seed

# ------------------------------------------------------------------------------------

import matplotlib

from keras import optimizers
matplotlib.use("Agg")

# import matplotlib.pyplot as plt

# import argparse

import numpy as np

from sklearn.preprocessing import LabelEncoder

n_label = 15 + 1

classes = [0., 179., 146., 164., 156., 92., 140., 148., 117., 73., 83., 161., 60., 23., 192., 180.]

labelencoder = LabelEncoder()  # 标签标准化

labelencoder.fit(classes)


#def mean_iou(y_true, y_pred):
#    prec = []
#
#    for t in np.arange(0.5, 1.0, 0.05):
#        y_pred_ = tf.to_int32(y_pred > t)
#
#        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, n_label)
#
#        K.get_session().run(tf.local_variables_initializer())
#
#        with tf.control_dependencies([up_opt]):
#            score = tf.identity(score)
#
#        prec.append(score)
#
#    return K.mean(K.stack(prec), axis=0)


# -----------------------------------------------------------------------------

inputs = Input((img_w, img_h, IMG_CHANNELS))

# s = Lambda(lambda x: x / 255) (inputs)

s = inputs

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)

c1 = Dropout(0.2)(c1)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)

p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)

c2 = Dropout(0.2)(c2)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)

p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)

c3 = Dropout(0.2)(c3)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)

p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)

c4 = Dropout(0.2)(c4)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)

p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)

c5 = Dropout(0.2)(c5)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

u6 = concatenate([u6, c4])

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)

c6 = Dropout(0.2)(c6)

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

u7 = concatenate([u7, c3])

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)

c7 = Dropout(0.2)(c7)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

u8 = concatenate([u8, c2])

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)

c8 = Dropout(0.1)(c8)

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

u9 = concatenate([u9, c1], axis=3)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)

c9 = Dropout(0.1)(c9)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

outputs = Conv2D(16, (1, 1), activation='softmax')(c9)

outputs=Reshape((img_w * img_h,n_label))(outputs)

model = Model(inputs=[inputs], outputs=[outputs])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#earlystopper = EarlyStopping(patience=5, verbose=1)

checkpointer = ModelCheckpoint(modelpath + 'unet_1_3.h5', verbose=1, save_best_only=False)

callable = [checkpointer]

# ------------------------------------------------------------------------------


data_set = os.listdir(filepath + '\\RGB')  #####修改

train_set, val_set = rand_split_data(data_set)  #####修改

train_numb = len(train_set)

valid_numb = len(val_set)

print("the number of train data is", train_numb)

print("the number of val data is", valid_numb)

model.load_weights('E:\\data_segmentated\\model\\unet_1_3.h5')

EPOCHS = 5

# BS=16

BS = 5

H = model.fit_generator(generator=generateTrainData(BS, train_set),  # 一个generator或Sequence实例

                        steps_per_epoch=train_numb // BS,  # 从generator产生的步骤的总数（样本批次总数）。通常情况下，应该等于数据集的样本数量除以批量的大小。

                        epochs=EPOCHS,  # 整数，在数据集上迭代的总数。

                        verbose=1,  # 日志显示,0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

                        validation_data=generateValidData(BS, val_set),  # 生成验证集的生成器

                        validation_steps=valid_numb // BS,

                        callbacks=callable, max_queue_size=4)  # 生成器队列的最大容量

plt.style.use("ggplot")

plt.figure()

N = EPOCHS

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy on uNet Satellite Seg")

plt.xlabel("Epoch ")

plt.ylabel("Loss/acc")

plt.legend(loc="lower left")

plt.savefig(resultpath + 'unet_1_3_8.jpg')

