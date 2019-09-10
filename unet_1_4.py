#coding=utf-8

from keras import regularizers
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Lambda,Activation,Deconv2D,Concatenate,Permute,Softmax,SpatialDropout2D,Add,Multiply
from keras import backend as K
#import tensorflow as tf
from keras import optimizers
if K.image_dim_ordering() == 'tf':
    bn_axis = 3
else:
    bn_axis = 1

def conv_block(input_tensor, filters, kernel_size, strides, name, padding='same', dila=1):
    x = Conv2D(filters, kernel_size, strides=strides, name= name, padding=padding, kernel_initializer='he_uniform',kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-5),dilation_rate=dila)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='bn_'+ name)(x)
    x = Activation('relu')(x)
    return x

def backend_expand_dims_1(x):
    return K.expand_dims(x, axis=1)

def backend_expand_dims_last(x):
    return K.expand_dims(x, axis=-1)

def backend_dot(x):
    return K.batch_dot(x[0], x[1])

def global_context_block(x, channels):
    bs, h, w, c = x.shape.as_list()
    input_x = x
    input_x = Reshape((h * w, c))(input_x)  # [N, H*W, C]
    input_x = Permute((2,1))(input_x)       # [N, C, H*W]
    input_x = Lambda(backend_expand_dims_1,name='a')(input_x)  # [N, 1, C, H*W]

    context_mask = Conv2D(1,(1,1), name='gc-conv0')(x)
    context_mask = Reshape((h * w, 1))(context_mask) # [N, H*W, 1]
    context_mask = Softmax(axis=1)(context_mask)  # [N, H*W, 1]
    context_mask = Permute((2,1))(context_mask)   # [N, 1, H*W]
    context_mask = Lambda(backend_expand_dims_last,name='b')(context_mask) # [N, 1, H*W, 1]

    context = Lambda(backend_dot,name='c')([input_x, context_mask])
    context = Reshape((1,1,c))(context) # [N, 1, 1, C]

    context_transform = conv_block(context, channels, 1, strides=1, name='gc-conv1')
    context_transform = Conv2D(c,(1,1), name='gc-conv2')(context_transform)
    context_transform = Activation('sigmoid')(context_transform)
    x = Multiply()([x , context_transform])

    context_transform = conv_block(context, channels, 1, strides=1, name='gc-conv3')
    context_transform = Conv2D(c,(1,1), name='gc-conv4')(context_transform)
    x = Add()([x,context_transform])

    return x
# -----



import cv2

import os

import numpy as np

import random

import warnings

import matplotlib.pyplot as plt

from keras.models import Model



from keras import backend as K





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

drop_rate=0.2

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

from keras.layers import Input

from keras.callbacks import  ModelCheckpoint

matplotlib.use("Agg")

# import matplotlib.pyplot as plt

# import argparse

import numpy as np

from sklearn.preprocessing import LabelEncoder

n_label = 15 + 1

classes = [0., 179., 146., 164., 156., 92., 140., 148., 117., 73., 83., 161., 60., 23., 192., 180.]

labelencoder = LabelEncoder()  # 标签标准化

labelencoder.fit(classes)


# 模型部分

# -----------------------------------------------------------------------------

inputs = Input((img_w, img_h, IMG_CHANNELS))

# s = Lambda(lambda x: x / 255) (inputs)

# ---------left branch -----
x = conv_block(inputs, 32, (3, 3), strides=1, name='L_conv1-1')
x = SpatialDropout2D(drop_rate)(x)
L1 = conv_block(x, 32, (3, 3), strides=1, name='L_conv1-2')
x = conv_block(L1, 32, (3, 3), strides=2, name='L_conv1-3')
#   400 -> 200

x = conv_block(x, 64, (3, 3), strides=1, name='L_conv2-1')
x = SpatialDropout2D(drop_rate)(x)
L2 = conv_block(x, 64, (3, 3), strides=1, name='L_conv2-2')
x = conv_block(L2, 32, (3, 3), strides=2, name='L_conv2-3')
#   200 -> 100

x = conv_block(x, 128, (3, 3), strides=1, name='L_conv3-1')
x = SpatialDropout2D(drop_rate)(x)
L3 = conv_block(x, 128, (3, 3), strides=1, name='L_conv3-2')
x = conv_block(L3, 32, (3, 3), strides=2, name='L_conv3-3')
#   100 -> 50

x = conv_block(x, 256, (3, 3), strides=1, name='L_conv4-1')
x = SpatialDropout2D(drop_rate)(x)
L4 = conv_block(x, 256, (3, 3), strides=1, name='L_conv4-2')
x = conv_block(L4, 32, (3, 3), strides=2, name='L_conv4-3')
#   50 -> 25

x = conv_block(x, 512, (3, 3), strides=1, name='L_conv5-1')
x = conv_block(x, 512, (3, 3), strides=1, dila=2, name='L_conv5-2')
x = SpatialDropout2D(drop_rate)(x)
x = conv_block(x, 512, (3, 3), strides=1, dila=2, name='L_conv5-3')
L5 = conv_block(x, 512, (3, 3), strides=1, name='L_conv5-4')

#    25


# ---------Right branch -----

#   25 -> 50
x = Deconv2D(256, kernel_size=2, strides=2, padding='same',name='R_conv1-1')(L5)
x = BatchNormalization(axis=bn_axis, name='R_conv1-1_' + 'bn')(x)
x = conv_block(Concatenate(axis=-1)([x, L4]), 256, (3, 3), strides=1, name='R_conv1-2')
x = SpatialDropout2D(drop_rate)(x)
x = conv_block(x, 256, (3, 3), strides=1, name='R_conv1-3')
R_out1 = Conv2D(n_label,(1,1),name='R_out1')(x)

#   50 -> 100
x = Deconv2D(128, kernel_size=2, strides=2, padding='same', name='R_conv2-1')(x)
x = BatchNormalization(axis=bn_axis, name='R_conv2-1_' + 'bn')(x)
x = conv_block(Concatenate(axis=-1)([x, L3]), 128, (3, 3), strides=1, name='R_conv2-2')
x = SpatialDropout2D(drop_rate)(x)
x = conv_block(x, 128, (3, 3), strides=1, name='R_conv2-3')
R_out2 = Conv2D(n_label, (1, 1), name='R_out2')(x)

#   100 -> 200
x = Deconv2D(64, kernel_size=2, strides=2, padding='same', name='R_conv3-1')(x)
x = BatchNormalization(axis=bn_axis, name='R_conv3-1_' + 'bn')(x)
x = conv_block(Concatenate(axis=-1)([x, L2]), 64, (3, 3), strides=1, name='R_conv3-2')
x = SpatialDropout2D(drop_rate)(x)
x = conv_block(x, 64, (3, 3), strides=1, name='R_conv3-3')
R_out3 = Conv2D(n_label, (1, 1), name='R_out3')(x)

#   200 -> 400
x = Deconv2D(32, kernel_size=2, strides=2, padding='same', name='R_conv4-1')(x)
x = BatchNormalization(axis=bn_axis, name='R_conv4-1_' + 'bn')(x)
x = conv_block(Concatenate(axis=-1)([x, L1]), 32, (3, 3), strides=1, name='R_conv4-2')
x = SpatialDropout2D(drop_rate)(x)
x = conv_block(x, 32, (3, 3), strides=1, name='R_conv4-3')
R_out4 = Conv2D(n_label, (1, 1), name='R_out4')(x)

# ---------Recoding branch -----

x = conv_block(R_out4, 32, (1, 1), strides=1, name='E_conv1-1')
x = conv_block(x, 32, (3, 3), strides=1, name='E_conv1-2')
x = SpatialDropout2D(drop_rate)(x)
x = conv_block(x, 32, (3, 3), strides=2, name='E_conv1-3')
#   400 -> 200

x = conv_block(Concatenate(axis=-1)([x, conv_block(R_out3,64, (1, 1), strides=1,name='c1')]), 64, (3, 3), strides=1, name='E_conv2-1')
x = conv_block(x, 64, (3, 3), strides=1, name='E_conv2-2')
x = SpatialDropout2D(drop_rate)(x)
x = conv_block(x, 64, (3, 3), strides=2, name='E_conv2-3')
#   200 -> 100

x = conv_block(Concatenate(axis=-1)([x, conv_block(R_out2,128, (1, 1), strides=1,name='c2')]), 128, (3, 3), strides=1, name='E_conv3-1')
x = conv_block(x, 128, (3, 3), strides=1, name='E_conv3-2')
x = SpatialDropout2D(drop_rate)(x)
x = conv_block(x, 128, (3, 3), strides=2, name='E_conv3-3')
#   100 -> 50

x = conv_block(Concatenate(axis=-1)([x, conv_block(R_out1,256, (1, 1), strides=1,name='c3')]), 256, (3, 3), strides=1, name='E_conv4-1')
x = conv_block(x, 256, (3, 3), strides=1, name='E_conv4-2')
x = SpatialDropout2D(drop_rate)(x)
x = conv_block(x, 256, (3, 3), strides=1, dila=2, name='E_conv4-3')
x = conv_block(x, 256, (3, 3), strides=1, dila=2, name='E_conv4-4')
x = conv_block(x, 256, (3, 3), strides=1, name='E_conv4-5')
#   50

x = global_context_block(x, channels=64)
# -----------------------------------------
final_out = Conv2D(n_label,(1,1), name='final_out')(x)
final_out = UpSampling2D(size=(8,8))(final_out)

final_out = Activation('softmax',name='l0')(Reshape((256 * 256, n_label))(final_out))

model = Model(inputs=[inputs], outputs=[final_out])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#earlystopper = EarlyStopping(patience=5, verbose=1)

checkpointer = ModelCheckpoint(modelpath + 'unet_1_4.h5', monitor='val_acc', verbose=1, save_best_only=True  ,mode='auto')

callable = [checkpointer]

# ------------------------------------------------------------------------------


data_set = os.listdir(filepath + '\\RGB')  #####修改

train_set, val_set = rand_split_data(data_set)  #####修改

train_numb = len(train_set)

valid_numb = len(val_set)

print("the number of train data is", train_numb)

print("the number of val data is", valid_numb)

model.load_weights('E:\\data_segmentated\\model\\unet_1_4.h5') #第一次运行时注释此行

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

#---------------------------------画图部分

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

plt.savefig(resultpath + 'unet_1_4.jpg')

