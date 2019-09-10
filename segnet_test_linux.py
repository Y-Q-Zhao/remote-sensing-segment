#segnet_1.py
#2019/07/06
#zhaoyiqun
#搭建segnet网络，进行训练和预测

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import os
from tqdm import tqdm
from keras.models import load_model

import read_data

# os.environ["CUDA_VISIBLE_DEVICES"]="1"  # 使用指定的GPU及GPU显存 CUDA_VISIBLE_DEVICES
seed=7
np.random.seed(seed)  # 生成相同的随机数
#image_shape
img_w=256
img_h=256
image_size=256
# 有一个为背景
n_label=15+1

classes=[0., 179., 146., 164., 156., 92., 140., 148., 117., 73., 83., 161., 60., 23., 192., 180.]
labelencoder = LabelEncoder()  # 标签标准化
labelencoder.fit(classes)

filepath='/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/date_with_num/'                                                  #####修改数据集路径
modelpath='/home/yiqunzhao/Desktop/竞赛/语义分割/code/semantic_segmentation/model/'                                    #####修改模型路径
resultpath='/home/yiqunzhao/Desktop/竞赛/语义分割/code/semantic_segmentation/result/'                                  #####修改结果保存路径

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读入灰度图片
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0  # 归一化
    return img

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
            tif_img = np.zeros((img_w, img_h, 4))
            img_rgb=load_img(filepath+'train/RGB/'+url)                                                              #####修改路径
            img_nir=load_img(filepath+'train/NIR/'+url)                                                              #####修改路径
            tif_img[:,:,0:3]=img_rgb
            tif_img[:, :, 3]=img_nir[:,:,0]
            img=tif_img
            img=img_to_array(img)
            tif_img=np.zeros((img_w,img_h,4))
            img=img_to_array(img)            #将图片转化成数组
            train_data.append(img)

            label=load_img(filepath+'train/LABEL/'+url,grayscale=True)      #####修改
            label=img_to_array(label).reshape((img_w*img_h,))
            # print label.shape
            train_label.append(label)
            if batch % batch_size==0:
                #print 'get enough bacth!\n'
                train_data=np.array(train_data)       #batch_size*img_w*img_h*3
                #标签转化成one-hot
                train_label=np.array(train_label).flatten()       #降成一维向量
                train_label=labelencoder.transform(train_label)   #转换成连续的数值型变量
                train_label=to_categorical(train_label, num_classes=n_label)          #将类别向量转换为二进制（只有0和1）的矩阵类型表示
                train_label=train_label.reshape((batch_size,img_w * img_h,n_label))

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
            tif_img=np.zeros((img_w, img_h, 4))
            img_rgb=load_img(filepath+'validation/RGB/'+url)                                                         #####修改
            img_nir=load_img(filepath+'validation/NIR/'+url)                                                         #####修改
            tif_img[:,:,0:3]=img_rgb
            tif_img[:, :, 3]=img_nir[:,:,0]
            img=tif_img
            img=img_to_array(img)
            tif_img=np.zeros((img_w,img_h,4))
            img=img_to_array(img)  # 将图片转化成数组
            valid_data.append(img)

            label=load_img(filepath+'validation/LABEL/'+url, grayscale=True)                                         #####修改
            label=img_to_array(label).reshape((img_w*img_h,))
            # print label.shape
            valid_label.append(label)
            if batch%batch_size==0:
                # print 'get enough bacth!\n'
                valid_data=np.array(valid_data)  # batch_size*img_w*img_h*3
                # 标签转化成one-hot
                valid_label=np.array(valid_label).flatten()  # 降成一维向量
                valid_label=labelencoder.transform(valid_label)  # 转换成连续的数值型变量
                valid_label=to_categorical(valid_label,num_classes=n_label)  # 将类别向量转换为二进制（只有0和1）的矩阵类型表示
                valid_label=valid_label.reshape((batch_size,img_w*img_h,n_label))

                yield (valid_data,valid_label)  # 输出数据和标签
                valid_data=[]
                valid_label=[]
                batch=0

def SegNet():
    model=Sequential()
    #encoder
    # model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(4,img_w,img_h),padding='same',activation='relu'))
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(img_w,img_h,4),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    #(128,128)
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    #(64,64)
    # model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # #(32,32)
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # #(16,16)
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # #(8,8)
    # #decoder
    model.add(UpSampling2D(size=(2,2)))
    #(16,16)
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(UpSampling2D(size=(2,2)))
    # #(32,32)
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(UpSampling2D(size=(2,2)))
    # #(64,64)
    # model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(UpSampling2D(size=(2,2)))
    # #(128,128)
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2,2)))
    #(256,256)
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(3,img_w,img_h),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1, 1),padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(n_label,(1,1),strides=(1,1),padding='same'))
    model.add(Reshape((img_w*img_h,n_label)))

    #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  #构建编译网络
    model.summary()  #输出模型各层的参数状况
    return model

def train():
    EPOCHS=30
    # BS=16
    BS=10                                                                                                            #####可修改
    model=SegNet()
    modelcheck=ModelCheckpoint(modelpath+'model_test_net.h5',monitor='val_acc',save_best_only=True,mode='max')       #####修改模型名称
    #监控模型中的'val_acc'参数，当该参数增大时，保存模型（保存最佳模型）
    callable=[modelcheck]  #回调函数
    train_set=os.listdir(filepath+'train/RGB')                                                                       #####修改
    val_set=os.listdir(filepath+'validation/RGB')                                                                    #####修改
    train_numb=len(train_set)
    valid_numb=len(val_set)
    print ("the number of train data is",train_numb)
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateTrainData(BS,train_set),  #一个generator或Sequence实例
                            steps_per_epoch=train_numb//BS,        #从generator产生的步骤的总数（样本批次总数）。通常情况下，应该等于数据集的样本数量除以批量的大小。
                            epochs=EPOCHS,                         #整数，在数据集上迭代的总数。
                            verbose=1,                             #日志显示,0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                            validation_data=generateValidData(BS,val_set),  #生成验证集的生成器
                            validation_steps=valid_numb//BS,
                            callbacks=callable,
                            max_queue_size=10)  #生成器队列的最大容量
    #分批训练,生成器，返回一个history

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N=EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(resultpath+'loss_accuracy.jpg')             #保存图片

def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model=load_model(modelpath+'model_test_net.h5')  # 载入模型                      #####修改路径
    stride=256
    TEST_SET=os.listdir(filepath+'test/image')                                      #####修改路径
    print(TEST_SET)
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        # load the image
        # image = cv2.imread('./test/' + path)  # 读取图片
        image=read_data.read_images(filepath+'test/image/'+path,single_channel=False)#####修改路径
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

                crop = np.expand_dims(crop, axis=0)
                # print 'crop:',crop.shape
                pred = model.predict_classes(crop, verbose=2)  # 预测的是类别，打印出来的值就是类别号
                pred = labelencoder.inverse_transform(pred[0])
                # print (np.unique(pred))
                pred = pred.reshape((256, 256)).astype(np.uint8)
                # print 'pred:',pred.shape
                mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred[:, :]

        cv2.imwrite(filepath+'test/test_result/'+str(n + 1) + '.png', mask_whole[0:h, 0:w])#####修改路径
    
if __name__ == '__main__':
    train()
