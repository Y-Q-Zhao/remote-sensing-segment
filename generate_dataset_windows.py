#2019/07/05
#在windows上读取遥感tiff图像数据,并将图片分割成256x256的图像，保存成bmp格式
#原始图片的大小为6800x7200
#其中原始图片是4维，NIR,R.G.B
#label图片为3维，只有RGB三维
#6800/256~=26,7200/256~=28 多余的部分不保留
#裁剪之后最终得到的train图片数量为26x28x8=5824
#liuhc

import numpy as np
from libtiff import TIFF
import cv2
import os

#输入是三个路径add,file_train,file_val
#add为将图像分割之后存放的路径，第一次运行时需要建立如下的文件系统
#-data_segmentated
#--train
#----RGB
#----NIR
#----LABEL
#-validation
#----RGB
#----NIR
#----LABEL
#-test
#----RGB
#----NIR
#----LABEL
#file_train时训练数据存储的地址
#file_val时验证数据存储的地址
add = 'E:\\data_segmentated\\'
train_RGB_path=add+'train\\RGB'
train_NIR_path=add+'train\\NIR'
train_LABEL_path=add+'train\\LABEL'

if not os.path.exists(train_RGB_path):
    os.mkdir(train_RGB_path )
if not os.path.exists(train_NIR_path):
    os.mkdir(train_NIR_path )
if not os.path.exists(train_LABEL_path):
    os.mkdir(train_LABEL_path )
validation_RGB_path=add+'validation\\RGB'
validation_NIR_path=add+'validation\\NIR'
validation_LABEL_path=add+'validation\\LABEL'

if not os.path.exists(validation_RGB_path):
    os.mkdir(validation_RGB_path )
if not os.path.exists(validation_NIR_path):
    os.mkdir(validation_NIR_path )
if not os.path.exists(validation_LABEL_path):
    os.mkdir(validation_LABEL_path )
     
#分割后图像存储文件地址，该文件夹下需要建立3个文件夹：train(RGB,NIR,LABEL),validation(RGB,NIR,LABEL),test
file_train='E:\\train'
#训练数据集存储地址
file_val  ='E:\\val'
#验证数据集存储地址

def read_images(add_tif,single_channel=True):
    '''
    读取tiff原始图像，tiff图像为4维，分别为NIR,R,G,B,尺寸为(6800,7200,4)
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



def save_to_bmp(image,filename,data_kind='train'):
    '''
    将图像一bmp格式保存到目标文件夹中
    '''
    # add='/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/segmented_image/'+data_kind+'/'+filename +'.bmp'
    # add = '/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/date_with_num/' + data_kind + '/' + filename + '.bmp'####
    ad=add+data_kind + '\\' + filename + '.bmp'
    image=image.astype(np.uint8)
    cv2.imwrite(ad,image)

def segmentation_image():
    '''
    分割遥感图像
    '''
    im_w=256
    im_h=256

    # file_train='/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/rssrai2019_semantic_segmentation/train/train'
    # file_val  ='/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/rssrai2019_semantic_segmentation/val/val'

    dir_list_train=os.listdir(file_train)
    dir_list_val = os.listdir(file_val  )

    index1=0
    for filename in dir_list_train:
        if len(filename.split('_'))==1:
            name=filename.split('.')[0]
            add=file_train+'\\'+filename
            add_lab=file_train+'\\'+name+'_label.tif'
            image=read_images(add,single_channel=False)
            label=read_labels(add_lab,single_channel=False)
#            rgb=image[:,:,1:4]
#            nir=image[:,:,0]
#根据文件系统下的文件名称来进行区别
#如：
#1.tiff
#1_label.tiff
#并用对应的函数去读取
            # index1=0
            for i in range(26):
                for j in range(28):
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,0  ],'NIR/'+name+'_'+str(index1),data_kind='train')
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,1:4],'RGB/'+name+'_'+str(index1),data_kind='train')
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,:],'LABEL/'+name+'_'+str(index1),data_kind='train')
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],
                                'NIR\\' + str(index1), data_kind='train')
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],
                                'RGB\\' + str(index1), data_kind='train')
                    save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],
                                'LABEL\\' + str(index1), data_kind='train')
                    index1+=1
            print(filename+'    is done !')

    index2 = 0
    for filename in dir_list_val:
        if len(filename.split('_'))== 1:
            name=filename.split('.')[0]
            add=file_val+'\\'+filename
            add_lab=file_val+'\\'+name+'_label.tif'
            image=read_images(add,single_channel=False)
            label=read_labels(add_lab,single_channel=False)
#            rgb=image[:,:,1:4]
#            nir=image[:,:,0]

            # index2=0
            for i in range(26):
                for j in range(28):
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,0  ],'NIR/'+name+'_'+str(index2),data_kind='validation')
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,1:4],'RGB/'+name+'_'+str(index2),data_kind='validation')
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,:],'LABEL/'+name+'_'+str(index2),data_kind='validation')
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],
                                'NIR\\' + str(index2), data_kind='validation')
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],
                                'RGB\\' + str(index2), data_kind='validation')
                    save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],
                                'LABEL\\'+ str(index2), data_kind='validation')
                    index2+=1
            print(filename + '    is done !')


if __name__ == '__main__':
    # read_images('fileneme.tif')
    # read_all_data()
    segmentation_image()
