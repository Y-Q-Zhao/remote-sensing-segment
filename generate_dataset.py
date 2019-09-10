#2019/07/05
#读取tiff图像数据,并将图片分割成256x256的图像，保存成bmp格式
#zhaoyiqun

#注意
#在分割图片时，需要将图片名字中的'(2)'手动去掉
#本程序使用了libtiff库，需要提前安装好

import numpy as np
from libtiff import TIFF
import cv2
import os

add = '/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/date_with_num_test/'
#分割后图像存储文件地址，该文件夹下需要建立3个文件夹：train(RGB,NIR,LABEL),validation(RGB,NIR,LABEL),test
file_train='/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/rssrai2019_semantic_segmentation/train/train'
#训练数据集存储地址
file_val  ='/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/rssrai2019_semantic_segmentation/val/val'
#验证数据集存储地址

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

def save_to_bmp(image,filename,data_kind='train'):
    '''
    将图像一bmp格式保存到目标文件夹中
    '''
    # add='/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/segmented_image/'+data_kind+'/'+filename +'.bmp'
    # add = '/home/yiqunzhao/Desktop/竞赛/语义分割/数据集/date_with_num/' + data_kind + '/' + filename + '.bmp'####
    ad=add+data_kind + '/' + filename + '.bmp'
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
        if len(filename.split('_'))==5:
            name=filename.split('.')[0]
            add=file_train+'/'+filename
            add_lab=file_train+'/'+name+'_label.tif'
            image=read_images(add,single_channel=False)
            label=read_labels(add_lab,single_channel=False)
#             rgb=image[:,:,1:4]
#             nir=image[:,:,0]

            # index1=0
            for i in range(26):
                for j in range(28):
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,0  ],'NIR/'+name+'_'+str(index1),data_kind='train')
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,1:4],'RGB/'+name+'_'+str(index1),data_kind='train')
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,:],'LABEL/'+name+'_'+str(index1),data_kind='train')
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],
                                'NIR/' + str(index1), data_kind='train')
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],
                                'RGB/' + str(index1), data_kind='train')
                    save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],
                                'LABEL/' + str(index1), data_kind='train')
                    index1+=1
            print(filename+'    is done !')

    index2 = 0
    for filename in dir_list_val:
        if len(filename.split('_'))== 5:
            name=filename.split('.')[0]
            add=file_val+'/'+filename
            add_lab=file_val+'/'+name+'_label.tif'
            image=read_images(add,single_channel=False)
            label=read_labels(add_lab,single_channel=False)
            rgb=image[:,:,1:4]
            nir=image[:,:,0]

            # index2=0
            for i in range(26):
                for j in range(28):
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,0  ],'NIR/'+name+'_'+str(index2),data_kind='validation')
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,1:4],'RGB/'+name+'_'+str(index2),data_kind='validation')
                    # save_to_bmp(image[256*i:256+256*i,256*j:256*j+256,:],'LABEL/'+name+'_'+str(index2),data_kind='validation')
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],
                                'NIR/' + str(index2), data_kind='validation')
                    save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],
                                'RGB/' + str(index2), data_kind='validation')
                    save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],
                                'LABEL/'+ str(index2), data_kind='validation')
                    index2+=1
            print(filename + '    is done !')


if __name__ == '__main__':
    # read_images('fileneme.tif')
    # read_all_data()
    segmentation_image()
