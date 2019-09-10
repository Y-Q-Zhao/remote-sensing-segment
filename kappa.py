#2019/07/23
#zhaoyiqun
#计算tiff格式标签的kappa系数

import cv2
import numpy as np
import libtiff as TIFF
from sklearn.preprocessing import LabelEncoder
from read_data import read_labels

# classes=[0,179,146,164,156,92,140,148,117,73,83,161,60,23,192,180]
classes=[117,192,179,83,73,161,192,177,60,92,165,180,23,111,146,0]
labelencoder=LabelEncoder()
labelencoder.fit(classes)

def confusion_matrix(pred,label,conf_mat):
    pred_h,pred_w=pred.shape
    label_h,label_w=label.shape
    if pred_h!=label_h or pred_w!=label_w:
        print('the pred ang label have different shape!')
        return 0
    else:
        h=pred_h
        w=pred_w
    pred=np.array(pred).flatten()
    pred=labelencoder.transform(pred)
    label=np.array(label).flatten()
    label=labelencoder.transform(label)
    for i in range(h*w):
        conf_mat[pred[i],label[i]]+=1
    return conf_mat

def kappa(pred_list,label_list):
    conf_mat=np.zeros((16,16))
    if len(pred_list)!=len(label_list):
        print('wrong!')
        return 0
    else:
        for i in range(len(pred_list)):
            pred=read_labels(pred_list[i],single_channel=False)
            label=read_labels(label_list[i],single_channel=False)
            pred=cv2.cvtColor(pred,cv2.COLOR_RGB2GRAY)
            label=cv2.cvtColor(label,cv2.COLOR_RGB2GRAY)
            conf_mat=confusion_matrix(pred,label,conf_mat)
        P0=0
        print(conf_mat)
        for j in range(16):
            P0 += conf_mat[j, j]
        xsum = np.sum(conf_mat, axis=1)
        ysum = np.sum(conf_mat, axis=0)
        # xsum是个k行1列的向量，ysum是个1行k列的向量
        k=np.sum(np.sum(conf_mat,axis=1),axis=0)
        Pe = float(np.dot(ysum ,xsum)) / (k*k)
        P0 = float(P0 / k)
        cohens_coefficient = float((P0 - Pe) / (1 - Pe))
        return cohens_coefficient

def cacul_kappa():
    path_pred='D:\\academic\\competition\\segment\\data\\data_with_num\\test\\test_result12\\tif_3.tif'
    path_label='D:\\academic\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\GF2_PMS1__20150212_L1A0000647768-MSS1_label.tif'
    pred_list=[path_pred]
    label_list=[path_label]
    kappa_rate=kappa(pred_list,label_list)
    print('kappa:',kappa_rate)

if __name__ == '__main__':
    cacul_kappa()