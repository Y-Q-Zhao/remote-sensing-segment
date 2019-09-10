import cv2
import numpy as np
import libtiff as TIFF
from sklearn.preprocessing import LabelEncoder
from read_data import read_labels

classes=[0,179,146,164,156,92,140,148,117,73,83,161,60,23,192,180]
labelencoder=LabelEncoder()
labelencoder.fit(classes)

def confusion_matrix(pred,label,conf_mat):
    pred_h,pred_w=pred.shape
    label_h,label_w=label.shape
    if pred_h!=label_h or pred_w!=label_w or pred_channel!=label_channel:
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
    conf_mat=np.zeros((6800,7200))
    if len(pred_list)!=len(label_list):
        return 0
    else:
        for i in range(len(pred_list)):
            pred=read_labels(pred_list[i],single_channel=False)
            label=read_labels(label_list[i],single_channel=False)
            conf_mat=confusion_matrix(pred,label,conf_mat)

        for j in range(16):
            P0 += conf_mat[i, i] * 1.0
        xsum = np.sum(dataMat, axis=1)
        ysum = np.sum(dataMat, axis=0)
        # xsum是个k行1列的向量，ysum是个1行k列的向量
        Pe = float(ysum * xsum) / k ** 2
        P0 = float(P0 / k * 1.0)
        cohens_coefficient = float((P0 - Pe) / (1 - Pe))
        return cohens_coefficient


if __name__ == '__main__':
    kappa()
