#2019/07/11
#transform label from gray to rgb
#zhaoyiqun
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import os
def gray2rgb(lab_gray):         #lab_gray必须为单通道
    '''transform data from gray to rgb'''
    dict={117:[  0,200,  0],        #水田
          164:[150,250,  0],        #水浇地
          179:[150,200,150],        #旱耕地
           83:[200,  0,200],        #园地
           92:[150,  0,250],        #乔木林地
          180:[150,150,250],        #灌木林地
          146:[250,200,  0],        #天然草地
          140:[200,200,  0],        #人工草地
           23:[200,  0,  0],        #工业用地
           73:[250,  0,150],        #城市住宅
          156:[200,150,150],        #城镇住宅
          161:[250,150,150],        #交通运输
           60:[  0,  0,200],        #河流
          148:[  0,150,200],        #湖泊
          192:[  0,200,250],        #坑塘
            0:[  0,  0,  0]}        #其他类别
    lab_h,lab_w=lab_gray.shape
    lab_rgb=np.zeros((lab_h,lab_w,3))
    for i in range(lab_h):
        for j in range(lab_w):
            lab_rgb[i,j,:]=np.array(dict[lab_gray[i,j]])

    return np.uint8(lab_rgb)

def label2rgb(label):   #label必须为单通道
    '''transform data from label to rgb'''
#    dict1={ 0:  0,
#            1: 23,
#            2: 60,
#            3: 73,
#            4: 83,
#            5: 92,
#            6:117,
#            7:140,
#            8:146,
#            9:148,
#           10:156,
#           11:161,
#           12:164,
#           13:179,
#           14:180,
#           15:192}
    dict2 = {117: [  0, 200, 0],  # 水田
            164: [150, 250, 0],  # 水浇地
            179: [150, 200, 150],  # 旱耕地
             83: [200, 0, 200],  # 园地
             92: [150, 0, 250],  # 乔木林地
            180: [150, 150, 250],  # 灌木林地
            146: [250, 200, 0],  # 天然草地
            140: [200, 200, 0],  # 人工草地
             23: [200, 0, 0],  # 工业用地
             73: [250, 0, 150],  # 城市住宅
            156: [200, 150, 150],  # 城镇住宅
            161: [250, 150, 150],  # 交通运输
             60: [  0, 0, 200],  # 河流
            148: [  0, 150, 200],  # 湖泊
            192: [  0, 200, 250],  # 坑塘
              0: [  0, 0, 0]}  # 其他类别
    lab_h, lab_w = label.shape
    lab_rgb = np.zeros((lab_h, lab_w, 3))
    for i in range(lab_h):
        for j in range(lab_w):
            pix=label[i,j]
#            pix=dict1[pix]
            pix=dict2[pix]
            lab_rgb[i, j, :] = np.array(pix)

    return np.uint8(lab_rgb)
from libtiff import TIFF
def save_to_tiff(image,filepath):
    tif=TIFF.open(filepath,mode='w')
    tif.write_image(image[:,:,0:3],compression=None,write_rgb=True)
    tif.close()

if __name__ == '__main__':
    # gray2rgb()
    filepath='E:\\result\\test_result1\\'
    dataset=os.listdir(filepath)
    for i in range(len(dataset)):
        file=dataset[i]
        lab_rgb=label2rgb( cv2.imread(filepath+file, cv2.IMREAD_GRAYSCALE))
        store_path='E:\\test\\'+'label_'+str(i+1)+'.tif'
        save_to_tiff(lab_rgb,store_path)
#        cv2.imwrite('E:\\test\\'+str(i+1)+'.png',lab_rgb)