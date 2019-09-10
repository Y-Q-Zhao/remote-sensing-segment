#2019/08/03
'''
生成数据，针对样本少的类别进行扩充,只对train进行分割
'''

from libtiff import TIFF
import numpy as np
import cv2
import random
from tqdm import tqdm
import os
import random

#-----------------------------------------------------------------------------------------------------------------------
im_w=256                #小图大小
im_h=256
X_width=7200            #大图尺寸
X_height=6800
img_each=2000           #每张大图产生的小图数
aug_rate=0.5            #进行数据增强的概率
colormap = [0, 179, 146, 164, 156, 92, 140, 148, 117, 73, 83, 161, 60, 23, 192, 180]
namespace = ['GF2_PMS1__20150212_L1A0000647768-MSS1',
                 'GF2_PMS1__20150902_L1A0001015649-MSS1',
                 'GF2_PMS1__20151203_L1A0001217916-MSS1',
                 'GF2_PMS1__20160327_L1A0001491417-MSS1',
                 'GF2_PMS1__20160816_L1A0001765570-MSS1',
                 'GF2_PMS1__20160827_L1A0001793003-MSS1',
                 'GF2_PMS2__20160225_L1A0001433318-MSS2',
                 'GF2_PMS2__20160510_L1A0001573999-MSS2']
namespace_val= ['GF2_PMS2__20150217_L1A0000658637-MSS2',
                'GF2_PMS1__20160421_L1A0001537716-MSS1']

lab_bmp_path = 'E:\\academic\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train_val_bmp\\'
train_path = 'E:\\academic\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train_val\\'
seg_result_path = 'E:\\academic\\competition\\segment\\data\\data_with_num\\whole_dataset9\\train\\'
#-----------------------------------------------------------------------------------------------------------------------
def read_images(add):
    tif=TIFF.open(add,mode='r')
    image=tif.read_image()
    return image
def save_to_bmp(image,filename):
    '''
    将图像一bmp格式保存到目标文件夹中
    '''
    ad=filename + '.bmp'
    image=image.astype(np.uint8)
    cv2.imwrite(ad,image)
def tif2bmp(filename):
    img=read_images(filename)
    name=filename.split('.')[0]
    cv2.imwrite(name+'.bmp',img)
def all_tif2bmp():
    # path1='E:\\academic\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train_val'
    # path2='E:\\academic\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train_val_bmp'
    dir_list=os.listdir(train_path)
    for filename in dir_list:
        if len(filename.split('_'))==5:
            name=filename.split('.')[0]
            # add_img=path+'\\'+name+'.tif'
            add_lab=train_path+'\\'+name+'_label.tif'
            tif2bmp(add_lab)
#-----------------------------------------------------------------------------------------------------------------------
def gamma_transform(img, gamma):
    '''
    伽马校正
    '''
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)
def random_gamma_transform(img, gamma_vari):
    '''
    随机伽马校正
    '''
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
def rotate(xb, yb, angle):
    '''
    旋转
    '''
    M_rotate = cv2.getRotationMatrix2D((im_w / 2, im_h / 2), angle, 1)  # 获得仿射变换矩阵
    xb = cv2.warpAffine(xb, M_rotate, (im_w, im_h))  # 进行仿射变化
    yb = cv2.warpAffine(yb, M_rotate, (im_w, im_h))
    return xb, yb
def blur(img):
    '''
    均值滤波
    '''
    img = cv2.blur(img, (3, 3));
    return img
def add_noise(img):
    '''
    加噪声
    '''
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img
def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb
#-----------------------------------------------------------------------------------------------------------------------
def special_gen():
    axis = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    step=15
    for order in tqdm(range(8)):
        name=namespace[order]
        img_gray=cv2.imread(lab_bmp_path+name+'_label.bmp',cv2.IMREAD_GRAYSCALE)

        for i in range((X_height-128)//step):
            for j in range((X_width-128)//step):
                for k in range(16):
                    if img_gray[i*step,j*step]==colormap[k] and i*step-128>=0 and i*step+128<=6800 and j*step-128>=0 and j*step+128<7200:
                        axis[k].append((order,i,j))

    print(len(axis[0]), len(axis[1]), len(axis[2]), len(axis[3]),
          len(axis[4]), len(axis[5]), len(axis[6]), len(axis[7]),
          len(axis[8]), len(axis[9]), len(axis[10]), len(axis[11]),
          len(axis[12]), len(axis[13]), len(axis[14]), len(axis[15]))

    index=0

    img_tif = []
    img_lab = []
    for i in range(8):
        name = namespace[i]
        tif = read_images(train_path + name + '.tif')
        lab = read_images(train_path + name + '_label.tif')

        img_tif.append(tif)
        img_lab.append(lab)

    for k in tqdm(range(16)):
        list=axis[k]
        random.shuffle(list)
        random.shuffle(list)
        random.shuffle(list)
        for i in range(4000):
            # name=namespace[list[i][0]]
            img_num=list[i][0]
            h=list[i][1]
            w=list[i][2]

            # add_img_tif = train_path + name + '.tif'
            # add_lab_tif = train_path + name + '_label.tif'

            # image = read_images(add_img_tif)
            # label = read_images(add_lab_tif)

            # aug_img=image[h*step-128:h*step+128,w*step-128:w*step+128,:]
            # aug_lab=label[h*step-128:h*step+128,w*step-128:w*step+128,:]

            aug_img = img_tif[img_num][h * step - 128:h * step + 128, w * step - 128:w * step + 128, :]
            aug_lab = img_lab[img_num][h * step - 128:h * step + 128, w * step - 128:w * step + 128, :]

            aug_img,aug_lab=data_augment(aug_img,aug_lab)

            save_to_bmp(aug_img[:, :, 0],   seg_result_path + 'NIR\\'   + 'kind' + str(k) + '_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\'   + 'kind' + str(k) + '_spe_' + str(index))
            save_to_bmp(aug_lab,            seg_result_path + 'LABEL\\' + 'kind' + str(k) + '_spe_' + str(index))

            index+=1
def test_read_data_all():
    img_tif=[]
    img_lab=[]
    for i in range(8):
        name=namespace[i]
        tif=read_images(train_path + name + '.tif')
        lab=read_images(train_path + name + '_label.tif')

        img_tif.append(tif)
        img_lab.append(lab)
    for i in range(8):
        print(img_tif[i].shape)
        print(img_lab[i].shape)

if __name__ == '__main__':
    # all_tif2bmp()
    special_gen()
    # test_read_data_all()