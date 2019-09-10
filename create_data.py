#2019/07/29
'''
生成数据，针对样本少的类别进行扩充
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
#-----------------------------------------------------------------------------------------------------------------------
def read_images(add):
    tif=TIFF.open(add,mode='r')
    image=tif.read_image()
    return image

def save2tiff(image,filename):
    tif.TIFF.open(filename,mode='w')
    tf.write_image(image[:,:,0:3],compression=None)
    tf.close()

def tif2bmp(filename):
    img=read_images(filename)
    name=filename.split('.')[0]
    cv2.imwrite(name+'.bmp',img)

def save_to_bmp(image,filename):
    '''
    将图像一bmp格式保存到目标文件夹中
    '''
    ad=filename + '.bmp'
    image=image.astype(np.uint8)
    cv2.imwrite(ad,image)

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
def expand_tif_1():
    img_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\GF2_PMS1__20150212_L1A0000647768-MSS1.tif'
    lab_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\GF2_PMS1__20150212_L1A0000647768-MSS1_label.tif'
    lab_path_gray='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\GF2_PMS1__20150212_L1A0000647768-MSS1_label.bmp'

    img=read_images(img_path)
    lab=read_images(lab_path)
    lab_gray=cv2.imread(lab_path_gray,cv2.IMREAD_GRAYSCALE)

    colormap=[0,179,146,164,156,92,140,148,117,73,83,161,60,23,192,180]

    coloraccount=np.zeros((16))

    for i in tqdm(range(6800)):
        for j in range(7200):
            flag=True
            for k in range(16):
                if lab_gray[i,j]==colormap[k]:
                    coloraccount[k]+=1
                    flag=False
            if flag:
                print('the color map is wrong !')

    print(coloraccount)

def all_tif2bmp():
    path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train'
    dir_list=os.listdir(path)
    for filename in dir_list:
        if len(filename.split('_'))==5:
            name=filename.split('.')[0]
            # add_img=path+'\\'+name+'.tif'
            add_lab=path+'\\'+name+'_label.tif'
            tif2bmp(add_lab)

def account_units():
    filepath='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp'
    img_list=os.listdir(filepath)
    colormap = [0, 179, 146, 164, 156, 92, 140, 148, 117, 73, 83, 161, 60, 23, 192, 180]

    for file in img_list:
        lab_gray=cv2.imread(filepath+'\\'+file,cv2.IMREAD_GRAYSCALE)

        coloraccount = np.zeros((16))

        # np.savetxt(filepath + '\\' + file.split('.')[0] + '.txt', coloraccount)

        for i in tqdm(range(6800)):
            for j in range(7200):
                flag = True
                for k in range(16):
                    if lab_gray[i, j] == colormap[k]:
                        coloraccount[k] += 1
                        flag = False
                if flag:
                    print('the color map is wrong !')

        print(coloraccount)
        np.savetxt(filepath+'\\'+file.split('.')[0]+'.txt',coloraccount)

def account_all_img():
    account_all=np.zeros((16))
    filepath = 'E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp'
    img_list = os.listdir(filepath)
    for file in img_list:
        if file.split('.')[-1]=='txt':
            account=np.loadtxt(filepath+'\\'+file)
            account_all=account_all+account

    print(account_all)
    np.savetxt(filepath+'\\'+'account_all.txt',account_all)
#-----------------------------------------------------------------------------------------------------------------------
def cacul_units(img,gray_scale):
    h,w=img.shape
    account=0
    for i in range(h):
        for j in range(w):
            if img[i,j]==gray_scale:
                account+=1
    if account/(h*w)>=0.1:
        return True
    else:
        return False

def special_segment_img1():
    '''
    分割遥感图像
    GF2_PMS1__20150212_L1A0000647768-MSS1
    选择特定的目标的图像
    '''
    name='GF2_PMS1__20150212_L1A0000647768-MSS1'

    lab_bmp_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
    train_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
    account_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\label_account\\'

    seg_result_path='E:\\competition\\segment\\data\\sepcial_seg\\'

    add_img_tif=train_path+name+'.tif'
    add_lab_tif=train_path+name+'_label.tif'
    add_lab_bmp=lab_bmp_path+name+'_label.bmp'
    # add_img_act=account_path+name+'_label.txt'

    image=read_images(add_img_tif)
    label=read_images(add_lab_tif)
    print(label.shape)
    lab_gray=cv2.imread(add_lab_bmp,cv2.IMREAD_GRAYSCALE)
    print(lab_gray.shape)
    # img_account=np.load(add_img_act)

    ##顺序分割部分
    index=0
    account=0
    for i in range(26):
        for j in range(28):
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],seg_result_path+'NIR\\' + 'tif1_ori_'+str(index))
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],seg_result_path+'RGB\\'+'tif1_ori_'+str(index))
            save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],seg_result_path+'LABEL\\'+'tif1_ori_'+str(index))
            index+=1
            account+=1

    ##随机分割部分
    index=0
    while account < img_each:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        if np.random.random() < aug_rate:
            aug_img, aug_lab=data_augment(ran_image,ran_lab)
        else:
            aug_img=ran_image
            aug_lab=ran_lab
        save_to_bmp(aug_img[:,:,0],seg_result_path+'NIR\\' + 'tif1_ran_'+str(index))
        save_to_bmp(aug_img[:,:,1:4],seg_result_path+'RGB\\'+'tif1_ran_'+str(index))
        save_to_bmp(aug_lab,seg_result_path+'LABEL\\'+'tif1_ran_'+str(index))
        index+=1
        account+=1

    ##特别分割部分
    account_4=20
    account_5=100
    account_6=100
    account_8=100
    account_10=20
    account_11=100
    account_12=100
    account_13=100
    account_14=100
    account_15=100
    index=0
    time=0
    while account_4!=0 or account_5!=0 or account_6!=0 or account_8!=0 or account_10!=0 or account_11!=0 or account_12!=0 or account_13!=0 or account_14!=0 or account_15!=0:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        ran_gray=lab_gray[random_height: random_height + im_h, random_width: random_width + im_w]

        if cacul_units(ran_gray,colormap[3]) and account_4>0:
            account_4-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index+=1
        elif cacul_units(ran_gray,colormap[4]) and account_5>0:
            account_5-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[5]) and account_6>0:
            account_6-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[7]) and account_8>0:
            account_8-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[9]) and account_10>0:
            account_10-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[10]) and account_11>0:
            account_11-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[11]) and account_12>0:
            account_12-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[12]) and account_13>0:
            account_13-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[13]) and account_14>0:
            account_14=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[14]) and account_15>0:
            account_15-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif1_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif1_spe_' + str(index))
            index += 1

        time+=1
        if time>=1000000:
            break
    print(time)

def special_segment_img2():
    '''
    分割遥感图像
    GF2_PMS1__20150902_L1A0001015649-MSS1
    选择特定的目标的图像
    '''
    name='GF2_PMS1__20150902_L1A0001015649-MSS1'

    lab_bmp_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
    train_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
    account_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\label_account\\'

    seg_result_path='E:\\competition\\segment\\data\\sepcial_seg\\'

    add_img_tif=train_path+name+'.tif'
    add_lab_tif=train_path+name+'_label.tif'
    add_lab_bmp=lab_bmp_path+name+'_label.bmp'
    # add_img_act=account_path+name+'_label.txt'

    image=read_images(add_img_tif)
    label=read_images(add_lab_tif)
    print(label.shape)
    lab_gray=cv2.imread(add_lab_bmp,cv2.IMREAD_GRAYSCALE)
    print(lab_gray.shape)
    # img_account=np.load(add_img_act)

    ##顺序分割部分
    index=0
    account=0
    for i in range(26):
        for j in range(28):
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],seg_result_path+'NIR\\' + 'tif2_ori_'+str(index))
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],seg_result_path+'RGB\\'+'tif2_ori_'+str(index))
            save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],seg_result_path+'LABEL\\'+'tif2_ori_'+str(index))
            index+=1
            account+=1
    print('sequencial part is done !')
    ##随机分割部分
    index=0
    while account < img_each:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        if np.random.random() < aug_rate:
            aug_img, aug_lab=data_augment(ran_image,ran_lab)
        else:
            aug_img=ran_image
            aug_lab=ran_lab
        save_to_bmp(aug_img[:,:,0],seg_result_path+'NIR\\' + 'tif2_ran_'+str(index))
        save_to_bmp(aug_img[:,:,1:4],seg_result_path+'RGB\\'+'tif2_ran_'+str(index))
        save_to_bmp(aug_lab,seg_result_path+'LABEL\\'+'tif2_ran_'+str(index))
        index+=1
        account+=1
    print('randon part is done !')
    ##特别分割部分
    account_4=20
    account_5=100
    account_9=20
    account_10=20
    account_13=100
    account_15=100
    index=0
    time=0
    while account_4!=0 or account_5!=0 or account_9!=0 or account_10!=0 or account_13!=0 or  account_15!=0:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        ran_gray=lab_gray[random_height: random_height + im_h, random_width: random_width + im_w]

        flag=False
        if cacul_units(ran_gray,colormap[3]) and account_4>0:
            account_4-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            # index+=1
        elif cacul_units(ran_gray,colormap[4]) and account_5>0:
            account_5-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[8]) and account_9>0:
            account_9-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[9]) and account_10>0:
            account_10-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[12]) and account_13>0:
            account_13-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[14]) and account_15:
            account_15-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            # index += 1
        if flag:
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            index+=1
            print('account_4:',account_4,
                  '  account_5:',account_5,
                  '  account_9:',account_9,
                  '  account_10:',account_10,
                  '  account_13:',account_13,
                  '  account_15:',account_15)

        time+=1
        if time>=1000000:
            break
    print(time)

def special_segment_img3():
    '''
    分割遥感图像
    GF2_PMS1__20151203_L1A0001217916-MSS1_label
    选择特定的目标的图像
    '''
    name='GF2_PMS1__20151203_L1A0001217916-MSS1_label'

    lab_bmp_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
    train_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
    account_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\label_account\\'

    seg_result_path='E:\\competition\\segment\\data\\sepcial_seg\\'

    add_img_tif=train_path+name+'.tif'
    add_lab_tif=train_path+name+'_label.tif'
    add_lab_bmp=lab_bmp_path+name+'_label.bmp'
    # add_img_act=account_path+name+'_label.txt'

    image=read_images(add_img_tif)
    label=read_images(add_lab_tif)
    print(label.shape)
    lab_gray=cv2.imread(add_lab_bmp,cv2.IMREAD_GRAYSCALE)
    print(lab_gray.shape)
    # img_account=np.load(add_img_act)

    ##顺序分割部分
    index=0
    account=0
    for i in range(26):
        for j in range(28):
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],seg_result_path+'NIR\\' + 'tif3_ori_'+str(index))
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],seg_result_path+'RGB\\'+'tif3_ori_'+str(index))
            save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],seg_result_path+'LABEL\\'+'tif3_ori_'+str(index))
            index+=1
            account+=1

    ##随机分割部分
    index=0
    while account < img_each:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        if np.random.random() < aug_rate:
            aug_img, aug_lab=data_augment(ran_image,ran_lab)
        else:
            aug_img=ran_image
            aug_lab=ran_lab
        save_to_bmp(aug_img[:,:,0],seg_result_path+'NIR\\' + 'tif3_ran_'+str(index))
        save_to_bmp(aug_img[:,:,1:4],seg_result_path+'RGB\\'+'tif3_ran_'+str(index))
        save_to_bmp(aug_lab,seg_result_path+'LABEL\\'+'tif3_ran_'+str(index))
        index+=1
        account+=1

    ##特别分割部分
    account_2=20
    account_10=20
    account_12=100
    account_13=100
    account_14=100

    index=0
    time=0
    while account_2!=0 or account_10!=0 or account_12!=0 or account_13!=0 or account_14!=0 :
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        ran_gray=lab_gray[random_height: random_height + im_h, random_width: random_width + im_w]

        flag=False
        if cacul_units(ran_gray,colormap[1]) and account_2>0:
            account_2-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif3_spe_' + str(index))
            # index+=1
        elif cacul_units(ran_gray,colormap[9]) and account_10>0:
            account_10-=1
            flag = True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif3_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[11]) and account_12>0:
            account_12-=1
            flag = True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif3_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[12]) and account_13>0:
            account_13-=1
            flag = True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif3_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[13]) and account_14>0:
            account_14-=1
            flag = True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif3_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif3_spe_' + str(index))
            # index += 1

        if flag:
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            index+=1
            print('account_2:',account_2,
                  '  account_10:',account_10,
                  '  account_12:',account_12,
                  '  account_13:',account_13,
                  '  account_14:',account_14)
        time += 1
        if time >= 1000000:
            break
    print(time)

def special_segment_img4():
    '''
    分割遥感图像
    GF2_PMS1__20160327_L1A0001491417-MSS1
    选择特定的目标的图像
    '''
    name='GF2_PMS1__20160327_L1A0001491417-MSS1'

    lab_bmp_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
    train_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
    account_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\label_account\\'

    seg_result_path='E:\\competition\\segment\\data\\sepcial_seg\\'

    add_img_tif=train_path+name+'.tif'
    add_lab_tif=train_path+name+'_label.tif'
    add_lab_bmp=lab_bmp_path+name+'_label.bmp'
    # add_img_act=account_path+name+'_label.txt'

    image=read_images(add_img_tif)
    label=read_images(add_lab_tif)
    print(label.shape)
    lab_gray=cv2.imread(add_lab_bmp,cv2.IMREAD_GRAYSCALE)
    print(lab_gray.shape)
    # img_account=np.load(add_img_act)

    ##顺序分割部分
    index=0
    account=0
    for i in range(26):
        for j in range(28):
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],seg_result_path+'NIR\\' + 'tif4_ori_'+str(index))
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],seg_result_path+'RGB\\'+'tif4_ori_'+str(index))
            save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],seg_result_path+'LABEL\\'+'tif4_ori_'+str(index))
            index+=1
            account+=1

    ##随机分割部分
    index=0
    while account < img_each:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        if np.random.random() < aug_rate:
            aug_img, aug_lab=data_augment(ran_image,ran_lab)
        else:
            aug_img=ran_image
            aug_lab=ran_lab
        save_to_bmp(aug_img[:,:,0],seg_result_path+'NIR\\' + 'tif4_ran_'+str(index))
        save_to_bmp(aug_img[:,:,1:4],seg_result_path+'RGB\\'+'tif4_ran_'+str(index))
        save_to_bmp(aug_lab,seg_result_path+'LABEL\\'+'tif4_ran_'+str(index))
        index+=1
        account+=1

    ##特别分割部分
    account_2=20
    account_5=100
    account_8=100
    account_10=20
    account_11=100
    account_12=100
    account_14=100
    account_15=100
    account_16=100
    index=0
    time=0
    while account_2!=0 or account_5!=0 or account_8!=0 or account_10!=0 or account_11!=0 or account_12!=0 or account_14!=0 or account_15!=0 or account_16!=0:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        ran_gray=lab_gray[random_height: random_height + im_h, random_width: random_width + im_w]

        flag=False
        if cacul_units(ran_gray,colormap[1]) and account_2>0:
            account_2-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif4_spe_' + str(index))
            # index+=1
        elif cacul_units(ran_gray,colormap[4]) and account_5>0:
            account_5-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif4_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[7]) and account_8>0:
            account_8-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif4_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[9]) and account_10>0:
            account_10-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif4_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[10]) and account_11>0:
            account_11-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif4_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[11]) and account_12>0:
            account_12-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif4_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[13]) and account_14>0:
            account_14-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif4_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[14]) and account_15>0:
            account_15=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif4_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[15]) and account_16>0:
            account_16-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif4_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif4_spe_' + str(index))
            # index += 1

        if flag:
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            index+=1
            print('account_2:',account_2,
                  '  account_5:',account_5,
                  '  account_8:',account_8,
                  '  account_10:',account_10,
                  '  account_11:',account_11,
                  '  account_12:',account_12,
                  '  account_14:',account_14,
                  '  account_15:',account_15,
                  '  account_16:',account_16)

        time += 1
        if time >= 1000000:
            break
    print(time)

def special_segment_img5():
    '''
    分割遥感图像
    GF2_PMS1__20160816_L1A0001765570-MSS1
    选择特定的目标的图像
    '''
    name='GF2_PMS1__20160816_L1A0001765570-MSS1'

    lab_bmp_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
    train_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
    account_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\label_account\\'

    seg_result_path='E:\\competition\\segment\\data\\sepcial_seg\\'

    add_img_tif=train_path+name+'.tif'
    add_lab_tif=train_path+name+'_label.tif'
    add_lab_bmp=lab_bmp_path+name+'_label.bmp'
    # add_img_act=account_path+name+'_label.txt'

    image=read_images(add_img_tif)
    label=read_images(add_lab_tif)
    print(label.shape)
    lab_gray=cv2.imread(add_lab_bmp,cv2.IMREAD_GRAYSCALE)
    print(lab_gray.shape)
    # img_account=np.load(add_img_act)

    ##顺序分割部分
    index=0
    account=0
    for i in range(26):
        for j in range(28):
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],seg_result_path+'NIR\\' + 'tif5_ori_'+str(index))
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],seg_result_path+'RGB\\'+'tif5_ori_'+str(index))
            save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],seg_result_path+'LABEL\\'+'tif5_ori_'+str(index))
            index+=1
            account+=1

    ##随机分割部分
    index=0
    while account < img_each:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        if np.random.random() < aug_rate:
            aug_img, aug_lab=data_augment(ran_image,ran_lab)
        else:
            aug_img=ran_image
            aug_lab=ran_lab
        save_to_bmp(aug_img[:,:,0],seg_result_path+'NIR\\' + 'tif5_ran_'+str(index))
        save_to_bmp(aug_img[:,:,1:4],seg_result_path+'RGB\\'+'tif5_ran_'+str(index))
        save_to_bmp(aug_lab,seg_result_path+'LABEL\\'+'tif5_ran_'+str(index))
        index+=1
        account+=1

    ##特别分割部分
    account_2=20
    account_3=20
    account_5=100
    account_11=100
    account_13=100
    account_15=100
    index=0
    time=0
    while account_2!=0 or account_3!=0 or account_5!=0 or account_11!=0 or account_13!=0 or account_15!=0 :
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        ran_gray=lab_gray[random_height: random_height + im_h, random_width: random_width + im_w]

        flag=False
        if cacul_units(ran_gray,colormap[1]) and account_2>0:
            account_2-=1
            flag=True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif5_spe_' + str(index))
            # index+=1
        elif cacul_units(ran_gray,colormap[2]) and account_3>0:
            account_3-=1
            flag = True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif5_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[4]) and account_5>0:
            account_5-=1
            flag = True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif5_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[10])and account_11>0:
            account_11-=1
            flag = True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif5_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[12]) and account_13>0:
            account_13-=1
            flag = True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif5_spe_' + str(index))
            # index += 1
        elif cacul_units(ran_gray,colormap[14]) and account_15>0:
            account_15-=1
            flag = True
            # if np.random.random() < aug_rate:
            #     aug_img, aug_lab = data_augment(ran_image, ran_lab)
            # else:
            #     aug_img = ran_image
            #     aug_lab = ran_lab
            # save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif5_spe_' + str(index))
            # save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif5_spe_' + str(index))
            # index += 1

        if flag:
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif2_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif2_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif2_spe_' + str(index))
            index+=1
            account_2 = 20
            account_3 = 20
            account_5 = 100
            account_11 = 100
            account_13 = 100
            account_15 = 100
            print('account_2:',account_2,
                  '  account_3:',account_3,
                  '  account_5:',account_5,
                  '  account_11:',account_11,
                  '  account_13:',account_13,
                  '  account_15:',account_15)

        time += 1
        if time >= 1000000:
            break
    print(time)

def special_segment_img6():
    '''
    分割遥感图像
    GF2_PMS1__20160827_L1A0001793003-MSS1
    选择特定的目标的图像
    '''
    name='GF2_PMS1__20160827_L1A0001793003-MSS1'

    lab_bmp_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
    train_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
    account_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\label_account\\'

    seg_result_path='E:\\competition\\segment\\data\\sepcial_seg\\'

    add_img_tif=train_path+name+'.tif'
    add_lab_tif=train_path+name+'_label.tif'
    add_lab_bmp=lab_bmp_path+name+'_label.bmp'
    # add_img_act=account_path+name+'_label.txt'

    image=read_images(add_img_tif)
    label=read_images(add_lab_tif)
    print(label.shape)
    lab_gray=cv2.imread(add_lab_bmp,cv2.IMREAD_GRAYSCALE)
    print(lab_gray.shape)
    # img_account=np.load(add_img_act)

    ##顺序分割部分
    index=0
    account=0
    for i in range(26):
        for j in range(28):
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],seg_result_path+'NIR\\' + 'tif6_ori_'+str(index))
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],seg_result_path+'RGB\\'+'tif6_ori_'+str(index))
            save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],seg_result_path+'LABEL\\'+'tif6_ori_'+str(index))
            index+=1
            account+=1

    ##随机分割部分
    index=0
    while account < img_each:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        if np.random.random() < aug_rate:
            aug_img, aug_lab=data_augment(ran_image,ran_lab)
        else:
            aug_img=ran_image
            aug_lab=ran_lab
        save_to_bmp(aug_img[:,:,0],seg_result_path+'NIR\\' + 'tif6_ran_'+str(index))
        save_to_bmp(aug_img[:,:,1:4],seg_result_path+'RGB\\'+'tif6_ran_'+str(index))
        save_to_bmp(aug_lab,seg_result_path+'LABEL\\'+'tif6_ran_'+str(index))
        index+=1
        account+=1

    ##特别分割部分
    account_2=20
    account_5=100
    account_7=400
    account_10=20
    account_11=100
    account_12=20
    account_13=100
    account_14=100
    account_15=100
    account_16=100
    index=0
    time=0
    while account_2!=0 or account_5!=0 or account_10!=0 or account_11!=0 or account_12!=0 or account_13!=0 or account_14!=0 or account_15!=0 or account_16!=0:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        ran_gray=lab_gray[random_height: random_height + im_h, random_width: random_width + im_w]

        if cacul_units(ran_gray,colormap[1]) and account_2>0:
            account_2-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index+=1
        elif cacul_units(ran_gray,colormap[4]) and account_5>0:
            account_5-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[6]) and account_7>0:
            account_7-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[9]) and account_10>0:
            account_10-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[10]) and account_11>0:
            account_11-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[11]) and account_12>0:
            account_12-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[12]) and account_13>0:
            account_13-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[13]) and account_14>0:
            account_14-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[14]) and account_15>0:
            account_15-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[15]) and account_16>0:
            account_16=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif6_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif6_spe_' + str(index))
            index += 1

        time += 1
        if time >= 1000000:
            break
    print(time)

def special_segment_img7():
    '''
    分割遥感图像
    GF2_PMS2__20160225_L1A0001433318-MSS2
    选择特定的目标的图像
    '''
    name='GF2_PMS2__20160225_L1A0001433318-MSS2'

    lab_bmp_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
    train_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
    account_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\label_account\\'

    seg_result_path='E:\\competition\\segment\\data\\sepcial_seg\\'

    add_img_tif=train_path+name+'.tif'
    add_lab_tif=train_path+name+'_label.tif'
    add_lab_bmp=lab_bmp_path+name+'_label.bmp'
    # add_img_act=account_path+name+'_label.txt'

    image=read_images(add_img_tif)
    label=read_images(add_lab_tif)
    print(label.shape)
    lab_gray=cv2.imread(add_lab_bmp,cv2.IMREAD_GRAYSCALE)
    print(lab_gray.shape)
    # img_account=np.load(add_img_act)

    ##顺序分割部分
    index=0
    account=0
    for i in range(26):
        for j in range(28):
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],seg_result_path+'NIR\\' + 'tif7_ori_'+str(index))
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],seg_result_path+'RGB\\'+'tif7_ori_'+str(index))
            save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],seg_result_path+'LABEL\\'+'tif7_ori_'+str(index))
            index+=1
            account+=1

    ##随机分割部分
    index=0
    while account < img_each:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        if np.random.random() < aug_rate:
            aug_img, aug_lab=data_augment(ran_image,ran_lab)
        else:
            aug_img=ran_image
            aug_lab=ran_lab
        save_to_bmp(aug_img[:,:,0],seg_result_path+'NIR\\' + 'tif7_ran_'+str(index))
        save_to_bmp(aug_img[:,:,1:4],seg_result_path+'RGB\\'+'tif7_ran_'+str(index))
        save_to_bmp(aug_lab,seg_result_path+'LABEL\\'+'tif7_ran_'+str(index))
        index+=1
        account+=1

    ##特别分割部分
    account_3=20
    account_4=20
    account_5=100
    account_6=100
    account_11=100
    account_12=100
    account_13=100
    account_16=100
    index=0
    time=0
    while account_3!=0 or account_4!=0 or account_5!=0 or account_6!=0 or account_11!=0 or account_12!=0 or account_13!=0 or account_16!=0:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        ran_gray=lab_gray[random_height: random_height + im_h, random_width: random_width + im_w]

        if cacul_units(ran_gray,colormap[2]) and account_3>0:
            account_3-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif7_spe_' + str(index))
            index+=1
        elif cacul_units(ran_gray,colormap[3]) and account_4>0:
            account_4-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif7_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[4]) and account_5>0:
            account_5-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif7_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[5]) and account_6>0:
            account_6-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif7_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[10]) and account_11>0:
            account_11-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif7_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[11]) and account_12>0:
            account_12-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif7_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[12]) and account_13>0:
            account_13-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif7_spe_' + str(index))
            index += 1
        elif cacul_units(ran_gray,colormap[15]) and account_16>0:
            account_16-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif7_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif7_spe_' + str(index))
            index += 1

        time += 1
        if time >= 1000000:
            break
    print(time)

def special_segment_img8():
    '''
    分割遥感图像
    GF2_PMS2__20160510_L1A0001573999-MSS2
    选择特定的目标的图像
    '''
    name='GF2_PMS2__20160510_L1A0001573999-MSS2'

    lab_bmp_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
    train_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
    account_path='E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\label_account\\'

    seg_result_path='E:\\competition\\segment\\data\\sepcial_seg\\'

    add_img_tif=train_path+name+'.tif'
    add_lab_tif=train_path+name+'_label.tif'
    add_lab_bmp=lab_bmp_path+name+'_label.bmp'
    # add_img_act=account_path+name+'_label.txt'

    image=read_images(add_img_tif)
    label=read_images(add_lab_tif)
    print(label.shape)
    lab_gray=cv2.imread(add_lab_bmp,cv2.IMREAD_GRAYSCALE)
    print(lab_gray.shape)
    # img_account=np.load(add_img_act)

    ##顺序分割部分
    index=0
    account=0
    for i in range(26):
        for j in range(28):
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],seg_result_path+'NIR\\' + 'tif8_ori_'+str(index))
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],seg_result_path+'RGB\\'+'tif8_ori_'+str(index))
            save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],seg_result_path+'LABEL\\'+'tif8_ori_'+str(index))
            index+=1
            account+=1

    ##随机分割部分
    index=0
    while account < img_each:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        if np.random.random() < aug_rate:
            aug_img, aug_lab=data_augment(ran_image,ran_lab)
        else:
            aug_img=ran_image
            aug_lab=ran_lab
        save_to_bmp(aug_img[:,:,0],seg_result_path+'NIR\\' + 'tif8_ran_'+str(index))
        save_to_bmp(aug_img[:,:,1:4],seg_result_path+'RGB\\'+'tif8_ran_'+str(index))
        save_to_bmp(aug_lab,seg_result_path+'LABEL\\'+'tif8_ran_'+str(index))
        index+=1
        account+=1

    ##特别分割部分
    account_3=20
    index=0
    time=0
    while account_3!=0:
        random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
        random_height = random.randint(0, X_height - im_h - 1)
        ran_image= image[random_height: random_height + im_h, random_width: random_width + im_w,:]
        ran_lab= label[random_height: random_height + im_h, random_width: random_width + im_w]
        ran_gray=lab_gray[random_height: random_height + im_h, random_width: random_width + im_w]

        if cacul_units(ran_gray,colormap[2]) and account_3>0:
            account_3-=1
            if np.random.random() < aug_rate:
                aug_img, aug_lab = data_augment(ran_image, ran_lab)
            else:
                aug_img = ran_image
                aug_lab = ran_lab
            save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif8_spe_' + str(index))
            save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif8_spe_' + str(index))
            save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif8_spe_' + str(index))
            index+=1

        time += 1
        if time >= 1000000:
            break
    print(time)

def special_segment_all(order,units=[]):
    namespace=['GF2_PMS1__20150212_L1A0000647768-MSS1',
               'GF2_PMS1__20150902_L1A0001015649-MSS1',
               'GF2_PMS1__20151203_L1A0001217916-MSS1',
               'GF2_PMS1__20160327_L1A0001491417-MSS1',
               'GF2_PMS1__20160816_L1A0001765570-MSS1',
               'GF2_PMS1__20160827_L1A0001793003-MSS1',
               'GF2_PMS2__20160225_L1A0001433318-MSS2',
               'GF2_PMS2__20160510_L1A0001573999-MSS2']
    name=namespace[order]


    lab_bmp_path = 'E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train_label_bmp\\'
    train_path = 'E:\\competition\\segment\\data\\rssrai2019_semantic_segmentation\\train\\train\\'
    seg_result_path = 'E:\\competition\\segment\\data\\special_seg\\'

    add_img_tif = train_path + name + '.tif'
    add_lab_tif = train_path + name + '_label.tif'
    add_lab_bmp = lab_bmp_path + name + '_label.bmp'

    image = read_images(add_img_tif)
    label = read_images(add_lab_tif)
    print(label.shape)
    lab_gray = cv2.imread(add_lab_bmp, cv2.IMREAD_GRAYSCALE)
    print(lab_gray.shape)

    # ##顺序分割部分
    index = 0
    account = 0
    for i in range(26):
        for j in range(28):
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 0],
                        seg_result_path + 'NIR\\' +  'tif'+str(order)+'_ori_'  + str(index))
            save_to_bmp(image[256 * i:256 + 256 * i, 256 * j:256 * j + 256, 1:4],
                        seg_result_path + 'RGB\\' + 'tif'+str(order)+'_ori_'  + str(index))
            save_to_bmp(label[256 * i:256 + 256 * i, 256 * j:256 * j + 256, :],
                        seg_result_path + 'LABEL\\' + 'tif'+str(order)+'_ori_'  + str(index))
            index += 1
            account += 1
    #
    # ##随机分割部分
    # index = 0
    # while account < img_each:
    #     random_width = random.randint(0, X_width - im_w - 1)  # 生成随机随机坐标
    #     random_height = random.randint(0, X_height - im_h - 1)
    #     ran_image = image[random_height: random_height + im_h, random_width: random_width + im_w, :]
    #     ran_lab = label[random_height: random_height + im_h, random_width: random_width + im_w]
    #     if np.random.random() < aug_rate:
    #         aug_img, aug_lab = data_augment(ran_image, ran_lab)
    #     else:
    #         aug_img = ran_image
    #         aug_lab = ran_lab
    #     save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif'+str(order)+'_ran_' + str(index))
    #     save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' +  'tif'+str(order)+'_ran_' + str(index))
    #     save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' +  'tif'+str(order)+'_ran_'  + str(index))
    #     index += 1
    #     account += 1

    ##特殊分割部分
    # axis_x=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    # axis_y=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    # unit_account=np.array([0,20,20,20,100,100,200,100,20,20,100,100,100,100,100,100])
    # step=1000
    # for i in range((6800-256)//step):
    #     for j in tqdm(range((7200-256)//step)):
    #         gray = lab_gray[i*step: i*step + im_h, j*step: j*step + im_w]
    #
    #         for k in range(len(units)):
    #             if cacul_units(gray,colormap[units[k]]):
    #                 axis_x[units[k]].append(i)
    #                 axis_y[units[k]].append(j)
    #
    # print(len(axis_x[0]),len(axis_x[1]),len(axis_x[2]),len(axis_x[3]),
    #       len(axis_x[4]),len(axis_x[5]),len(axis_x[6]),len(axis_x[7]),
    #       len(axis_x[8]),len(axis_x[9]),len(axis_x[10]),len(axis_x[11]),
    #       len(axis_x[12]),len(axis_x[13]),len(axis_x[14]),len(axis_x[15]))
    #
    # print('1 step is done')#--------------------------------------------------------------------------------------------
    #
    # index=0
    # for k in range(len(units)):
    #     x=axis_x[units[k]]
    #     y=axis_y[units[k]]
    #     random.shuffle(x)
    #     random.shuffle(y)
    #
    #     if len(x)>unit_account[units[k]]:
    #         for i in range(units[k]):
    #             spe_image=image[x[i],x[i]+im_h,y[i],y[i]+im_w,:]
    #             spe_label=label[x[i],x[i]+im_h,y[i],y[i]+im_w,:]
    #
    #             if np.random.random() < aug_rate:
    #                 aug_img, aug_lab = data_augment(spe_image, spe_label)
    #             else:
    #                 aug_img = ran_image
    #                 aug_lab = ran_lab
    #             save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif'+str(order)+'_ran_' + str(index))
    #             save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif'+str(order)+'_ran_' + str(index))
    #             save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif'+str(order)+'_ran_' + str(index))
    #             index += 1
    #     else:
    #         for i in range(len(x)):
    #             spe_image=image[x[i],x[i]+im_h,y[i],y[i]+im_w,:]
    #             spe_label=label[x[i],x[i]+im_h,y[i],y[i]+im_w,:]
    #
    #             if np.random.random() < aug_rate:
    #                 aug_img, aug_lab = data_augment(spe_image, spe_label)
    #             else:
    #                 aug_img = ran_image
    #                 aug_lab = ran_lab
    #             save_to_bmp(aug_img[:, :, 0], seg_result_path + 'NIR\\' + 'tif'+str(order)+'_ran_' + str(index))
    #             save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif'+str(order)+'_ran_' + str(index))
    #             save_to_bmp(aug_lab, seg_result_path + 'LABEL\\' + 'tif'+str(order)+'_ran_' + str(index))
    #             index += 1

    axis = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    unit_account = np.array([0, 20, 20, 20, 100, 100, 200, 100, 20, 20, 100, 100, 100, 100, 100, 100])
    step = 256
    for i in range((6800 - 256) // step):
        for j in tqdm(range((7200 - 256) // step)):
            gray = lab_gray[i * step: i * step + im_h, j * step: j * step + im_w]

            for k in range(len(units)):
                if cacul_units(gray, colormap[units[k]]):
                    axis[units[k]].append((i,j))

    print(len(axis[0]), len(axis[1]), len(axis[2]), len(axis[3]),
          len(axis[4]), len(axis[5]), len(axis[6]), len(axis[7]),
          len(axis[8]), len(axis[9]), len(axis[10]), len(axis[11]),
          len(axis[12]), len(axis[13]), len(axis[14]), len(axis[15]))

    index = 0
    for k in range(len(units)):
        x = axis[units[k]]
        random.shuffle(x)

        if len(x) > unit_account[units[k]]:
            for i in range(units[k]):
                print(x[i][0],x[i][1])
                spe_image = image[x[i][0]*step:x[i][0]*step + im_h, x[i][1]*step:x[i][1]*step + im_w, :]
                spe_label = label[x[i][0]*step:x[i][0]*step + im_h, x[i][1]*step:x[i][1]*step + im_w, :]

                if np.random.random() < aug_rate:
                    aug_img, aug_lab = data_augment(spe_image, spe_label)
                else:
                    aug_img = spe_image
                    aug_lab = spe_label
                save_to_bmp(aug_img[:, :, 0],   seg_result_path + 'NIR\\' + 'tif' + str(order) + '_ran_' + str(index))
                save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif' + str(order) + '_ran_' + str(index))
                save_to_bmp(aug_lab,            seg_result_path + 'LABEL\\' + 'tif' + str(order) + '_ran_' + str(index))
                index += 1
        else:
            for i in range(len(x)):
                print(x[i][0], x[i][1])
                spe_image = image[x[i][0] * step:x[i][0] * step + im_h, x[i][1] * step:x[i][1] * step + im_w, :]
                spe_label = label[x[i][0] * step:x[i][0] * step + im_h, x[i][1] * step:x[i][1] * step + im_w, :]


                if np.random.random() < aug_rate:
                    aug_img, aug_lab = data_augment(spe_image, spe_label)
                else:
                    aug_img = spe_image
                    aug_lab = spe_label
                save_to_bmp(aug_img[:, :, 0],   seg_result_path + 'NIR\\' + 'tif' + str(order) + '_ran_' + str(index))
                save_to_bmp(aug_img[:, :, 1:4], seg_result_path + 'RGB\\' + 'tif' + str(order) + '_ran_' + str(index))
                save_to_bmp(aug_lab,            seg_result_path + 'LABEL\\' + 'tif' + str(order) + '_ran_' + str(index))
                index += 1





def test():
    a=[[1,2,3],[2,4],[2,3,4]]
    b=[[1,2,3],[2,4],[2,3,4]]
    c=[(1,1),(2,2),(3,3)]
    random.shuffle(c)
    print(c,c[0][1])

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # all_tif2bmp()
    # expand_tif_1()
    # account_units()
    # account_all_img()
    # special_segment_img1()
    # special_segment_img2()
    # special_segment_img3()
    # special_segment_img4()
    # special_segment_img5()
    # special_segment_img6()
    # special_segment_img7()
    # special_segment_img8()
    # test()
    special_segment_all(order=0,units=[4-1,5-1,6-1,8-1,10-1,11-1,12-1,13-1,14-1,15-1])