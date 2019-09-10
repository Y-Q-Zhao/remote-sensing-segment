import cv2
import numpy as np
import tensorflow as tf
# import keras

img=cv2.imread('D:\\IMG_20181229_161920.jpg')
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()

a=tf.constant('hello world !')
sess=tf.Session()
print(sess.run(a))