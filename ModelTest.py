import cv2
import numpy as np
import matplotlib.pyplot as plt

from BatchPadSeg import pad_img, pad_seg
from SegmentPage import unet

from keras.models import Model
from keras.layers import *

model = unet()
model.load_weights('/home/couch/Documents/GitHub/ocrfromscratch-WIP/model/model.h5')

file_test = '/home/couch/Documents/GitHub/ocrfromscratch-WIP/data2/img/lineA18.jpg'
img = cv2.imread(f'{file_test}', 0)
img = pad_img(img)
ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
img = cv2.resize(img, (512, 512))
img = np.expand_dims(img, axis = -1)
img = img / 255

img = np.expand_dims(img, axis = 0)

pred = model.predict(img)
pred=np.squeeze(np.squeeze(pred,axis=0),axis=-1)
plt.imshow(pred,cmap='gray')
plt.imsave('test_img_mask.JPG',pred)


img = cv2.imread('/home/couch/Documents/GitHub/ocrfromscratch-WIP/data2/mask/lineA18_mask.png', 0) 
cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,img)
ori_img = cv2.imread(f'{file_test}',0)
ori_img = pad_img(ori_img)
(H, W) = ori_img.shape[:2]
(newW, newH) = (512, 512)
rW = W / float(newW)
rH = H / float(newH)
ori_img_copy = np.stack((ori_img,) * 3, axis = -1)

contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(ori_img_copy, (int(x * rW), int(y * rH)), (int((x + w) * rW), int((y + h) * rH)), (255, 0, 0), 1)

cv2.imwrite("output.png", ori_img_copy)