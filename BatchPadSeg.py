import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import random

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from SegmentPage import get_seg_img

image_list = os.listdir('/home/couch/Documents/GitHub/ocrfromscratch-WIP/data2/img')
image_list = [filename.split('.')[0] for filename in image_list]

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def pad_img(img):
    old_h, old_w = img.shape[0], img.shape[1]
    
    if old_h < 512:
        to_pad = np.ones((512 - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = 512
        
    else:
        to_pad = np.ones((roundup(old_h) - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = roundup(old_h)
        
    if old_w < 512:
        to_pad = np.ones((new_height, 512 - old_w)) * 255
        img = np.concatenate((img, to_pad), axis = 1)
        new_widht = roundup(old_w) - old_w
        
    return img

def pad_seg(img):
    old_h, old_w = img.shape[0], img.shape[1]
    
    if old_h < 512:
        to_pad = np.zeros((512 - old_h, old_w))
        img = np.concatenate((img, to_pad))
        new_height = 512
        
    else:
        to_pad = np.zeros((roundup(old_h) - old_h, old_w))
        img = np.concatenate((img, to_pad))
        new_height = roundup(old_h)
    
    if old_w < 512:
        to_pad = np.zeros((new_height, 512 - old_w))
        img = np.concatenate((img, to_pad), axis = 1)
        new_width = 512
        
    else:
        to_pad = np.zeros((new_height, roundup(old_w) - old_w))
        img = np.concatenate((img, to_pad), axis = 1)
        new_width = roundup(old_w)
        
    return img

# Added pad_img and pad_seg to batch_generator
def batch_generator(filelist, n_classes, batch_size):
    while True:
        x = []
        y = []
        
        for i in range(batch_size):
            fn = random.choice(filelist)
            img = cv2.imread(f'/home/couch/Documents/GitHub/ocrfromscratch-WIP/data2/img/{fn}.JPG', 0)
            img = pad_img(img)
            ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
            
            img = cv2.resize(img, (512, 512))
            img = np.expand_dims(img, axis = -1)
            img = img / 255
            
            seg = cv2.imread(f'/home/couch/Documents/GitHub/ocrfromscratch-WIP/data2/seg/{fn}_mask.png', 1)
            seg = pad_seg(seg)
            seg = cv2.resize(seg, (512, 512))
            seg = np.stack((seg, ) * 3, axis = -1)
            seg = get_seg_img(seg, n_classes)
            
            x.append(img)
            y.append(seg)
            
        yield np.array(x), np.array(y)