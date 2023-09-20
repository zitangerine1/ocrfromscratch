from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

image_list = os.listdir('/home/couch/code/ocr/data/PageSegData/PageImg')
image_list = [filename.split('.')[0] for filename in image_list]

def get_seg_img(img, n_classes):
    seg_labels = np.zeros((512, 512, n_classes))
    img = cv2.resize(img, (512, 512))
    img = img[:, :, 0]
    cl_list = [0, 24]
    seg_labels[:, :, 0] = (img != 0).astype(int)
    return seg_labels

def batch_generator(file_list, batch_size, n_classes):
    while True:
        x = []
        y = []
        
        for i in range(batch_size):
            fn = random.choice(file_list)
            img = cv2.imread(f'/home/couch/code/ocr/data/PageSegData/PageImg/{fn}.JPG', 0)
            ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
            # Inverse binary thresholding
            img = cv2.resize(img, (512, 512))
            img = np.expand_dims(img, axis = -1)
            img = img / 255
            
            seg = cv2.imread(f'/home/couch/code/ocr/data/PageSegData/PageSeg/{fn}_mask.png', 1)
            seg = get_seg_img(seg, n_classes)
            
            x.append(img)
            y.append(seg)
        yield np.array(x), np.array(y)

def unet(pretrained_weights = None, input_size = (512, 512, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(inputs, conv10)
    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        
    return model
        
model = unet()
model.save('model.h5')
# model.summary()

from keras.callbacks import ModelCheckpoint

random.shuffle(image_list)
file_train = image_list[0:int(len(image_list) * 0.8)]
file_test = image_list[int(len(image_list) * 0.8):]

mc = ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=1)
model.fit_generator(batch_generator(file_train, 2, 2), steps_per_epoch=1000, epochs=3, callbacks=[mc], validation_data=batch_generator(file_test, 2, 2), validation_steps=400, shuffle=1)
