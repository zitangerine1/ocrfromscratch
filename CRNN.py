import fnmatch
import cv2
import numpy as np
import string
import time
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.utils import shuffle

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPooling2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

charlist = string.ascii_letters + string.digits

def encode_to_labels(txt):
    digit_list = []
    
    for index, char in enumerate(txt):
        try:
            digit_list.append(charlist.index(char))
        except:
            print(char)
            
    return digit_list

def find_dominant_color(image):
    width, height = 150, 150
    image = image.resize((width, height),resample = 0)
    pixels = image.getcolors(width * height)
    sorted_pixels = sorted(pixels, key = lambda t : t[0])
    dominant_color = sorted_pixels[-1][1]
    return dominant_color

def preprocess(img, imgSize):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])
        print("Image None")
        
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, newSize, interpolation = cv2.INTER_CUBIC)
    most_frequent_pixel = find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_frequent_pixel
    target[0:newSize[1], 0:newSize[0]] = img
    img = target
    
    return img

training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

max_label_len = 0

annotations = open('/home/couch/Documents/GitHub/ocrfromscratch-WIP/Data-generator-for-CRNN/annotation.txt').readlines()
imagenames = []
texts = []

for i in annotations:
    filename, text = i.split(',')[0], i.split(',')[1].split('\n')[0]
    imagenames.append(filename)
    texts.append(text)
    
c = list(zip(imagenames, texts))
random.shuffle(c)
imagenames, texts = zip(*c)

for i in range(len(imagenames)):
    img = cv2.imread('/home/couch/Documents/GitHub/ocrfromscratch-WIP/Data-generator-for-CRNN/images/' + imagenames[i], 0)
    img = preprocess(img, (128, 32))
    img = np.expand_dims(img , axis = -1)
    img = img / 255
    text = texts[i]
    
    if len(text) > max_label_len:
        max_label_len = len(text)
        
    if i % 10 == 0:
        valid_orig_txt.append(text)
        valid_label_length.append(len(text))
        valid_input_length.append(31)
        valid_img.append(img)
        valid_txt.append(encode_to_labels(text))
        
    else:
        orig_txt.append(text)
        train_label_length.append(len(text))
        train_input_length.append(31)
        training_img.append(img)
        training_txt.append(encode_to_labels(text))
    
    if i == 150000:
        flag = 1
        break
    
    i += 1
    
train_padded_text = pad_sequences(training_txt, maxlen = max_label_len, padding = 'post', value = len(charlist))
valid_padded_text = pad_sequences(valid_txt, maxlen = max_label_len, padding = 'post', value = len(charlist))

inputs = Input(shape = (32, 128, 1))

conv_1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(inputs)
pool_1 = MaxPooling2D(pool_size = (2, 2), strides = 2)(conv_1)

conv_2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(pool_1)
pool_2 = MaxPooling2D(pool_size = (2, 2), strides = 2)(conv_2)

conv_3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(pool_2)

conv_4 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv_3)
pool_4 = MaxPooling2D(pool_size = (2, 1))(conv_4)

conv_5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(pool_4)
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPooling2D(pool_size = (2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2, 2), activation = 'relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

blstm_1 = Bidirectional(LSTM(256, return_sequences = True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(256, return_sequences = True, dropout = 0.2))(blstm_1)

outputs = Dense(len(charlist) + 1, activation = 'softmax')(blstm_2)

CRNNmodel = Model(inputs, outputs)

labels = Input(name = 'labels', shape = [max_label_len], dtype = 'float32')
input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape = (1,), name = 'ctc')([outputs, labels, input_length, label_length])

model = Model(inputs = [inputs, labels, input_length, label_length], outputs = loss_out)

model.compile(loss = {'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
filepath = "/home/couch/Documents/GitHub/ocrfromscratch-WIP/model/bestmodel.h5"
checkpoint = ModelCheckpoint(filepath = filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
callbacks_list = [checkpoint]

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

batch_size = 256
epochs = 15

model.fit(x = [training_img, train_padded_text, train_input_length, train_label_length], y = np.zeros(len(training_img)), batch_size = batch_size, epochs = epochs, validation_data = ([valid_img, valid_padded_text, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]), verbose = 1, callbacks = callbacks_list)
