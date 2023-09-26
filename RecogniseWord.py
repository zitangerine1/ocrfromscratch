import cv2
import numpy as np
import string
import os
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPooling2D, Lambda, Bidirectional
from keras.models import Model
import keras.backend as K

from spellchecker import SpellChecker
from PIL import Image

spell = SpellChecker()
char_list = string.ascii_letters + string.digits

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

blstm_1 = Bidirectional(LSTM(128, return_sequences = True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences = True, dropout = 0.2))(blstm_1)

outputs = Dense(len(char_list) + 1, activation = 'softmax')(blstm_2)

wordModel = Model(inputs, outputs)
wordModel.load_weights('/home/couch/Documents/GitHub/ocrfromscratch-WIP/model/CRNN_model.hdf5')

def recognize_words(line_indicator, word_array, n_lines):
    file = open('/home/couch/Documents/GitHub/ocrfromscratch-WIP/output/recognised.txt', 'w')
    
    line_record = []
    
    for i in range(n_lines):
        line_record.append([])
        
    predictions = wordModel.predict(word_array)
    out = K.get_value(K.ctc_decode(predictions, input_length = np.ones(predictions.shape[0]) * predictions.shape[1], greedy = True)[0][0])
    
    lw_index = 0
    
    for wordindex in out:
        word = []
        
        for char in wordindex: 
            if int(char) != -1:
                word.append(char_list[int(char)])
                
        word = ''.join(word)
        line_record[line_indicator[lw_index]].append(word)
        lw_index += 1
            
    for listindex in range(n_lines): 
        line = ' '.join(line_record[listindex])
        print(line)
        file.writelines(line + '\n')
    
    file.close()
            
    