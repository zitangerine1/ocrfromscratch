import numpy as np
import cv2
import os
import math

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from spellchecker import SpellChecker
from PIL import Image

from BatchPadSeg import pad_img, roundup
from SegmentPage import model

model.load_weights('/home/couch/Documents/GitHub/ocrfromscratch-WIP/model/word_seg_model.h5')

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

def sort_word(wordlist):
    wordlist.sort(key = lambda x : x[0])
    return wordlist

def segment_to_words(line_img, index):
    img = pad_img(line_img)
    original_img = img.copy()
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis = -1)
    img = img / 255
    img = np.expand_dims(img, axis = 0)
    
    seg_pred = model.predict(img)
    seg_pred = np.squeeze(np.squeeze(seg_pred, axis = 0), axis = -1)
    seg_pred = cv2.normalize(src = seg_pred, dst = None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    cv2.threshold(seg_pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, seg_pred)
    
    contours, hier = cv2.findContours(seg_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    (H, W) = original_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)
    
    coordinates = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        coordinates.append((int(x * rW), int(y * rH), int((x + w) * rW), int((y + h) * rH)))
    
    coordinates = sort_word(coordinates)
    word_count = 0
    
    word_array = []
    line_indicator = []
    
    for (x1, y1, x2, y2) in coordinates:
        word_img = original_img[y1:y2, x1:x2]
        word_img = preprocess(word_img, (128, 32))
        word_img = np.expand_dims(word_img, axis = -1)
        word_array.append(word_img)
        line_indicator.append(index)
        
    return line_indicator, word_array