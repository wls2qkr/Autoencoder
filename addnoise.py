# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:02:06 2019

@author: jp2jj
"""

import os
import numpy as np
import cv2
import glob


path_dir = './dataset/DIV2K'
save_dir = './dataset'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(save_dir+'/train'):
    os.mkdir('./dataset/train')
if not os.path.exists(save_dir+'/original'):
    os.mkdir('./dataset/original')

files = glob.glob(path_dir + '/*.png')

print("ADD Noise")

sig = np.linspace(0,50,len(files))
np.random.shuffle(sig)

for i in range(len(files)):
    image = cv2.imread(files[i])
    image = cv2.resize(image,(1024,1024), interpolation = cv2.INTER_AREA)
    row, col, ch = image.shape
    sigma = sig[i]
    gauss = np.random.normal(0, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    noisy = noisy.astype('uint8')
    cv2.imwrite(os.path.join(save_dir, "train/%04d.png" %i), noisy)
    cv2.imwrite(os.path.join(save_dir, "original/%04d.png" %i), image)
    
print("ADD Noise and resize complete")
    
