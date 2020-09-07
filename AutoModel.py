# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:24:18 2019

@author: jp2jj
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


###

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
 
from keras import optimizers
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, BatchNormalization
np.random.seed(111)


train = glob.glob('dataset/train/*.png') # 훈련 이미지
train_cleaned = glob.glob('dataset/original/*.png') #오리지날 이미지
test = glob.glob('dataset/test/*.png') # 테스트이미지

print("number of images training set : ", len(train))
print("number of original images set : ", len(train_cleaned))
print("number of images test set : ", len(test))

epochs = 240 #에폭
batch_size = 8 #배치사이즈

X = [] # 인풋 이미지 리스트 train dataset
X_target = [] # 아웃풋 이미지 리스트 cleaned dataset

for img in train:
    img = load_img(img, target_size=(1024,1024))
    img = img_to_array(img).astype('float32')/255.
    X.append(img)

for img in train_cleaned:
    img = load_img(img, target_size=(1024,1024))
    img = img_to_array(img).astype('float32')/255.
    X_target.append(img)
    
X = np.array(X) # 이미지 데이터 리스트 numpy 배열로 형변환
X_target = np.array(X_target)

print(" Size of X : ", X.shape) # 인풋 이미지
print(" Size of X_target : ", X_target.shape) # 아웃풋 이미지

test_list = []
for img in test:
    img = load_img(img, target_size=(1024,1024))
    img = img_to_array(img).astype('float32')/255.
    test_list.append(img)
    
# 모델
    
def autoencoder():
    input_img = Input(shape=(1024,1024,3), name='image_input')
    
    #### enoder
    x = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    #### decoder
    x = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='sigmoid ', kernel_initializer='he_normal', padding='same')(x)

    # model
    autoencoder = Model(inputs=input_img, outputs=x)

    return autoencoder

# 옵티마이저와 손실함수 정의
    
model = autoencoder()
model.compile(optimizer=optimizers.Adam(), loss='MSE')


# 모델 훈련

hist = model.fit(X, X_target, epochs=epochs, batch_size=batch_size)

predicted_list = []
for img in test_list:
    img = np.reshape(img, (1,1024,1024,3))
    predicted = np.squeeze(model.predict(img, batch_size=1))
    predicted_list.append(predicted)
    
_, ax = plt.subplots(1,2, figsize=(12,9.338))
ax[0].imshow(np.squeeze(test_list[0]), cmap='brg')
ax[1].imshow(np.squeeze(predicted_list[0].astype('float32')),cmap='brg')
plt.show()

plt.plot(hist.history['loss']) #모델의 loss 표시
plt.show()

#save

import imageio
i = 0
for img in predicted_list:
    img = np.reshape(img, (1024,1024,3))
    imageio.imwrite('dataset/test_result/'+str(i)+'.png',img)
    i+=1
