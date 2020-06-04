# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
import numpy as np
import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
##train_test_split・・・学習用データ(train_images,train_labels,)と検証用データ(valid_images,valid_labels)に分割
from PIL import Image
import glob
from keras.preprocessing.image import load_img, img_to_array
from keras.initializers import TruncatedNormal, Constant
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.optimizers import SGD
import json

# モデル構築
#LSTM層を持つCNNの定義

from keras.layers import Input,TimeDistributed,LSTM,Conv2D,MaxPooling2D,Flatten,Dense,Activation,Dropout
from keras.models import Model

#LSTM使用分類ネットワーク
def build_model():
        
    TIME_LENGTH = 5
    IMAGE_SIZE = 32
    CHANNEL = 3
    input_img = Input(shape=(TIME_LENGTH,IMAGE_SIZE,IMAGE_SIZE,CHANNEL))

    x = TimeDistributed(Conv2D(32,(3,3),padding='same'))(input_img)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=2))(x)
    x = TimeDistributed(Conv2D(64,(3,3),padding='same'))(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=2))(x)

    x = TimeDistributed(Flatten())(x)
    x = LSTM(1024,dropout=0.5,return_sequences=False)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(3)(x)
    x = Activation('softmax')(x)
    
    model = Model(input_img,x)

    return model
    
def make_timedata(imglist , lablelist , time_length):
    imglist_result = []
    height , width , channel = imglist.shape[1::]
    # print("画像値",height , width , channel)
    for i in range(len(imglist) - time_length):
        imglist_result.append(imglist[i:i+time_length])

    imglist_result = np.array(imglist_result).reshape(len(imglist_result),time_length,height,width,channel)

    #labellistの処理
    lablelist_result = []
    for i in range(time_length,len(lablelist)):
        lablelist_result.append(lablelist[i])

    lablelist_result = np.array(lablelist_result).reshape(len(lablelist_result))

    #print("imglist_result",imglist_result,"lablelist_result",lablelist_result)
    return imglist_result,lablelist_result


import matplotlib.pyplot as plt
def plot_history (history):
    plt.plot (history.history['acc'])
    plt.title ('model accuracy')
    plt.xlabel ('epoch')  
    plt.ylabel ('accuracy')
    plt.legend (['acc'], loc='lower right')
    plt.show ()  
    plt.plot (history.history['loss'])
    plt.title ('model loss')
    plt.xlabel ('epoch')
    plt.ylabel ('loss')  
    plt.legend (['loss'], loc='lower right')
    plt.show ()

# 参考URL https://woraise.com/2019/01/19/tenclassify/
#https://books.google.co.jp/books?id=XEqrDwAAQBAJ&pg=PA113&lpg=PA113&dq=Keras+CNN%E3%81%ABRNN%E3%81%AE%E5%B1%A4%E3%82%92%E8%BF%BD%E5%8A%A0%E3%81%99%E3%82%8B%E3%80%80%E5%8C%BB%E7%99%82&source=bl&ots=kcKE5fcGlT&sig=ACfU3U2VpfhdSioMtUfUSm1_LC88wXqwSQ&hl=ja&sa=X&ved=2ahUKEwiP_suj3KDpAhVaQd4KHYWdCMsQ6AEwAHoECAoQAQ#v=onepage&q=Keras%20CNN%E3%81%ABRNN%E3%81%AE%E5%B1%A4%E3%82%92%E8%BF%BD%E5%8A%A0%E3%81%99%E3%82%8B%E3%80%80%E5%8C%BB%E7%99%82&f=false
