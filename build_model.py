# -*- coding: utf-8 -*-
from keras.models import Model
from keras.models import Sequential
from keras.initializers import TruncatedNormal, Constant
from keras.layers import Input,TimeDistributed,LSTM,Conv2D,MaxPooling2D,Flatten,Dense,Activation,Dropout
import json
import matplotlib.pyplot as plt

#LSTM層を持つCNNの定義
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
    x = Dense(4)(x)
    x = Activation('softmax')(x)
    
    model = Model(input_img,x)
    model.summary()

    return model


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

    # 学習履歴を保存
    json.dump(history.history, open("history.json", "w"))
