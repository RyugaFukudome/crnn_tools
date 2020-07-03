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

# -------------------------------------------------------------------------------------
#                        初期設定部
# -------------------------------------------------------------------------------------

def main():

# 入力画像サイズ(画像サイズは正方形とする)
    INPUT_IMAGE_SIZE = 32

# 使用する訓練画像の入ったフォルダ(ルート)
    TRAIN_PATH = ".\data"
# 使用する訓練画像の各クラスのフォルダ名
    folder = ["bat", "run","srow" ]
# CLASS数を取得する
    CLASS_NUM = len(folder)
    print("クラス数 : " + str(CLASS_NUM))
    


# -------------------------------------------------------------------------------------
#                        訓練画像入力部
# -------------------------------------------------------------------------------------

# 各フォルダの画像を読み込む
    v_image = []
    v_label = []
    for index, name in enumerate(folder):
        dir = TRAIN_PATH + "/" + name
        files = glob.glob(dir + "/*.png")
        print(dir)
        for i, file in enumerate(files):
            img = load_img(file, target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
            array = img_to_array(img)
            v_image.append(array)
            v_label.append(index)

##.append(array)・・・末尾(最後)に要素arrayを追加
##array = img_to_array(img) ・・・画像をarrayに変換する
##for (インデックス, 要素) in enumerate(s):　・・・sからインデックス,と要素を取り出す

    v_image = np.array(v_image)
    v_label = np.array(v_label)
##Numpy array・・・配列として計算する？
    #print("1_v_lavel",v_label)

# imageの画素値をint型からfloat型にする
    v_image = v_image.astype('float32')
# .astype('float32')は下方の画像データの正規化でしている
# 画素値を[0～255]⇒[0～1]とする
    v_image = v_image / 255.0
    TIME_LENGTH = 5
    v_image , v_label = make_timedata(v_image,v_label,TIME_LENGTH)

# 正解ラベルの形式を変換
    v_label = np_utils.to_categorical(v_label, CLASS_NUM)

# 学習用データ(train_images,train_labels,)と検証用データ(valid_images,valid_labels)に分割する
    # v_image = np.insert(v_image,0,2,axis=1)
    # print(v_image)
    train_images, valid_images, train_labels, valid_labels = train_test_split(v_image, v_label, test_size=0.10)
    x_train = train_images
    y_train = train_labels
    x_test = valid_images
    y_test = valid_labels

    #print("v_image",v_image.shape,"v_lavel",v_label)

    # CNNのモデル構築
    model = build_model()

    # モデルをコンパイル
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # 学習　※学習させるデータが少ないとエラーが発生する　verbose=0にて回避は可能
    history = model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=2)
    #plot_history(history)
##validation_split・・・訓練データの中で検証データとして使う割合
##nb_epoch・・・学習を繰り返す回数

    # モデルと重みを保存
    open('./model/json/test.json',"w").write(model.to_json())
    model.save_weights('./model/h5/test.h5')

    # モデルの詳細表示
    model.summary()

    # モデルの性能評価
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0]) # Test loss: 0.683512847137
    print('Test accuracy:', score[1]) # Test accuracy: 0.7871

    #テスト
    from sklearn.metrics import accuracy_score
    result = model.predict(x_test,verbose=1)
    #print("result",result)
    score = accuracy_score(y_test,result.round())
    print("score accury",str(score))
    # 学習履歴を保存
    # json.dump(history.history, open("history.json", "w"))

if __name__ == '__main__':
    main()

# 参考URL https://woraise.com/2019/01/19/tenclassify/
#https://books.google.co.jp/books?id=XEqrDwAAQBAJ&pg=PA113&lpg=PA113&dq=Keras+CNN%E3%81%ABRNN%E3%81%AE%E5%B1%A4%E3%82%92%E8%BF%BD%E5%8A%A0%E3%81%99%E3%82%8B%E3%80%80%E5%8C%BB%E7%99%82&source=bl&ots=kcKE5fcGlT&sig=ACfU3U2VpfhdSioMtUfUSm1_LC88wXqwSQ&hl=ja&sa=X&ved=2ahUKEwiP_suj3KDpAhVaQd4KHYWdCMsQ6AEwAHoECAoQAQ#v=onepage&q=Keras%20CNN%E3%81%ABRNN%E3%81%AE%E5%B1%A4%E3%82%92%E8%BF%BD%E5%8A%A0%E3%81%99%E3%82%8B%E3%80%80%E5%8C%BB%E7%99%82&f=false
