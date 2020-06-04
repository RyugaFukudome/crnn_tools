# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential, model_from_json
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.utils import np_utils
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img, img_to_array
##############add###############
import glob

def main():
    # ラベル
    folder = ["bat", "run","srow" ]
    v_image = []
    v_label = []
    INPUT_IMAGE_SIZE = 32
    CLASS_NUM = 0
    dir = "./test_img2"
    files = glob.glob(dir + "/*.png")
    # print(files,".file")
    for i, file in enumerate(files):
        img = load_img(file, target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        array = img_to_array(img)
        v_image.append(array)
        # print("v_image",v_image)

    for i , value in enumerate(folder):
        v_label.append(i)
        CLASS_NUM += 1
        print("CLASS_NUM",CLASS_NUM)

    v_image = np.array(v_image)
    v_label = np.array(v_label)
    
    # imageの画素値をint型からfloat型にする
    v_image = v_image.astype('float32')
    # .astype('float32')は下方の画像データの正規化でしている
    # 画素値を[0～255]⇒[0～1]とする
    v_image = v_image / 255.0
    TIME_LENGTH = 5
    print("v_label",v_label)
    v_image , v_label = make_timedata(v_image,v_label,TIME_LENGTH)

    # print("v_image",v_image)
    # 正解ラベルの形式を変換
    v_label = np_utils.to_categorical(v_label, CLASS_NUM)

    # モデルの読み込み
    model = model_from_json(open('./model/json/test.json', 'r').read())

    # 重みの読み込み
    model.load_weights('./model/h5/test.h5')

    # コンパイル
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    from sklearn.metrics import accuracy_score
    result = model.predict(v_image)
    print("result",result)
    y_pred = np.argmax(result,axis=0)
    y_pred = np.argmax(y_pred)
    print("argmax ",y_pred)
    print('予測結果：', folder[int(y_pred)])
    #score = accuracy_score(v_label,result.round())

    # 10枚（0～9番目）のテスト画像を入力し、予測結果を出力
    # y_pred = model.predict(v_image)
    # 出力をクラスベクトルから整数値に変換
    # y_pred = np.argmax(y_pred, axis=1)
    # 予測結果の表示
    #print('予測結果：', folder[np.exp(y_pred)])


def make_timedata(imglist , lablelist , time_length):
    imglist_result = []
    height , width , channel = imglist.shape[1::]
    # print("画像値",height , width , channel)
    for i in range(len(imglist) - time_length):
        imglist_result.append(imglist[i:i+time_length])

    imglist_result = np.array(imglist_result).reshape(len(imglist_result),time_length,height,width,channel)

    #labellistの処理
    lablelist_result = []
    print("label_list",lablelist)
    for i in range(time_length,len(lablelist)):
        lablelist_result.append(lablelist[i])
    print("labellist_result",lablelist_result)
    lablelist_result = np.array(lablelist_result).reshape(len(lablelist_result))

    #print("imglist_result",imglist_result,"lablelist_result",lablelist_result)
    return imglist_result,lablelist_result

if __name__ == '__main__':
    main()