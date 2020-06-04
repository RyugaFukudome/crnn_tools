# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img, img_to_array
import glob
from sklearn.metrics import accuracy_score
##add module##
from img_proccess import make_timedata as mk 

def time_series_juage():
    folder = ["walk","run","sit","squat"]
    INPUT_IMAGE_SIZE = 32
    CLASS_NUM = 4
    TIME_LENGTH = 5
    test_img = []
    test_label = [0,1,2,3]

    dir = "./test_img/"
    files = glob.glob(dir + "/*.png")
    for i, file in enumerate(files):
        img = load_img(file, target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        array = img_to_array(img)
        test_img.append(array)

    test_img = np.array(test_img)
    test_label = np.array(test_label)

    test_img = test_img.astype('float32')
    test_img = test_img / 255.0
    #時系列画像データ変換
    test_img , test_label = mk(test_img,test_label,TIME_LENGTH)
    # 正解ラベルの形式を変換
    test_label = np_utils.to_categorical(test_label, CLASS_NUM)
    # モデルの読み込み
    model = model_from_json(open('./model/json/crnn.json', 'r').read())
    model.load_weights('./model/h5/crnn.h5')
    # コンパイル
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #判別結果
    result = model.predict(test_img)
    result = np.argmax(result,axis=0)
    result = np.argmax(result)
    print("argmax ",result)
    print('予測結果：', folder[int(result)])

    
if __name__ == '__main__':
    time_series_juage()