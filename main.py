# -*- coding: utf-8 -*-
from PIL import Image
import glob
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
##add module##
from img_proccess import make_timedata as mtd
from build_model import build_model as bm

def main():

    # 入力画像サイズ
    INPUT_IMAGE_SIZE = 32
    TIME_LENGTH = 5

    # 使用する訓練画像の入ったフォルダ(ルート)
    TRAIN_PATH = "./data"
    # 使用する訓練画像の各クラスのフォルダ名
    folder = ["walk","run","sit","squat"]
    # CLASS数を取得する
    CLASS_NUM = len(folder)
    print("クラス数 : " + str(CLASS_NUM))
    
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

    # .astype('float32')は下方の画像データの正規化でしている
    v_image = v_image.astype('float32')
    # 画素値を[0～255]⇒[0～1]とする
    v_image = v_image / 255.0
    #時系列データに変換
    v_image , v_label = mtd(v_image,v_label,TIME_LENGTH)

    #正解ラベルの形式を変換
    v_label = np_utils.to_categorical(v_label, CLASS_NUM)

    #学習用データ(train_images,train_labels,)と検証用データ(valid_images,valid_labels)に分割する
    #train_images, valid_images, train_labels, valid_labels = train_test_split(v_image, v_label, test_size=0.10)
    x_train = v_image
    y_train = v_label
    #x_test = valid_images
    #y_test = valid_labels


    # モデル構築
    model = bm()
    # モデルをコンパイル
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # 学習　※学習させるデータが少ないとエラーが発生する　verbose=0にて回避は可能
    history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=2) 
    ##validation_split・・・訓練データの中で検証データとして使う割合
    ##nb_epoch・・・学習を繰り返す回数

    # モデルと重みを保存
    open('./model/json/crnn.json',"w").write(model.to_json())
    model.save_weights('./model/h5/crnn.h5')


    # モデルの性能評価
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0]) # Test loss: 0.683512847137
    # print('Test accuracy:', score[1]) # Test accuracy: 0.7871

if __name__ == '__main__':
    main()

