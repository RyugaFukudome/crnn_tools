# -*- coding: utf-8 -*-

import numpy as np

def make_timedata(img_list , label_list , time_length):
   ##時系列画像リストの処理##
    img_list_result = []
    # 画像値,height,width,channel
    height , width , channel = img_list.shape[1::]
    for i in range(len(img_list) - time_length):
        img_list_result.append(img_list[i:i+time_length])

    img_list_result = np.array(img_list_result).reshape(len(img_list_result),time_length,height,width,channel)

    ##ラベルリストの処理##
    label_list_result = []
    for i in range(time_length,len(label_list)):
        label_list_result.append(label_list[i])

    label_list_result = np.array(label_list_result).reshape(len(label_list_result))

    return img_list_result,label_list_result