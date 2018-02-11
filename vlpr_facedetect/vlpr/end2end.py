#coding=utf-8
from keras import backend as K
from keras.models import load_model
from keras.layers import *
import numpy as np
import random
import string
import time
import cv2
import end2end_model as model


chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z","港","学","使","警","澳","挂","军","北","南","广","沈","兰","成","济","海","民","航","空"
             ];
pred_model = model.construct_model("./model/ocr_plate_all_w_rnn_2.h5",)


def fastdecode(y_pred):
    results = ""
    confidence = 0.0
    table_pred = y_pred.reshape(-1, len(chars)+1)
    #print("fastdecode table_pred:",table_pred)
    res = table_pred.argmax(axis=1)
    print("fastdecode res:",res)
    for i,one in enumerate(res):
        if one<len(chars) and (i==0 or (one!=res[i-1])):
            print("fastdecode  i:",i,"chars[",one,"]:", chars[one])
            results+= chars[one]
            confidence+=table_pred[i][one]
    confidence/= len(results)
    return results,confidence

def recognizeOne(src):
    x_tempx = src
    x_temp = cv2.resize(x_tempx,( 160,40))
    x_temp = x_temp.transpose(1, 0, 2)
    t0 = time.time()
    y_pred = pred_model.predict(np.array([x_temp]))
    #print("recognizeOne y_pred:",y_pred)
    y_pred = y_pred[:,2:,:]
    #print("recognizeOne y_pred:",y_pred)
    return fastdecode(y_pred)
#
#
# import os
#
# path = "cache/finemapping"
# for filename in os.listdir(path):
#     if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
#         x = os.path.join(path,filename)
#         recognizeOne(x)
#         # print time.time() - t0
#
#         # cv2.imshow("x",x)
#         # cv2.waitKey()
