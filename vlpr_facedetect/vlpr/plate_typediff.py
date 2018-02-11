#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras import backend as K

K.set_image_dim_ordering('tf')

import cv2
import numpy as np


plateType  = ["蓝牌","单层黄牌","新能源车牌","白色","黑色-港澳"]
def model_keras(nb_classes,img_rows,img_cols,kernel_size,pool_size,conv_size):
    model = Sequential()
    model.add(Conv2D(16, (5, 5),input_shape=(img_rows, img_cols,3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(pool_size, pool_size)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print("model_keras")
    return model

def platetype_predict(image):
    img_rows, img_cols = 9, 34
    kernel_size = 32
    pool_size = 2
    conv_size = 3
    num_platetype = len(plateType)
    
    model = model_keras(num_platetype,img_rows,img_cols,kernel_size,pool_size,conv_size)
    model.load_weights("./model/plate_type.h5")
    model.save("./model/plate_type.h5")

    image = cv2.resize(image, (img_cols, img_rows))
    image = image.astype(np.float) / 255
    res = np.array(model.predict(np.array([image]))[0])
    print("platetype_predict:res:",res)
    return res.argmax()


