import tensorflow as tf
import keras
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten, Concatenate, GlobalAveragePooling2D, DepthwiseConv2D, AveragePooling2D, Add


def fire_module(inputs, s1x1, e1x1, e3x3):
    squeeze = Conv2D(s1x1, (1, 1), padding="same", strides=1)(inputs)
    squeeze = Activation("relu")(squeeze)
    expand1 = Conv2D(e1x1, (1, 1), padding="same", strides=1)(squeeze)
    expand1 = Activation("relu")(expand1)
    expand2 = Conv2D(e3x3, (3, 3), padding="same", strides=1)(squeeze)
    expand2 = Activation("relu")(expand2)
    outputs = Concatenate()([expand1, expand2])

    return outputs


inputs = keras.Input(shape=(224, 224, 3))

conv1 = Conv2D(96, (7, 7), padding="valid", strides=2)(inputs)
max_pool1 = MaxPooling2D((3, 3), padding="same", strides=2)(conv1)
fire2 = fire_module(max_pool1, 16, 64, 64)
fire3 = fire_module(fire2, 16, 64, 64)
fire4 = fire_module(fire3, 32, 128, 128)
max_pool4 = MaxPooling2D((3, 3), padding="valid", strides=2)(fire4)
fire5 = fire_module(max_pool4, 32, 128, 128)
fire6 = fire_module(fire5, 48, 192, 192)
fire7 = fire_module(fire6, 48, 192, 192)
fire8 = fire_module(fire7, 64, 256, 256)
max_pool8 = MaxPooling2D((3, 3), padding="valid", strides=2)(fire8)
fire9 = fire_module(max_pool8, 64, 256, 256)
dropout = Dropout(0.5)(fire9)
conv10 = Conv2D(1000, (1, 1), padding="same", strides=1)(dropout)
avgpool10 = GlobalAveragePooling2D()(conv10)

model = tf.keras.Model(inputs=inputs, outputs=avgpool10)

