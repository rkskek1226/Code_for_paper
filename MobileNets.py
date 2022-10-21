import tensorflow as tf
import keras
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten, Concatenate, GlobalAveragePooling2D, DepthwiseConv2D


def MobileBlock(inputs, kerners1, kerners2, stride):
    x = DepthwiseConv2D(kerners1, (stride, stride), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(kerners2, (1, 1), padding="same")(x)
    return x


(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

inputs = keras.Input(shape=(224, 224, 3))
tmp = Conv2D(32, (3, 3), padding="same", strides=2)(inputs)
mobile_1 = MobileBlock(tmp, 32, 64, 1)
mobile_2 = MobileBlock(mobile_1, 64, 128, 2)
mobile_3 = MobileBlock(mobile_2, 128, 128, 1)
mobile_4 = MobileBlock(mobile_3, 128, 256, 2)
mobile_5 = MobileBlock(mobile_4, 256, 256, 1)
mobile_6 = MobileBlock(mobile_5, 256, 512, 2)
mobile_7a = MobileBlock(mobile_6, 512, 512, 1)
mobile_7b = MobileBlock(mobile_7a, 512, 512, 1)
mobile_7c = MobileBlock(mobile_7b, 512, 512, 1)
mobile_7d = MobileBlock(mobile_7c, 512, 512, 1)
mobile_7e = MobileBlock(mobile_7d, 512, 512, 1)
mobile_8 = MobileBlock(mobile_7e, 512, 1024, 2)
mobile_9 = MobileBlock(mobile_8, 1024, 1024, 2)
gap = GlobalAveragePooling2D()(mobile_9)
flat = Flatten()(gap)
dense = Dense(1000, activation="softmax")(flat)

model = tf.keras.Model(inputs=inputs, outputs=dense)
print(model.summary())


