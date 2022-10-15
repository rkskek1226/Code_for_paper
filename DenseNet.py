import tensorflow as tf
import keras
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten, Concatenate, GlobalAveragePooling2D, DepthwiseConv2D, AveragePooling2D


def ConvBlock(inputs, kerners):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(kerners, (1, 1), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(kerners, (3, 3), padding="same")(x)

    return Concatenate()([inputs, x])


def DenseBlock(inputs, loop, kerners):
    for i in range(loop):
        inputs = ConvBlock(inputs, kerners)
    return inputs


def TransitionBlock(inputs, kerners):
    x = BatchNormalization()(inputs)
    x = Conv2D(kerners, (1, 1), padding="same")(x)
    x = AveragePooling2D((2, 2), strides=2)(x)
    return x


inputs = keras.Input(shape=(224, 224, 3))
x = Conv2D(10, (7, 7), padding="same", strides=2)(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPooling2D((3, 3), padding="same", strides=2)(x)

d1 = DenseBlock(x, 6, 32)
t1 = TransitionBlock(d1, 32)

d2 = DenseBlock(t1, 12, 32)
t2 = TransitionBlock(d2, 32)

d3 = DenseBlock(t2, 24, 32)
t3 = TransitionBlock(d3, 32)

d4 = DenseBlock(t3, 16, 32)
gap = GlobalAveragePooling2D()(d4)
flat = Flatten()(gap)
dense = Dense(1000, activation="softmax")(flat)

model = tf.keras.Model(inputs=inputs, outputs=dense)
print(model.summary())
