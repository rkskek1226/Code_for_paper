import tensorflow as tf
import keras
import keras_cv
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten, Concatenate, GlobalAveragePooling2D, DepthwiseConv2D, AveragePooling2D, Add


def shuffle_block(inputs, kerners, group, stride):
    x = Conv2D(kerners, (1, 1), padding="same", strides=stride, groups=group)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    tmp1 = keras_cv.layers.ChannelShuffle(groups=group, seed=1)
    x = tmp1(x)
    x = DepthwiseConv2D(kerners, (3, 3), padding="same", strides=stride)(x)
    x = Conv2D(kerners, (1, 1), padding="same", strides=stride, groups=group)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if stride == 2:
        tmp2 = AveragePooling2D((3, 3), padding="same", strides=stride)(x)
        outputs = Concatenate()([x, tmp2])
    else:
        outputs = Add()([x, inputs])

    return Activation("relu")(outputs)




(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

inputs = keras.Input(shape=(224, 224, 3))

tmp = Conv2D(24, (3, 3), padding="same", strides=2)(inputs)
tmp = MaxPooling2D((3, 3), padding="same", strides=2)(tmp)

stage_2 = shuffle_block(tmp, 100, 2, 2)
for i in range(3):
    stage_2 = shuffle_block(stage_2, 10, 2, 1)

stage_3 = shuffle_block(stage_2, 200, 2, 2)
for i in range(7):
    stage_3 = shuffle_block(stage_3, 10, 2, 1)

stage_4 = shuffle_block(stage_3, 400, 2, 2)
for i in range(3):
    stage_4 = shuffle_block(stage_4, 10, 2, 1)

gap = GlobalAveragePooling2D()(stage_4)
flat = Flatten()(gap)
dense1000 = Dense(1000, activation="softmax")(flat)

model = tf.keras.Model(inputs=inputs, outputs=dense1000)

