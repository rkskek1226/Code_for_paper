import tensorflow as tf
import keras
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten, Concatenate, GlobalAveragePooling2D


def Inception(inputs, kerners1, kerners2, kerners3, kerners4, kerners5, kerners6, stride):
    x1 = Conv2D(kerners1, (1, 1), padding="same", strides=stride)(inputs)
    x2_1 = Conv2D(kerners2, (1, 1), padding="same", strides=stride)(inputs)
    x2_2 = Conv2D(kerners3, (3, 3), padding="same", strides=stride)(x2_1)
    x3_1 = Conv2D(kerners4, (1, 1), padding="same", strides=stride)(inputs)
    x3_2 = Conv2D(kerners5, (5, 5), padding="same", strides=stride)(x3_1)
    x4_1 = MaxPooling2D((3, 3), padding="same", strides=stride)(inputs)
    x4_2 = Conv2D(kerners6, (1, 1), padding="same", strides=stride)(x4_1)
    outputs = Concatenate()([x1, x2_2, x3_2, x4_2])
    return outputs


(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

inputs = keras.Input(shape=(224, 224, 3))

x = Conv2D(64, (7, 7), activation="relu", padding="same", strides=2)(inputs)
x = MaxPooling2D((3, 3), padding="same", strides=2)(x)
x = BatchNormalization()(x)   # LRN 대체
x = Conv2D(64, (1, 1), activation="relu", padding="same")(x)
x = Conv2D(192, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), padding="same", strides=2)(x)

inception_3a = Inception(x, 64, 96, 128, 16, 32, 32, stride=1)
inception_3b = Inception(inception_3a, 128, 128, 192, 32, 96, 64, stride=1)
tmp1 = MaxPooling2D((3, 3), padding="same", strides=2)(inception_3b)

inception_4a = Inception(tmp1, 192, 96, 208, 16, 48, 64, stride=1)
inception_4b = Inception(inception_4a, 160, 112, 224, 24, 64, 64, stride=1)
inception_4c = Inception(inception_4b, 128, 128, 256, 24, 64, 64, stride=1)
inception_4d = Inception(inception_4c, 112, 144, 288, 32, 64, 64, stride=1)
inception_4e = Inception(inception_4d, 256, 160, 320, 32, 128, 128, stride=1)
tmp2 = MaxPooling2D((3, 3), padding="same", strides=2)(inception_4e)

inception_5a = Inception(tmp2, 256, 160, 320, 32, 128, 128, stride=1)
inception_5b = Inception(inception_5a, 384, 192, 384, 48, 128, 128, stride=1)
gap = GlobalAveragePooling2D()(inception_5b)
tmp3 = Flatten()(gap)
dense = Dense(1000, activation="softmax")(tmp3)


model = tf.keras.Model(inputs=inputs, outputs=dense)
print(model.summary())

