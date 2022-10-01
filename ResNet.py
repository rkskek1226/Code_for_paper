import tensorflow as tf
import keras
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten, Concatenate, GlobalAveragePooling2D


def ResidualBlock(inputs, kerners, stride):
    x = Conv2D(kerners, (3, 3), padding="same", strides=stride)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(kerners, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if kerners is 64:
        identity = Conv2D(kerners, (1, 1), padding="same")(inputs)
    else:
        identity = Conv2D(kerners, (1, 1), padding="same", strides=2)(inputs)
    x = tf.keras.layers.Add()([x, identity])
    return x


(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

inputs = keras.Input(shape=(224, 224, 3))
tmp = Conv2D(64, (7, 7), padding="same", strides=2)(inputs)
tmp = MaxPooling2D((3, 3), padding="same", strides=2)(tmp)

conv2_x = ResidualBlock(tmp, 64, stride=1)
conv3_x = ResidualBlock(conv2_x, 128, stride=2)
conv4_x = ResidualBlock(conv3_x, 256, stride=2)
conv5_x = ResidualBlock(conv4_x, 512, stride=2)
gap = GlobalAveragePooling2D()(conv5_x)
flat = Flatten()(gap)
dense = Dense(1000, activation="softmax")(flat)


model = tf.keras.Model(inputs=inputs, outputs=dense)
print(model.summary())



