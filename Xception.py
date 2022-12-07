import tensorflow as tf
import keras
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten, Concatenate, GlobalAveragePooling2D, DepthwiseConv2D, AveragePooling2D, Add, DepthwiseConv2D


def xception_module(inputs, channel_num, activate=True):
    if not activate:
        first = Conv2D(channel_num, (1, 1), strides=1, padding="same")(inputs)
    else:
        first = Activation("relu")(inputs)
        first = Conv2D(channel_num, (1, 1), strides=1, padding="same")(first)

    second = DepthwiseConv2D((3, 3), strides=1, padding="same")(first)
    outputs = BatchNormalization()(second)

    return outputs


inputs = keras.Input(shape=(299, 299, 3))

entry_flow1 = Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
entry_flow1 = Activation("relu")(entry_flow1)
entry_flow2 = Conv2D(64, (3, 3), strides=1, padding="same")(entry_flow1)
entry_flow2 = Activation("relu")(entry_flow2)

tmp1 = Conv2D(128, (1, 1), strides=2, padding="same")(entry_flow2)
entry_flow3_1 = xception_module(entry_flow2, 128, False)
entry_flow3_2 = xception_module(entry_flow3_1, 128, True)
entry_flow3_3 = MaxPooling2D((3, 3), strides=2, padding="same")(entry_flow3_2)
entry_flow3_4 = Add()([tmp1, entry_flow3_3])

tmp2 = Conv2D(256, (1, 1), strides=2, padding="same")(entry_flow3_4)
entry_flow4_1 = xception_module(entry_flow3_4, 256, True)
entry_flow4_2 = xception_module(entry_flow4_1, 256, True)
entry_flow4_3 = MaxPooling2D((3, 3), strides=2, padding="same")(entry_flow4_2)
entry_flow4_4 = Add()([entry_flow4_3, tmp2])

tmp3 = Conv2D(728, (1, 1), strides=2, padding="same")(entry_flow4_4)
entry_flow5_1 = xception_module(entry_flow4_4, 728, True)
entry_flow5_2 = xception_module(entry_flow5_1, 728, True)
entry_flow5_3 = MaxPooling2D((3, 3), strides=2, padding="same")(entry_flow5_2)
entry_flow5_4 = Add()([entry_flow5_3, tmp3])

tmp = Conv2D(728, (1, 1), strides=1, padding="same")(entry_flow5_4)
middle_flow1 = xception_module(entry_flow5_4, 728, True)
middle_flow1 = xception_module(middle_flow1, 728, True)
middle_flow1 = xception_module(middle_flow1, 728, True)
middle_flow1 = Add()([tmp, middle_flow1])

tmp = Conv2D(728, (1, 1), strides=1, padding="same")(middle_flow1)
middle_flow2 = xception_module(middle_flow1, 728, True)
middle_flow2 = xception_module(middle_flow2, 728, True)
middle_flow2 = xception_module(middle_flow2, 728, True)
middle_flow2 = Add()([tmp, middle_flow2])

tmp = Conv2D(728, (1, 1), strides=1, padding="same")(middle_flow2)
middle_flow3 = xception_module(middle_flow2, 728, True)
middle_flow3 = xception_module(middle_flow3, 728, True)
middle_flow3 = xception_module(middle_flow3, 728, True)
middle_flow3 = Add()([tmp, middle_flow3])

tmp = Conv2D(728, (1, 1), strides=1, padding="same")(middle_flow3)
middle_flow4 = xception_module(middle_flow3, 728, True)
middle_flow4 = xception_module(middle_flow4, 728, True)
middle_flow4 = xception_module(middle_flow4, 728, True)
middle_flow4 = Add()([tmp, middle_flow4])

tmp = Conv2D(728, (1, 1), strides=1, padding="same")(middle_flow4)
middle_flow5 = xception_module(middle_flow4, 728, True)
middle_flow5 = xception_module(middle_flow5, 728, True)
middle_flow5 = xception_module(middle_flow5, 728, True)
middle_flow5 = Add()([tmp, middle_flow5])

tmp = Conv2D(728, (1, 1), strides=1, padding="same")(middle_flow5)
middle_flow6 = xception_module(middle_flow5, 728, True)
middle_flow6 = xception_module(middle_flow6, 728, True)
middle_flow6 = xception_module(middle_flow6, 728, True)
middle_flow6 = Add()([tmp, middle_flow6])

tmp = Conv2D(728, (1, 1), strides=1, padding="same")(middle_flow6)
middle_flow7 = xception_module(middle_flow6, 728, True)
middle_flow7 = xception_module(middle_flow7, 728, True)
middle_flow7 = xception_module(middle_flow7, 728, True)
middle_flow7 = Add()([tmp, middle_flow7])

tmp = Conv2D(728, (1, 1), strides=1, padding="same")(middle_flow7)
middle_flow8 = xception_module(middle_flow7, 728, True)
middle_flow8 = xception_module(middle_flow8, 728, True)
middle_flow8 = xception_module(middle_flow8, 728, True)
middle_flow8 = Add()([tmp, middle_flow8])

tmp = Conv2D(1024, (1, 1), strides=2, padding="same")(middle_flow8)
exit_flow1 = xception_module(middle_flow8, 728, True)
exit_flow1 = xception_module(exit_flow1, 1024, True)
exit_flow1 = MaxPooling2D((3, 3), strides=2, padding="same")(exit_flow1)
exit_flow1 = Add()([tmp, exit_flow1])

exit_flow2 = xception_module(exit_flow1, 1536, False)
exit_flow2 = Activation("relu")(exit_flow2)

exit_flow3 = xception_module(exit_flow2, 2048, False)
exit_flow3 = Activation("relu")(exit_flow3)

outputs = GlobalAveragePooling2D()(exit_flow3)


model = tf.keras.Model(inputs=inputs, outputs=outputs)

