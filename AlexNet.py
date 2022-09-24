import tensorflow as tf
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

print(train_x[0].shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Resizing(227, 227))
model.add(Conv2D(96, (11, 11), input_shape=(227, 227, 3), activation="relu", strides=4))
model.add(MaxPooling2D((3, 3), strides=2))
model.add(BatchNormalization())   # 원래는 LRN이지만 BN으로 대체

model.add(Conv2D(256, (5, 5), activation="relu", padding="same", strides=1))
model.add(MaxPooling2D((3, 3), strides=2))
model.add(BatchNormalization())   # LRN이지만 BN으로 대체

model.add(Conv2D(384, (3, 3), activation="relu", padding="same", strides=1))
model.add(Conv2D(384, (3, 3), activation="relu", padding="same", strides=1))
model.add(Conv2D(192, (3, 3), activation="relu", padding="same", strides=1))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(100, activation="softmax"))

print(model.summary())

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
history = model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y))

print(model.evaluate(test_x, test_y))