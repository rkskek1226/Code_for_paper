import tensorflow as tf
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()

print(type(train_x))
print(type(train_x[0]))
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

print(train_x[0].shape)

model = tf.keras.models.Sequential()   # VGG16
model.add(tf.keras.layers.Resizing(224, 224))

model.add(Conv2D(64, (3, 3), input_shape=(224, 224), activation="relu", padding="same", strides=1))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=1))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=1))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=1))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=1))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=1))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=1))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=1))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=1))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=1))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=1))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=1))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same", strides=1))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(100, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")
model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y))

print(model.summary())
print(model.evaluate(test_x, test_y))

