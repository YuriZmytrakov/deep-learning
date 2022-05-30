# James Marcogliese - 501089745, Yuri Zmytrakov - 501074085
# source: https://www.tensorflow.org/tutorials/images/transfer_learning

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.optimizers import rmsprop_v2
from keras.models import Sequential
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import zipfile
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Constants
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3  # RGB
BATCH_SIZE = 20
EPOCHS = 5

cwd = os.getcwd()
zipped_file_location = cwd = os.getcwd() + "\\cats_dogs 2.zip"
unzipped_file_location = os.getcwd() + "\\cats_dogs 2"

if not os.path.isdir(unzipped_file_location):
    # Extract ZIP
    with zipfile.ZipFile(zipped_file_location, 'r') as zip_ref:
        zip_ref.extractall(unzipped_file_location)

# downloading and processing the train test images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    unzipped_file_location + "\\cats_dogs\\train",
    target_size=IMAGE_SIZE,
    batch_size=400,
    class_mode='binary',
    color_mode='rgb')  # 3 channel

test_generator = test_datagen.flow_from_directory(
    unzipped_file_location + "\\cats_dogs\\test",
    target_size=IMAGE_SIZE,
    batch_size=100,
    class_mode='binary')

# Building CNN for accuracy comparison
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=rmsprop_v2.RMSprop(lr=1e-4), metrics=['acc'])

history = model.fit(train_generator, epochs=EPOCHS)

test_loss_cnn, test_acc_cnn = model.evaluate(test_generator)
print("CNN Model test results:")
print(
    f"Test loss: {np.round(test_loss_cnn, 3)}, test accuracy: {np.round(test_acc_cnn, 3)}")
# print(model.summary())

history = model.fit(train_generator, epochs=EPOCHS)


# Q3A printing the summary of the conv_base model
conv_base = VGG16(weights='imagenet', include_top=False,
                  input_shape=(150, 150, 3))
conv_base.summary()

# Assign ImageDataGenerator result to Numpy array
x_train, y_train = next(iter(train_generator))
x_test, y_test = next(iter(test_generator))

print("x_train.shape, y_train.shape, x_test.shape, y_test.shape:")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Converts a class vector to binary class matrix.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# extracting features in pre-trained CNN
train_features_batch = conv_base.predict(x_train)
test_features_batch = conv_base.predict(x_test)

print("train_features_batch.shape, test_features_batch.shape before reshaping:")
print(train_features_batch.shape, test_features_batch.shape)

# reshaping train, test feature batches
train_features_batch = np.reshape(
    train_features_batch, (train_features_batch.shape[0], 4 * 4 * 512))
test_features_batch = np.reshape(
    test_features_batch, (test_features_batch.shape[0], 4 * 4 * 512))

# Q3B printing the features of the images
print("Base model train and test features:")
print(train_features_batch.shape, test_features_batch.shape)

# building a CNN model to be used with pretrained model
model_pretrained = models.Sequential()
model_pretrained.add(layers.Flatten())
model_pretrained.add(layers.Dense(
    256, activation='relu', input_dim=4 * 4 * 512))
model_pretrained.add(layers.Dropout(0.5))
model_pretrained.add(layers.Dense(2, activation='softmax'))
model_pretrained.compile(optimizer=optimizers.RMSprop(
    lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

# adding validation to see the learning history data
history_pretrained = model_pretrained.fit(
    train_features_batch, y_train, epochs=30, batch_size=20, validation_split=0.1)
acc = history_pretrained.history['acc']
val_acc = history_pretrained.history['val_acc']
loss = history_pretrained.history['loss']
val_loss = history_pretrained.history['val_loss']
epochs = range(1, len(acc) + 1)

# plotting the loss and accuracy training performance
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# evaluating the pretraining model model on train and test datasets
train_loss, train_acc = model_pretrained.evaluate(
    train_features_batch, y_train)
test_loss, test_acc = model_pretrained.evaluate(test_features_batch, y_test)
print(f"Train loss: {np.round(train_loss, 3)}, train accuracy: {np.round(train_acc,3)}, test loss {np.round(test_loss,3)}, test accuracy {np.round(test_acc,3)}")
