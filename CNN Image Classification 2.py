# James Marcogliese - 501089745, Yuri Zmytrakov - 501074085

# Constants
IMAGE_WIDTH=150
IMAGE_HEIGHT=150
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3 # RGB
BATCH_SIZE=20
EPOCHS=5

import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import zipfile

cwd = os.getcwd()
zipped_file_location = cwd = os.getcwd() + "\\cats_dogs 2.zip"
unzipped_file_location = os.getcwd() + "\\cats_dogs 2"

if not os.path.isdir(unzipped_file_location):
    # Extract ZIP
    with zipfile.ZipFile(zipped_file_location, 'r') as zip_ref:
        zip_ref.extractall(unzipped_file_location)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    unzipped_file_location + "\\cats_dogs\\train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb') # 3 channel

test_generator = test_datagen.flow_from_directory(
    unzipped_file_location + "\\cats_dogs\\test",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary')

from keras.models import Sequential
from keras.optimizers import rmsprop_v2
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
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

model.compile(loss='binary_crossentropy', optimizer=rmsprop_v2.RMSprop(lr=1e-4), metrics=['acc'])

print(model.summary())

history = model.fit(
    train_generator,
    epochs=EPOCHS,
)

acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(
    test_generator
)

print("Test loss:" + str(test_loss) + ". Test accuracy:" + str(test_acc))