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
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
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

