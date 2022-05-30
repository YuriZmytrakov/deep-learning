# James Marcogliese - 501089745, Yuri Zmytrakov - 501074085

import tensorflow as tf
import keras 
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import optimizers
from sklearn.datasets import load_boston
from  sklearn.model_selection import KFold
import keras.backend as K

dataset = load_boston()
X = dataset['data']
Y = dataset['target']

#normalization of the dataset
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std

#modify q and r values accordingly
q = 1
r_ = 2

# f(yˆ, y) = q * max(0, yˆ−y) + r * max(0, y−yˆ)
def twoMAE(y_true, y_pred):
    a = q * K.maximum(y_pred - y_true, 0)
    b = r_ * K.maximum(y_true - y_pred, 0)
    return a + b

#building the neural network as per question
model = Sequential()
model.add(layers.Dense(128, use_bias = True, input_dim=(X.shape[1]), activation='relu'))
model.add(layers.Dense(64, use_bias = True, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='sgd', loss=twoMAE, metrics=[twoMAE, 'mae'])
model.fit(X, Y, epochs=80, batch_size=16, verbose=0)
lossValue, twomaeValue, mae = model.evaluate(X, Y)
print('Loss: ', lossValue, '2MAE: ', twomaeValue, 'MAE: ', mae)