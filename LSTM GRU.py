# James Marcogliese - 501089745, Yuri Zmytrakov - 501074085

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Bidirectional, Embedding, SpatialDropout1D, GlobalMaxPooling1D, Dropout, Flatten, Conv1D, SpatialDropout1D, MaxPooling1D, Activation, RepeatVector
from keras import layers
from keras import optimizers
from sklearn.metrics import confusion_matrix
from keras.regularizers import l2
import tensorflow
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Q2B)
# we do not normalize the datasets since they are z-normilized
# LSTM model with COnv1D train accuracy = 96%, test accuracy 78%

# LSTM with conv1d layer and dropout


def model_LSTM_conv1d():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3,
              padding="same", input_shape=(1, 275)))
    model.add(Activation('relu'))
    model.add(LSTM(32, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    print("\n LSTM model:")
    # model.summary()

    return model

# LSTM with dropout train accuracy 98%, test accuracy 72%
# def model_LSTM():
#     model = Sequential()
#     model.add(LSTM(120, activation ='relu', input_shape=(1,275)))
#     model.add(Dropout(0.20)) #improved the accuracy
#     model.add(Dense(4, activation = 'softmax'))
#     print("\n LSTM model:")

#     return model

# Accuracy rate was lower comparing to the winning model, train accuracy = 84%, test accuracy 68%
# source: https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
# def model_LSTM_Conv1D():
#     input_layer = tensorflow.keras.layers.Input(shape= (1, 275))
#     conv1 = tensorflow.keras.layers.Conv1D(filters=64, kernel_size=5, padding="same")(input_layer)
#     conv1 = tensorflow.keras.layers.BatchNormalization()(conv1)
#     conv1 = tensorflow.keras.layers.ReLU()(conv1)
#     lstm1 = tensorflow.keras.layers.LSTM(units=32)(conv1)
#     output_layer = tensorflow.keras.layers.Dense(4, activation="softmax")(lstm1)
#     model = tensorflow.keras.models.Model(inputs=input_layer, outputs=output_layer)
#     return model

# LSTM Stacked model received train accuracy = 92% and test accuracy = 69%
# def model_LSTM_Stacked():
#     model = Sequential()
#     model.add(LSTM(200, activation='relu', input_shape=(1,275), return_sequences = True))
#     model.add(LSTM(100, activation='relu', return_sequences = True))
#     model.add(LSTM(50, activation='relu', return_sequences = True))
#     # model_LSTM_Stacked.add(Dropout(0.1)) #had negative effect on accuracy
#     model.add(LSTM(32, activation='relu'))
#     model.add(Dense(4, activation='softmax'))
#     print("\n LSTM_stacked model:")
#     model.summary()
#     return model

# LSTM stacked bidirectional model got low accuracy rate. Train accuracy = 90%, and test accuracy = 63%
# def model_LSTM_stacked_bidirectional():
#     model = Sequential()
#     model.add(Bidirectional(LSTM(128, activation ='relu'), input_shape=(1,275)))
#     model.add(Dropout(0.25)) #dropout improved the accuracy
#     model.add(Dense(36, activation='relu'))
#     model.add(Dense(4, activation = 'softmax'))
#     print("\n LSTM_stacked_bidirectional model:")

#     return model

# This is top performing GRU model to solve this problem, train accuracy = 100%, average test accuracy 78%, and increases once we increase the epochs value.


def model_GRU_Bidirectional():
    model = Sequential()
    model.add(Bidirectional(GRU(128, activation='relu',
              return_sequences=True, input_shape=(1, 275))))
    # model.add(RepeatVector(1, name='Repeat-Vector-Layer')) # Repeat Vector
    model.add(Bidirectional(GRU(64)))
    model.add(Dropout(0.25))  # improved the generalization accuracy by 7%
    model.add(Dense(4, activation='softmax'))
    # getGRUBidirectional.add(Dense(4, kernel_regularizer=l2(0.01), activation='softmax')) #the regularization accuracy rate has decreased
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['acc'])

    return model


def getTrained(model, x_train, y_train, x_test, y_test):
    # having tested multiple epochs, batch sized values, we have determined that epochs = 1000, batch_size = 32 yield the highest accuracy rate
    # while taking reasonable amount of time to build the model. We have tested the highest epochs of 10,000, and received accuracy of 84%
    model.fit(x_train, y_train, epochs=300, batch_size=32)
    model.summary()

    y_pred = np.argmax(model.predict(x_train), axis=1)
    y_act = np.argmax((y_train), axis=1)

    confusion_matrix_train = confusion_matrix(y_act, y_pred)
    print('Accuracy train dataset:', accuracy_score(y_act, y_pred))

    y_pred = model.predict(x_test, batch_size=128)
    loss, acc = model.evaluate(x_test, y_train)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_act = np.argmax((y_test), axis=1)
    confusion_matrix_test = confusion_matrix(y_act, y_pred)
    train_Acc = accuracy_score(y_act, y_pred)

    print('Accuracy test dataset:', accuracy_score(y_act, y_pred))

    # accuracy, cofusion matrix of train set, confusion matrix of test set
    return train_Acc, confusion_matrix_train, confusion_matrix_test


# Q1A
data1 = arff.loadarff('Trace_TEST.arff')  # reading the file with test data
data2 = arff.loadarff('Trace_TRAIN.arff')  # reading the file with train data
df_test = pd.DataFrame(data1[0])
df_train = pd.DataFrame(data2[0])

classes = [b'1', b'2', b'3', b'4']
X = range(len(df_train.loc[:, df_train.columns != 'target'].columns))

plt.figure(figsize=(25, 5))
for c in classes:
    df_train_c = df_train[df_train['target'] == c].head(1)
    Y = np.array(df_train_c.loc[:, df_train_c.columns != 'target'])[0]
    plt.plot(X, Y, linewidth=4)
plt.title('Transient Classification Benchmark')
plt.xlabel('Attributes')
plt.ylabel('Values')
plt.legend(['Class 1', 'Class 2', 'Class 3', 'Class 4'])
plt.grid(linewidth=0.4)
plt.show()
# class 4 and class 3 have similar pattern which causes misclassification between these two classes(see confusion matrix)

# preparing the data for models
df_train['target'] = df_train['target'].astype(int)
df_test['target'] = df_test['target'].astype(int)

X_train = np.array(df_train.loc[:, df_train.columns != 'target'])
X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
Y_train = np.array(df_train['target'])

X_test = np.array(df_test.loc[:, df_test.columns != 'target'])
X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
Y_test = np.array(df_test['target'])

onehot_encoder = OneHotEncoder(sparse=False)
Y_train = Y_train.reshape(100, 1)
Y_test = Y_test.reshape(100, 1)
Y_train = onehot_encoder.fit_transform(Y_train)
Y_test = onehot_encoder.fit_transform(Y_test)

output = {}  # for storing the models output
# list_of_models = ['model_LSTM_conv1d, 'model_LSTM','model_LSTM_Stacked','model_LSTM_stacked_bidirectional', 'model_GRU_Bidirectional']
# update the list to build all models see example above
list_of_models = ['model_LSTM_conv1d']

# training the model in the list
for m in list_of_models:
    t_model = globals()[m]()
    t_model.compile(optimizer='adam',
                    loss='categorical_crossentropy', metrics=['acc']) #different optimizers tested, rmsprop returned slighly lower accuracy rate, adam's was the winner
    output[m] = []
    output[m].append(getTrained(t_model, X_train, Y_train, X_test, Y_test))

# finding the winner
winner_model = max(output, key=output.get)
print(f'Winner model -> {winner_model}')
print(f'Accuracy rate: {output[winner_model][0][0]}')

# train confusion matrix
print('Confusion matrix train:')
print(pd.DataFrame(output[winner_model][0][1], columns=[
      'class 1', 'class 2', 'class 3', 'class 4'], index=['class 1', 'class 2', 'class 3', 'class 4']))

# test confusion matrix
print('Confusion matrix test:')
print(pd.DataFrame(output[winner_model][0][2], columns=[
      'class 1', 'class 2', 'class 3', 'class 4'], index=['class 1', 'class 2', 'class 3', 'class 4']))

# Q2C)
# having tested multiple epochs, batch sized values, we have determined that epochs = 1000, batch_size = 32 yield the highest accuracy rate
# while taking reasonable amount of time to build the model. We have tested the highest epochs of 10,000, and received accuracy of 84%
model = model_GRU_Bidirectional()
output = getTrained(model, X_train, Y_train, X_test, Y_test)
print(f'Test accuracy rate: {output[0]}')

# train confusion matrix
print('Train confusion matrix:')
print(pd.DataFrame(output[1], columns=['class 1', 'class 2', 'class 3', 'class 4'], index=[
      'class 1', 'class 2', 'class 3', 'class 4']))

# test confusion matrix
print('Test confusion matrix:')
print(pd.DataFrame(output[2], columns=['class 1', 'class 2', 'class 3', 'class 4'], index=[
      'class 1', 'class 2', 'class 3', 'class 4']))