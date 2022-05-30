# source
# https://www.analyticsvidhya.com/blog/2021/08/predict-the-next-word-of-your-text-using-long-short-term-memory-lstm/

import tensorflow
import numpy as np
import re
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import Reshape, LSTM, Dropout
from keras.layers.core import Dense, Activation
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split

# Source:
# https://stackoverflow.com/questions/52821919/index-word-in-dictionary


def build_word_index(unique_words):
    word_index = {}
    for index, eachword in enumerate(unique_words):
        freq = word_index.get(eachword, None)
        if freq == None:
            word_index[eachword] = index
        else:
            word_index[eachword].append(index)
    return word_index

# source:
# https://github.com/y33-j3T/Coursera-Deep-Learning/blob/master/Natural%20Language%20Processing%20with%20Probabilistic%20Models/Week%204%20-%20Word%20Embeddings%20with%20Neural%20Networks/NLP_C2_W4_lecture_nb_01.ipynb


def word_to_one_hot_vector(word, v):
    one_hot_vector = np.zeros(len(v))
    one_hot_vector[v[word]] = 1
    return one_hot_vector


# Q3A
# reading the file set
with open("1661-0.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

tokenizer = RegexpTokenizer('\w+')
words = tokenizer.tokenize(text)
# print 100, 400 and 1000th elements
print("Returning 100, 400 and 1,000th:", words[100], words[400], words[1000])

# Q3B)
words_x = 6
words_y = 2

unique_words = np.unique(words)
X_sentence = []
y_sentense = []

for index in range(len(words)-(7)):
    X_sentence.append(words[index: index + words_x])
    y_sentense.append(words[index + words_x: index + (words_x + words_y)])

X = np.zeros((len(X_sentence), words_x, len(unique_words)), dtype=bool)
Y = np.zeros((len(X_sentence), words_y, len(unique_words)), dtype=bool)

# Thousands example
print(f"X set 6 words: {X_sentence[999]}")
print(f"Y set 2 words: {y_sentense[999]}")

# Q3C)
word_index = build_word_index(unique_words)

for t in range(len(X_sentence)):
    for i in range(6):
        X[t][i] = word_to_one_hot_vector(X_sentence[t][i], word_index)
    for i in range(2):
        Y[t][i] = word_to_one_hot_vector(y_sentense[t][i], word_index)

model = Sequential()
model.add(Dense(128, activation=Activation('relu'),
          input_shape=(6, len(unique_words))))
model.add(LSTM(128))
# model.add(Dropout(0.2))
model.add(Dense(words_y * len(unique_words)))
model.add(Reshape((words_y, len(unique_words)),
          input_shape=(words_y * len(unique_words), )))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
              'accuracy'])  # tested rmsprop optimizer, it returns lower accuracy rate

# Taking as the sample 25%. Full set accuracy 12%, the learning take 6 mins
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.75, random_state=1)

history = model.fit(X_train, y_train, epochs=5, batch_size=128,
                    validation_split=0.05, shuffle=True)

print('Model summary:')
model.summary()

#uncomment if you would like to see the loss, accuracy metrics
# print("Accuracy:", round(np.max(history.history['accuracy']), 3))
# print("Val accuracy:", round(np.max(history.history['val_accuracy']), 3))

# print("Val loss:", round(np.min(history.history['val_loss']), 3))
# print("Val accuracy:", round(np.max(history.history['val_accuracy']), 3))

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'ro', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')

plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()