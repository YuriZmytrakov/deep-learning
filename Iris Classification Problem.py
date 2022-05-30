# James Marcogliese - 501089745, Yuri Zmytrakov - 501074085

import tensorflow as tf
import keras 
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import optimizers
from sklearn.datasets import load_iris
from  sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#loading the dataset
dataset = load_iris()
X = dataset['data']
Y = dataset['target']

#normalizing dataset
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std
Y = pd.get_dummies(Y)

#split train 70% test 30%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

#defining the optimizers
optimizer_list=['rmsprop','adam', 'sgd']

#creating the Kfold object
kfold = KFold(n_splits=5, shuffle=True, random_state = 1)

for opt in optimizer_list:
    def create_model_set(opt):
        def create_model():
            model = Sequential()
            model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(3, activation='softmax'))
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            
            return model
        return create_model
    my_model= create_model_set(opt)()
    history=my_model.fit(X_train, y_train, epochs=80,verbose=0)

model_acc={}
for opt in optimizer_list:
    model_estimator = KerasClassifier(build_fn=create_model_set(opt), epochs=80, verbose=0)
    kfold = KFold(n_splits = 5, shuffle = True, random_state=1)
    acc = cross_val_score(model_estimator, X_train, y_train, cv=kfold)
    model_acc[opt]=np.average(acc)

#Average optimizer accuracies
print('Accuracy values per optimizers:')
print(model_acc)

winner = max(model_acc, key=model_acc.get)
print('Winner optimizer: ', winner)

#initializing the model with the winning optimizer
model = Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer=winner, loss='categorical_crossentropy', metrics=['accuracy'])

#part 3C______________________________________________________________________________
avg_accuracy = {}
for i in range(X_train.shape[1]):
    temp_avg = []
    set_of_values = {-3,-2,-1,1,2,3}
    for _ in range(len(set_of_values)):
        # v from set {-3,-2,-1,1,2,3}
        v = set_of_values.pop()

        #Temporaty X set for feature replacement
        X_temp = X_train

        #replacing the columnvalue with a value from the set {-3,-2,-1,1,2,3}
        X_temp[:,i] = np.ones(X_temp.shape[0]) * v    

        # my_model = model()
        my_model.fit(X_temp, y_train, epochs=80, verbose=0)
        history = my_model.evaluate(X_test, y_test)

        #storing the average accuracy per column
        temp_avg.append(np.average(history[1]))
        
    print('Accuracy feature #', i, ' = ', np.average(temp_avg))
    avg_accuracy[i] = np.average(temp_avg)

sort_orders = sorted(avg_accuracy.items(), key=lambda x: x[1])
col_priority_dict = {('Feature # ' + str(i[0])): i[1] for i in sort_orders}
print("Features sorted by importance")
print(col_priority_dict)

#Based on the output, the feature 3 is the most important, because once we replace this feature with dummie values, 
#the accuracy drops significantly comparing to other features. Also, we observe a drop of accuracy on the test data, 
# because this is new data for this model, and generalization is usually worse. Also dummy values increase the biase 
# and cause drop of accuracy.