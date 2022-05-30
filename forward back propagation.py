# James Marcogliese - 501089745, Yuri Zmytrakov - 501074085
# Imports
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.datasets import boston_housing
from sklearn import preprocessing

# Hyper-parameters
H = 64 # num of hidden nodes
K = 1 # num of outputs
(X, Y), (test_data, test_targets) = boston_housing.load_data(test_split=0)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
D = X.shape[1] # dimensions of data

def sigmoid(a):
    if (a>= 0):
        return 1 / (1+math.exp(-a))
    else:
        return math.exp(a) / (1+math.exp(a)) 

def train(x,y,H,K,D,eta,iterations):
    v = np.ones((K,H))
    w = np.ones((H,D)) 

    iteration_losses = []
    for _ in range(iterations):
        delta_v = np.zeros((K,H))
        delta_w = np.zeros((H,D))
        ouputs = []

        for t in range(len(x)): 
            wx = np.dot(w, x[t])
            z_h = list(map(sigmoid,wx))
            out = v[0] * z_h
            out_2 = [sigmoid(i) for i in out]
            o = sum(out_2)
            ouputs.append(o)

            delta_v = eta * np.dot((y[t]-o), z_h) 
            for h in range(H):
                delta_w[h] = eta * np.dot((y[t]-o), v[0,h]) * z_h[h] * (1-z_h[h]) * x[t]
    
            v = v + delta_v   
            w = w + delta_w
        iteration_losses.append(mean_squared_error(y, ouputs))
    return iteration_losses

loss = train(X, Y, H, K, D, eta = 1, iterations = 100)

plt.figure(figsize=(12, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(loss, color = "r")
plt.title('mean_squared_error per iteration')
plt.ylabel('MSE')
plt.xlabel('iteration number')
plt.show()