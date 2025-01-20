# This code is a model of a single layer neural network. Only non linearity is the softmax function. 
# Sub 90 percent accuracy

import torch
import tensorflow
from tensorflow import keras
from keras.datasets import mnist

A = 1e-9*torch.randn(10, 28*28)
L1  = 60000
correct = 0
false = 0
L2 = 10000
class_amount = 10
step_size = 1e-7

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_y).float()
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_y).float()

train_X = torch.reshape(train_X, [60000,28*28])
train_Y = torch.reshape(train_Y, [60000,1])
test_X = torch.reshape(test_X, [10000,28*28])
test_Y = torch.reshape(test_Y, [10000,1])

for i in range(L1):
    output = torch.zeros(class_amount,1) 
    output[train_y[i]] = 1
    input_vec = torch.zeros(28*28,1)
    input_vec = train_X[i][:].reshape(-1,1)
    res1 = A @ input_vec
    res = torch.exp(res1)/torch.sum(torch.exp(res1))
    delta = res-output  
    grad = torch.matmul(delta, input_vec.T)
    A -= step_size*grad         
     
for i in range(L2):
    output = torch.zeros(class_amount,1) 
    output[test_y[i]] = 1
    input_vec = torch.zeros(28*28,1)
    input_vec = test_X[i][:].reshape(-1,1)
    res1 = A @ input_vec
    res = torch.exp(res1)/torch.sum(torch.exp(res1))
    mx = 1 
    for k in range(10):
        if res[k]>res[mx]:
            mx = k
    if test_y[i] == mx: 
        correct += 1
    else: 
        false += 1
        
print('Correct percentage: ')
print(correct/(false+correct))



















