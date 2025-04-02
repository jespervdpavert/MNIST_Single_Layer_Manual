import torch
from torch.utils.data import Dataset, DataLoader
import os
from keras.datasets import mnist
from torch.nn import ReLU, Softmax
alpha = 7e-2  # 10
stepsize = 1e-3 #1e-3
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = (torch.flatten(torch.from_numpy(train_X).float())/255.0).view(60000, 784,1)
train_Y = (torch.flatten(torch.from_numpy(train_y).float())).view(60000,1)
test_X = (torch.flatten(torch.from_numpy(test_X).float())/255.0).view(10000, 784,1)
test_Y = (torch.flatten(torch.from_numpy(test_y).float())).view(10000,1)
matrix_1 = alpha*torch.randn(392,784)
matrix_2 = alpha*torch.randn(10,392)
bias_1 = alpha*torch.randn(392,1)
bias_2 = alpha*torch.randn(10,1)
cntr = 0
delta = torch.zeros(10,1)
## Train phase
## epochs 
for j in range(3):  # epochs
    samples = [i for i in range(60000)]
    for i in range(60000):
        ind = torch.randint(0,len(samples),(1,))
        sample = samples[ind]
        samples.pop(ind)
        correct = torch.zeros(10,1)
        correct[int(train_Y[sample])] = 1.0  
        res3 = torch.matmul(matrix_1,train_X[sample])+bias_1
        res4 = torch.nn.functional.relu(res3)
        res5 = torch.matmul(matrix_2,res4)+bias_2
        res = torch.softmax(res5, dim=0)
        delta += (res-correct)
        cntr +=1
        if cntr%1==0:
            print(cntr)
        # print(f"Batched finished: {cntr}")
            matrix_2 += -stepsize*delta@res4.T
            bias_2 += -stepsize*(delta.T@torch.eye(10)).T
            relu_mask = torch.diag((res3.flatten()>0).float())
            matrix_1 += -stepsize*(((delta.T@matrix_2)@relu_mask)).T@train_X[sample].T
            bias_1 += -(stepsize*(delta.T@matrix_2)@relu_mask@torch.eye(392)).T
            delta = torch.zeros(10,1)
## Test phase
pos = 0
fal = 0
for i in range(10000):
    correct = torch.zeros(10,1)
    correct[int(test_Y[i])] = 1.0  
    res3 = torch.matmul(matrix_1,test_X[i])+bias_1
    res4 = torch.nn.functional.relu(res3)
    res5 = torch.matmul(matrix_2,res4)+bias_2
    res = torch.softmax(res5, dim=0)  
    mx =  torch.argmax(res)
    if int(test_Y[i]) == mx.item():
        pos += 1
    else:
        fal += 1
    print(f"Predicted was: {mx}")
    print(f"Actual is: {int(test_Y[i])}")
print(pos/(pos+fal))
