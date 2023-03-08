# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:23:32 2023

@author: mathi
"""

import numpy as np
import time
import random
from matplotlib import pyplot as plt

def sig(x):
    return 1/(1 + np.exp(-x))
def error(x1,x2):
    return (x1- x2)**2

w = np.array([random.random(), random.random()])
b = np.array([random.random(), random.random()])

def neural_simple(input_arr):
    # a1 = sig(w[0] *input_arr[0] + b[0])
    # output = sig(w[1] * a1 + b[1])
    output = sig(w[1] * input_arr[0] + b[1])
    return output


print(w)
print(b)
N = 10000

error_arr = np.zeros(N)

for i in range(N):
    rand = random.random()
    if rand <= 0.5:
        test_data = np.array([rand, 0])
    else:
        test_data = np.array([rand, 1])
        
    z_1 = w[0] * np.array([rand]) * b[0]    
    a1 = sig(z_1)   
    z_0 = w[1] * a1 + b[1]
    
   
  
    w[1] += a1 * (np.exp(-z_0) /(np.exp(-z_0) +1)**2) *2*(test_data[1] - neural_simple(np.array([rand]))) 
    b[1] += (np.exp(-z_0) /(np.exp(-z_0) +1)**2) *2*(test_data[1] - neural_simple(np.array([rand])))  
    w[0] += a1 * (np.exp(-z_0) /(np.exp(-z_0) +1)**2) *2*(test_data[1] - neural_simple(np.array([rand])))  * w[1]
    # b[0] += (np.exp(-z_1) /(np.exp(-z_1) +1)**2) *2*(test_data[1] - neural_simple(np.array([rand]))) * w[1] * \
    # (np.exp(-z_0) /(np.exp(-z_0) +1)**2)
    # w[1] += (test_data[1] - neural_simple(np.array([rand]))) 
    # b[1] += (test_data[1] - neural_simple(np.array([rand])))  
    # w[0] += (test_data[1] - neural_simple(np.array([rand]))) * w[1] * 0.01
    b[0] += (test_data[1] - neural_simple(np.array([rand]))) *w[1] * 0.01
    
    error_arr[i] = np.abs(test_data[1] - neural_simple(np.array([rand])))

print(w)
print(b)

input_arr = np.array([0.1])
print(neural_simple(input_arr))


for i in range(len(error_arr)- 4):
    error_arr[i+1] = (error_arr[i] + error_arr[i+1] + error_arr[i+2]+error_arr[i+3]+error_arr[i+4])/5
    
plt.plot(error_arr)
























