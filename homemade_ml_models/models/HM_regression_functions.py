'''ML homemade functions by Jan Stasinski
Linear and Logistic REGRESSIONS'''

import sklearn
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
import warnings


'''Linear regression functions:'''

def gradientDescent(a, b, data, L):
    '''where data has x at index: 0 and y,z,.. at consequtive indices'''
    a_gradient = 0
    b_gradient = 0
    n = len(data[0])

    for i in range(n):
        xi = data[0].iloc[i]
        yi = data[1].iloc[i]
        a_gradient += -(2/n) * xi * (yi - (a * xi + b))
        b_gradient += -(2/n) * (yi - (a * xi + b))
    
    a_out = a - a_gradient * L
    b_out = b - b_gradient * L
    return a_out, b_out


def runLinearRegression(x, y, L=0.00001, num_epochs=300, xlabel='', ylabel='', rand_init=True):
    if rand_init:
        a = random.randint(-10,10)
        b = random.randint(-10,10)
    else:
        a = 0.
        b = 0. 

    data = [x,y]

    fig, ax = plt.subplots(1) 
    ax.scatter(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x_lin_reg = np.linspace(min(x), max(x), 100)
    for i in range(num_epochs):
        # Calculating the corresponding y values for the linear regression line
        y_lin_reg = a * x_lin_reg + b
        a, b = gradientDescent(a,b, data , L)
        ax.plot(x_lin_reg, y_lin_reg, color='red', alpha =0.1)
    return a, b, ax

'''Logistic regression functions'''


def sigmoidal(x):
    return 1/ (1 + np.exp(-x))

def sigmoidalClip(x, cp = 10000):
    '''For large positive values of x, np.exp(-x) will be close to zero, so 1 / (1 + np.exp(-x)) will be close to 1 without risking overflow.
    For large negative values of x, np.exp(x) will be close to zero, so np.exp(x) / (1 + np.exp(x)) will be close to 0 without risking underflow.
    For values of x close to zero, both terms will be well-behaved.'''
    return 1 / (1 + np.exp(np.clip(-x, -cp, cp)))

def getAccuracy(y_pred, y_test):
    #print(y_pred == y_test)
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc

def check_types(data):
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {type(value)}")
            check_types(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"Index {i}: {type(item)}")
            check_types(item)
    elif isinstance(data, np.ndarray):
        for i in range(data.shape[0]):
            print(f"Index {i}: {type(data[i])}")
            check_types(data[i])
    else:
        print(f"{data}: {type(data)}")
        
class LogisticRegression():
    def __init__(self, L=0.00001, num_epochs=1000, reg=False, C=0.1):
        self.L = np.float64(L)
        self.num_epochs = num_epochs
        self.weights = None
        self.bias=None
        self.reg=reg
        self.C = C

    def gradientDescent(self, X, Y):
        '''C=lambda'''
        num_samp, num_feats = X.shape
        self.weights = np.float64(np.random.uniform(-0.1, 0.1, size=num_feats))
        self.bias = np.float64(0.)
        for i in range(self.num_epochs):
            predictions = sigmoidalClip(np.dot(X, self.weights)+self.bias)
            w_gradient = np.float64((1/num_samp) * np.dot(X.T, (predictions -Y)))
            bias_gradient = np.float64((1/num_samp) * np.sum(predictions -Y))
            ## Regularizartion:
            if self.reg == 'lasso':
                penalty = self.C * np.abs(self.weights) # L1 LASSO regularization
            elif self.reg == 'ridge':
                penalty = self.C * self.weights**2 # L2 RIGDE regularization
            else:
                penalty = 0.

            self.weights -= (w_gradient * self.L + np.float64(penalty))
            self.bias -= bias_gradient * self.L
            
            # Check if weights are approaching negative infinity
            if np.any(np.isinf(self.weights)):
                warnings.warn(f'Warning: Weights approached infinity at epoch {i}. Consider adjusting the learning rate or regularization parameter')
                break

    def makePredictions(self, X):
        y_predictions = sigmoidalClip(np.dot(X,self.weights) + self.bias)
        categorical_predictions = []
        for y in y_predictions:
            if y <= 0.5: cat_pred = 0
            else: cat_pred = 1
            categorical_predictions.append(cat_pred )
        return categorical_predictions


