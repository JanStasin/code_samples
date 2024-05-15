
'''This is a flexible neural network class that can be used for any number of layers and nodes
by JStasinski'''

import numpy as np
import pickle
import os
import pandas as pd
import math
import time
import random

def sigmoid(Z, derivative=False):
    if derivative:
        return sigmoid(Z) * (1 - sigmoid(Z))
    return 1.0 / (1.0 + np.exp(-Z))

def ReLu(Z, derivative=False):
    if derivative:
        return Z > 0
    return np.maximum(0, Z)

def SoftMax(Z):
    '''computes softmax takes Z as an argument and returns probabilities '''
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def SoftMax2(Z):
    exp_z = np.exp(Z - np.max(Z))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Define the Swish activation function
def swish(x, beta=1):
    return (x * sigmoid(beta * x))

def vectorize_labels(labels):
    v = np.zeros((labels.size, max(labels) + 1))
    v[np.arange(labels.size), labels] = 1.0
    v = v.T
    return v

class flexNetwork(object):
    def __init__(self , L_sizes):
        '''network setup based on layer sizes array supplied at the beginning
        each element of the array defines the number of nodes for each layer.
         is the basic input size 10 is the basic output size (last element of the array)
        we dont need to create activation values in advance they will be dynamically computed during feedforward 
        procedure '''
        self.L_sizes = L_sizes
        self.num_layers = len(self.L_sizes)
        print(f'Num_layers: {self.num_layers}')
        self.Ws = [np.random.rand(self.L_sizes[i], self.L_sizes[i-1])-0.5 for i in range(1, self.num_layers)]
        self.Bs = [np.random.rand(self.L_sizes[i], 1)- 0.5 for i in range(1, self.num_layers)]
        print('Ws shapes: ', [w.shape for w in self.Ws])
        print('Bs shapes: ', [b.shape for b in self.Bs])


    def forward_feed(self, data_input, a_func):
        '''runs the feed forward through the whole network
        inputs: network state, input data and activation function
        outputs: Zs and activations for each layer'''
        Zs = [np.zeros((self.L_sizes[i], 1)) for i in range(1, self.num_layers)]
        As = [np.zeros((self.L_sizes[i], 1)) for i in range(1, self.num_layers)]
        for lidx in range(self.num_layers-1):
            #print(f'Forward {lidx}')
            if lidx == 0:
                Zs[lidx] = np.dot(self.Ws[lidx], data_input) + self.Bs[lidx]
                As[lidx] = a_func(Zs[lidx])
            elif lidx == self.num_layers-2:
                Zs[lidx] = np.dot(self.Ws[lidx], As[lidx-1]) + self.Bs[lidx]
                As[-1] = SoftMax(Zs[-1])
            else:
                Zs[lidx] = np.dot(self.Ws[lidx], As[lidx-1]) + self.Bs[lidx]
                As[lidx] = a_func(Zs[lidx])
        return Zs, As

    def backprop_feed(self, Zs, As, data_input, labels, a_func):
        '''Runs the backpropagation through the whole network'''
        d_Ws = [np.zeros((self.L_sizes[i], self.L_sizes[i-1])) for i in range(1, self.num_layers)]
        d_Bs = [np.zeros((self.L_sizes[i], 1)) for i in range(1, self.num_layers)]
        d_Zs = [np.zeros((self.L_sizes[i], 1)) for i in range(1, self.num_layers)] 
        v_labels = vectorize_labels(labels)
        # Initialize d_Zs from the last layer
        d_Zs[-1] = As[-1] - v_labels 
        # Backpropagate through the network
        for lidx in range(self.num_layers-2, -1, -1):
            #print(f'Backprop {lidx}')
            if lidx == self.num_layers-2:
                d_Ws[-1] = 1/v_labels.size * np.dot(d_Zs[-1], As[lidx-1].T)
                d_Bs[-1] = 1/v_labels.size * np.sum(d_Zs[-1], axis=1)
            elif lidx == 0:
                d_Zs[lidx] = np.dot(self.Ws[lidx+1].T, d_Zs[lidx+1]) * a_func(Zs[lidx], derivative=True)
                d_Ws[lidx] = 1/v_labels.size * np.dot(d_Zs[lidx], data_input.T)
                d_Bs[lidx] = 1/v_labels.size * np.sum(d_Zs[lidx], axis=1)
            else:
                d_Zs[lidx] = np.dot(self.Ws[lidx+1].T, d_Zs[lidx+1]) * a_func(Zs[lidx], derivative=True)
                d_Ws[lidx] = 1/v_labels.size * np.dot(d_Zs[lidx], As[lidx-1].T)
                d_Bs[lidx] = 1/v_labels.size * np.sum(d_Zs[lidx], axis=1)
        #self._run_shape_checks(Zs, As, d_Zs, d_Ws, d_Bs)
        return d_Ws, d_Bs

    def _run_shape_checks(self, Zs, As, d_Zs, d_Ws, d_Bs):
            [print(f'Ws[{i}]-->{ar.shape}') for i, ar in enumerate(self.Ws)]
            [print(f'Bs[{i}]-->{ar.shape}') for i, ar in enumerate(self.Bs)]
            [print(f'Zs[{i}]-->{ar.shape}') for i, ar in enumerate(Zs)]
            [print(f'As[{i}]-->{ar.shape}') for i, ar in enumerate(As)]
            [print(f'd_Zs[{i}]-->{ar.shape}') for i, ar in enumerate(d_Zs)]
            [print(f'd_Ws[{i}]-->{ar.shape}') for i, ar in enumerate(d_Ws)]
            [print(f'd_Bs[{i}]-->{ar.shape}') for i, ar in enumerate(d_Bs)]

    def update(self, d_Ws, d_Bs, learning_rate):
        for uidx in range(self.num_layers-1):
            #print(f'Update layer: {uidx}')
            self.Ws[uidx] -= (learning_rate * d_Ws[uidx])
            self.Bs[uidx] -= (learning_rate * np.reshape(d_Bs[uidx], (len(d_Bs[uidx]),1)))
        return self.Ws, self.Bs

    def interpret_output(self, output_layer):
        '''takes As[-1] - activations of the ouput layer and outputs the corresponding 0-9 integer'''
        return np.argmax(output_layer, axis=0)
    
    def get_network_accuracy(self, guesses, labels):
        '''as advertised: compares all network guesses against the corresponding labels
        Arguments: guesses and labels 
        Output: Percentage correct - accuracy''' 
        return np.sum(guesses == labels) / labels.size
        
    def gradientDescent(self, data_input, labels, a_func, learning_rate, iterations, verbose=True):
        '''Runs a gradient decent algorithm on the whole input data set for the specified number of times
        Arguments: data, corresponding labels, activation function
        learning rate,  number of times to run the data through the algorythm
        Output: state of the network'''
        
        for i in range(iterations):
            Zs, As = self.forward_feed(data_input, a_func)
            d_Ws, d_Bs = self.backprop_feed(Zs, As, data_input, labels, a_func)
            self.Ws, self.Bs = self.update(d_Ws, d_Bs, learning_rate)
            if verbose and i % 200 == 0:
                print("Iteration: ", i)
                guesses = self.interpret_output(As[-1])
                print('Accuracy: ', self.get_network_accuracy(guesses, labels))
        guesses = self.interpret_output(As[-1])
        print('Training accuracy: ', self.get_network_accuracy(guesses, labels)) 
        return self.get_network_accuracy(guesses, labels)
        
    def makePredictions(self, data_input, a_func):
        '''function used for getting the predictions on the validation / testing datasets
        applies interpret output function to the output layer after a forward feed with data
        Arguments: data and activation function (used for training)'''
        print(data_input.shape)
        *_ , As = self.forward_feed(data_input, a_func)
        predictions = self.interpret_output(As[-1])
        return predictions

    def testPrediction(self, index, train_data, train_labels, a_func):
        prediction = self.makePredictions(train_data[:, index, None], a_func)
        label = train_labels[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
        # print("Prediction: ", selected_pos[int(prediction)])
        # print("Label: ", selected_pos[label])





    
                      