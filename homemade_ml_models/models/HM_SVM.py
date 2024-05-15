'''
HANDMADE SVM algorithm by Jan Stasinski
for a binary classification problem
with multiple features
'''
import numpy as np
from data_prep import *

from sklearn.model_selection import train_test_split 

class SupportVectorMachine:
    
    def __init__(self, L=0.0001, _lambda=0.05, num_iterations=500):
        self.L = L
        self._lambda = _lambda
        self.num_iterations = num_iterations
        self.weights = None
        self.bias =None
    
    def fit(self, X, Y):
        """
        Train the support vector machine model.
        Parameters:
        X (numpy.ndarray): Input features.
        Y (numpy.ndarray): Output labels.
        """
        self.num_samp = X.shape[0]  # Number of samples
        self.num_feats = X.shape[1]  # Number of features

        y = np.where(Y <= 0, -1, 1)  # Convert labels to -1 and 1
        self.weights = np.random.random(self.num_feats)  # Initialize weights randomly
        self.biases = np.random.random(1)  # Initialize bias randomly
        
        for _ in range(self.num_iterations):
            for idx, xi in enumerate(X):
                # Checking the condition for updating weights and biases
                if y[idx] * (np.dot(xi, self.weights) - self.biases) >= 1:
                    self.weights -= self.L * (2 * self._lambda * self.weights)
                else:
                    self.weights -= self.L * (2 * self._lambda * self.weights - np.dot(xi, y[idx]))
                    self.biases -= self.L * y[idx]


    def makePredictions(self, X):
        return np.sign(np.dot(X, self.weights) - self.biases) # np.sign returns the sign of each element in an array. 


if __name__ == '__main__':
    
    pos = ['POINT GUARD', 'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD', 'CENTER']
    selected_pos = pos[::4] ## select the positions
    print(selected_pos)

    X_scaled, Y = process_data('stats_24_basic.csv', selected_pos, chosen_feats=['defensive_rebounds', 'assists'])
    SVM = SupportVectorMachine(L=0.0005, _lambda=0.001, num_iterations=2000)
    train_X, test_X, train_Y, test_Y = train_test_split(X_scaled, Y, test_size=0.25, random_state=0)
    SVM.fit(train_X, train_Y)
    Y_predict = SVM.makePredictions(test_X)
    accuracy = getAccuracy(Y_predict , test_Y)
    print(f'Accuracy for basic data: {accuracy}')

    Xa_scaled, Ya = process_data('stats_24_adv.csv', selected_pos, chosen_feats=[ 'total_rebound_percentage', 'assist_percentage'])
    SVM = SupportVectorMachine(L=0.0005, _lambda=0.001, num_iterations=2000)
    train_X, test_X, train_Y, test_Y = train_test_split(Xa_scaled, Ya, test_size=0.25, random_state=0)
    SVM.fit(train_X, train_Y)
    Y_predict = SVM.makePredictions(test_X)
    accuracy = getAccuracy(Y_predict, test_Y)            
    # Print the cluster labels for each player
    #print("Cluster labels for each player:", Ya_predict)
    print(f'Accuracy for advanced data: {accuracy}')

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model):
    # Define a grid
    h = .02 # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for each point in the grid
    Z = model.makePredictions(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']), alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#00FF00']), edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(X_scaled, Y, SVM)

