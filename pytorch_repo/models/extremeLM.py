
import torch
import torch.nn as nn
from scipy import linalg
import numpy as np

def relu(x):
    """
    Applies the Rectified Linear Unit (ReLU) activation function element-wise to the input array.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying ReLU activation function.
    """
    return np.maximum(x, 0, x)

def hidden_nodes(X, input_weights, biases):
    """
    Computes the hidden nodes of the extremeLM model.

    Args:
        X (numpy.ndarray): Input array.
        input_weights (numpy.ndarray): Input weights.
        biases (numpy.ndarray): Biases.

    Returns:
        numpy.ndarray: Hidden nodes array.
    """
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H

def predict(X, input_weights, biases, beta):
    """
    Predicts the output of the extremeLM model.

    Args:
        X (numpy.ndarray): Input array.
        input_weights (numpy.ndarray): Input weights.
        biases (numpy.ndarray): Biases.
        beta (numpy.ndarray): Beta values.

    Returns:
        numpy.ndarray: Predicted output array.
    """
    out = hidden_nodes(X, input_weights, biases)
    out = np.dot(out, beta)
    return out



def evaluate(X, y, input_weights, biases, beta):
    """
    Evaluates the accuracy of the extremeLM model.

    Args:
        X (numpy.ndarray): Input array.
        y (numpy.ndarray): Target array.
        input_weights (numpy.ndarray): Input weights.
        biases (numpy.ndarray): Biases.
        beta (numpy.ndarray): Beta values.

    Returns:
        float: Accuracy of the model.
    """
    y_pred = predict(X, input_weights, biases, beta)
    correct = 0
    total = X.shape[0]

    for i in range(total):
        y_pred_s = np.round(y_pred[i], 0)
        y_true = y[i]
        correct += 1 if y_pred_s == y_true else 0
    accuracy = correct / total
    return accuracy






