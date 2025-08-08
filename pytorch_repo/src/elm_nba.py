import sys
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
# Add the parent directory of the pytorch directory to the system path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preproc.data_prep import process_data
from models.extremeLM import *
from models.nba_models.classifier import ClassifDs
from torch.utils.data import DataLoader


def main():
    """
    This script performs Extreme Learning Machine (ELM) classification on NBA player data.

    It loads the data, preprocesses it, splits it into training and validation sets,
    trains the ELM model, and evaluates its accuracy.

    The script uses the following modules:
    - preproc.data_prep: for data preprocessing
    - models.extremeLM: for ELM model implementation
    - models.nba_models.classifier: for dataset preparation
    """

    pos = ['POINT GUARD',  'SMALL FORWARD', 'CENTER'] #'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD',
    chosen_feats = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage', 
                    'three_point_attempt_rate', 'offensive_rebounds', 'blocks', 'assists', 'free_throw_attempt_rate','attempted_three_point_field_goals']

    X_scaled, Y = process_data('data/nba_stats_full_merged.csv', 'positions', pos, chosen_feats=chosen_feats)
    #print(X_scaled.shape,  Y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, Y, test_size = 0.2)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')

    input_size = X_train.shape[1]
    
    input_weights = np.random.normal(size=[input_size,h_nodes])
    biases = np.random.normal(size=[h_nodes])

    beta = np.dot(linalg.pinv(hidden_nodes(X_train, input_weights, biases)), y_train)

    y_val_pred = predict(X_val, input_weights, biases, beta)
    correct = 0
    total = X_val.shape[0]

    for i in range(total):
        y_val_pred_s = np.round(y_val_pred[i], 0)
        y_val_true = y_val[i]
        correct += 1 if y_val_pred_s == y_val_true else 0
    accuracy = correct/total
    print(f'Accuracy for {h_nodes} hidden nodes: {accuracy}')

    cnt = Counter(y_val)
    print(np.max(list(cnt.values())) / total)

if __name__ == "__main__":
    #h_nodes = 10000
    #main()

    for net_size in [1000, 5000, 10000, 50000]:
        #print(f'Hidden nodes: {net_size}')
        h_nodes = net_size
        main()

    




