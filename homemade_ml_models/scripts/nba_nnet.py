'''run code for neural net nba dataset'''
#
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import time
from flexHM_nnet import *

from data_prep import process_data

def train_and_evaluate_model(X_scaled, Y, num_inter= 1200, hl_shape=[8,5], L=0.5, verbose=True):
    train_x, test_x, train_y, test_y = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)

    Xt_train =  train_x.T
    print(Xt_train.shape, train_y.shape)
    Xt_test =  test_x.T

    network_shape = [len(chosen_feats)] + hl_shape + [5]
    print(f'Network Shape: {network_shape}')

    t1 = time.time()
    NN = flexNetwork(network_shape)
    NN.gradientDescent(Xt_train, train_y , ReLu, 1, num_inter,verbose=verbose)

    # for i in range(3):
    #     NN.testPrediction(random.randint(0,5), Xt_train, train_y, ReLu)
    Y_predictions = NN.makePredictions(Xt_test, ReLu)
    acc = NN.get_network_accuracy(Y_predictions, test_y)
    print(f'Test Accuracy: {acc}')
    return acc


if __name__ == "__main__":
    pos = ['POINT GUARD', 'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD', 'CENTER']
    #selected_pos = pos[::4] ## select the positions# Process and run LogR:
    selected_pos = pos

    drop_feats = ['slug', 'games_played', 'minutes_played']
    chosen_feats = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage', 
                    'three_point_attempt_rate', 'offensive_rebounds', 'blocks', 'assists', 'free_throw_attempt_rate','attempted_three_point_field_goals']

    X_scaled, Y = process_data('nba_data/stats_full_merged.csv', 'positions', selected_pos, feats2drop = drop_feats, chosen_feats=chosen_feats)

    #train_and_evaluate_model(X_scaled, Y, num_inter= 1000, hl_shape=[14,10,5])

    for i in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
        
        acc = train_and_evaluate_model(X_scaled, Y, num_inter= 1500, hl_shape=[14,10,5], L=i, verbose=False)
        print(f'Learning rate: {i} --> acc: {acc}')
    