import sklearn
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score

from HM_forest_functions import *

from data_prep import *

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def train_and_evaluate_model(X_scaled, Y, ts=0.2, method='DT'):
    train_x, test_x, train_y, test_y = train_test_split(X_scaled, Y, test_size=ts, random_state=0)
    if method == 'DT':
        # Create and fit the model
        DT = HMDecisionTree()
        DT.fit(train_x, train_y)
        y_predictions = DT.makePredictions(test_x)
        accuracy = getAccuracy(y_predictions, test_y)
        return accuracy
    elif method == 'RF':
        # Create and fit the model
        RF = HMRandomForest(num_trees=20, max_depth=20)
        RF.fit(train_x, train_y)
        y_predictions = RF.makePredictions(test_x)
        accuracy = getAccuracy(y_predictions, test_y)
        return accuracy


if __name__ == '__main__':

    pos = ['POINT GUARD', 'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD', 'CENTER']
    #selected_pos = pos[::4] ## select the positions# Process and run LogR:
    selected_pos = pos[:2]
    print(f'predicting player positions: {selected_pos}')

    drop_feats = ['slug', 'games_played', 'minutes_played']
    chosen_feats = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage', 
                    'three_point_attempt_rate', 'offensive_rebounds', 'blocks', 'assists', 'free_throw_attempt_rate','attempted_three_point_field_goals']

    X_scaled, Y = process_data('nba_data/stats_full_merged.csv', 'positions', selected_pos, feats2drop = drop_feats, chosen_feats=chosen_feats)

    acc1 = train_and_evaluate_model(X_scaled, Y, ts=0.2, method='DT')
    acc2 = train_and_evaluate_model(X_scaled, Y, ts=0.2, method='RF')

    print(f"DT Accuracy: {acc1}")
    print(f"RF Accuracy: {acc2}")


    # pos = ['POINT GUARD', 'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD', 'CENTER']
    # selected_pos = pos[::4]
    
    # # either drop or select features for X:
    # drop_feats_a = ['slug','age', 'team', 'games_played', 'is_combined_totals']
    # drop_feats_b = ['slug','age', 'team', 'games_played', 'games_started', 'minutes_played']
    # chosen_feats_a = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage']
    # chosen_feats_b = ['defensive_rebounds', 'assists', 'steals', 'blocks']
    # # single decision tree  basic data
    # X1_scaled, Y1 = process_data('stats_24_basic.csv', 'positions', pos, feats2drop = drop_feats_b)# ,chosen_feats=chosen_feats_a)
    # print(f'predicting player positions: {selected_pos}')
    # accuracy_basic = train_and_evaluate_model(X1_scaled, Y1, ts=0.2, method='DT')
    # print(f"DT Accuracy for basic data: {accuracy_basic}")

    # # single decision tree advanced data
    # X1a_scaled, Ya = process_data('stats_24_adv.csv',  'positions', pos,
    #  feats2drop = drop_feats_a)# ,chosen_feats=chosen_feats_a)
    # accuracy_advanced = train_and_evaluate_model(X1a_scaled, Ya, ts=0.2, method='DT')
    # print(f"DT Accuracy for advanced data: {accuracy_advanced}")

    # # random forest basic data
    # X1_scaled, Y1 = process_data('stats_24_basic.csv',  'positions', pos, chosen_feats=chosen_feats_b)
    # accuracy_basic = train_and_evaluate_model(X1_scaled, Y1, ts=0.2, method='RF')
    # print(f"RF Accuracy for basic data: {accuracy_basic}")

    # # random forest advanced data
    # X1a_scaled, Ya = process_data('stats_24_adv.csv', 'positions', pos,  chosen_feats=chosen_feats_a)
    # accuracy_advanced = train_and_evaluate_model(X1a_scaled, Ya, ts=0.2, method='RF')
    # print(f"RF Accuracy for advanced data: {accuracy_advanced}")

'''toy example:'''
# Xm = np.tile(np.array([[1, 2], [3, 4], [5, 6]]), (55,1))
# Ym = np.tile(np.array([1, 0, 1]), 55)

# Xmt = np.tile(np.array([[1, 2], [3, 7], [5, 6]]), (15,1))
# Ymt = np.tile(np.array([1, 0, 1]), 15)

# print(Y ,X1)

# DTm = HMDecisionTree()
# DTm.fit(Xm, Ym)
# m_predictions = DTm.makePredictions(Xmt)
# # sk_accuracy= accuracy_score(m_predictions, Ymt)
# # print(sk_accuracy)
# maccuracy = getAccuracy(m_predictions, Ymt)
# print(maccuracy)