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

from HM_regression_functions import *

from data_prep import *

def train_and_evaluate_model(X_scaled, Y):
    train_x, test_x, train_y, test_y = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)
    
    for r in ['None', 'lasso', 'ridge']:
        LR = LogisticRegression(L=0.001, num_epochs=1500, reg=r, C=0.01)
        LR.gradientDescent(train_x, train_y)
        y_predictions = LR.makePredictions(test_x)
        
        accuracy = getAccuracy(y_predictions, test_y)
        print(f"Accuracy {r}: {accuracy}")


if __name__ == "__main__":
    pos = ['POINT GUARD', 'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD', 'CENTER']
    #selected_pos = pos[::4] ## select the positions# Process and run LogR:
    selected_pos = pos[:2]

    drop_feats = ['slug', 'games_played', 'minutes_played']
    chosen_feats = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage', 
                    'three_point_attempt_rate', 'offensive_rebounds', 'blocks', 'assists', 'free_throw_attempt_rate','attempted_three_point_field_goals']

    X_scaled, Y = process_data('nba_data/stats_full_merged.csv', 'positions', selected_pos, feats2drop = drop_feats, chosen_feats=chosen_feats)

    train_and_evaluate_model(X_scaled, Y)



#X1_scaled, Y1 = process_data('stats_24.csv', ['POINT GUARD', 'CENTER'])



# drop_feats_a = ['slug','age', 'team', 'games_played', 'is_combined_totals']
# drop_feats_b = ['slug','age', 'team', 'games_played', 'games_started', 'minutes_played']
# chosen_feats_a = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage']
# chosen_feats_b = ['defensive_rebounds', 'assists', 'steals', 'blocks']

# X1_scaled, Y1 = process_data('stats_24_basic.csv', 'positions', selected_pos, feats2drop = drop_feats_b)#, chosen_feats=chosen_feats_b)

# # Process and run LogR - advanced data
# X1a_scaled, Ya = process_data('stats_24_adv.csv', 'positions', selected_pos, feats2drop = drop_feats_a)
# train_and_evaluate_model(X1a_scaled, Ya)