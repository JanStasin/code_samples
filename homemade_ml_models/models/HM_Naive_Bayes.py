'''Naive Bayes HOMEMADE by JanStasinski

P(Y/X) = [P(X/Y) * P(Y)] / P(X)
posterior_distribution = [likelihood * prior_distribution] / marginal data probability

Y - predicted classes / categories - LABELS
X - features

Naive: independence, normally distributed
'''

import sklearn
import numpy as np
import scipy
#import pandas as pd
import random
import itertools
import warnings
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
#from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from data_prep import *

class HMNaiveBayes:

    def fit(self, X, Y):
        self.X = X
        self.num_samp = X.shape[0]
        self.num_feats = X.shape[1]
        self.classes = np.unique(Y)
        num_classes = len( self.classes)

        #get the descriptive statistics:
        self.means = np.zeros((num_classes, self.num_feats))
        self.variances = np.zeros((num_classes, self.num_feats))
        #get priors:
        self.priors = np.zeros((num_classes))

        for kidx, klas in enumerate(self.classes):
            Xk = X[Y==klas]
            self.means[kidx:] = np.mean(Xk, axis=0)
            self.variances[kidx:] = np.var(Xk, axis=0)
            self.priors[kidx] = Xk.shape[0] / float(self.num_samp)
            #print(self.priors)

    def inPredict(self, x):
        posteriors = []
        for kidx, klas in enumerate(self.classes):
            # get the prior and posterior probabilities:
            prior = np.log(self.priors[kidx]) 
            posterior = np.sum(np.log(self.calcPDF(kidx,x)))
            ## update the beliefs
            posteriors.append(posterior + prior)
        return self.classes[np.argmax(posteriors)]

    def calcPDF(self,kidx, x):
        ### get the probability distribution
        class_mean = self.means[kidx]
        class_variance = self.variances[kidx]
        L = np.exp(-((x-class_mean) ** 2) / (2 * class_variance))
        M = np.sqrt(2 * np.pi * class_variance)
        return  L / M

    def makePredictions(self,X):
        return np.array([self.inPredict(x) for x in X])


if __name__ == '__main__':
    
    pos = ['POINT GUARD', 'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD', 'CENTER']
    selected_pos = pos[::2] ## select the positions# Process and run LogR:
    #selected_pos = pos[:2]
    print(f'predicting player positions: {selected_pos}')

    drop_feats = ['slug', 'games_played', 'minutes_played']
    chosen_feats = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage', 
                    'three_point_attempt_rate', 'offensive_rebounds', 'blocks', 'assists', 'free_throw_attempt_rate','attempted_three_point_field_goals']

    chosen_feats2 = ['assist_percentage', 'three_point_attempt_rate']

    X_scaled, Y = process_data('nba_data/stats_full_merged.csv', 'positions', selected_pos, feats2drop=drop_feats, chosen_feats=chosen_feats)
    NB = HMNaiveBayes()
    train_X, test_X, train_Y, test_Y = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)
    NB.fit(train_X, train_Y)
    Y_predict = NB.makePredictions(test_X)
    accuracy = getAccuracy(Y_predict , test_Y)
    print(f'Accuracy for merged data: {accuracy}')








