'''K-means HOMEMADE by JanStasinski
unsupervised, unlabeled'''

import sklearn
import numpy as np
import scipy
import pandas as pd
import random
import itertools
import warnings
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from data_prep import *

class HM_Kmeans:
    def __init__(self, K=5, max_iterations=150, plot_process=False):
        self.K = K
        self.max_iterations = max_iterations
        self.plot_process = plot_process

        # Clusters and centroids:
        self.cluster_idxs = [[] for c in range(self.K)]
        self.centriods = []

    def makePredictions(self, X):
        self.X = X
        self.num_samp = X.shape[0]
        self.num_feats = X.shape[1]
        
        rand_idxs = np.random.choice(self.num_samp, self.K, replace=False)
        self.centroids = [X[idx] for idx in rand_idxs]
        
        # Clustermaking loop:
        for i in range(self.max_iterations):
            self.clusters = self._makeClusters(self.centroids)

            if self.plot_process: self.plot()

            last_centroids = self.centroids
            self.centroids = self._getCentroids(self.clusters)
            
            if self._hasConverged(last_centroids):
                print(f'Clustering converged at iteration: {i} ')
                break

            if self.plot_process: self.plot()
        return self._returnClusterLabels(self.clusters)

    def _makeClusters(self, centroids):
        # assigning samples to clusters
        clusters = [[] for c in range(self.K)]
        for idx, samp in enumerate(self.X):
            centroid_idx = self._closestCentroid(samp, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closestCentroid(self , samp, centroids):
        #get all euclidean distances at once via LC:
        euc_distances = [calcEucDistance(samp, centroid) for centroid in centroids]
        min_dist_idx = np.argmin(euc_distances)
        return min_dist_idx

    def _getCentroids(self, clusters):
        centroids = np.zeros((self.K, self.num_feats))
        centroids = [np.mean(self.X[cluster], axis=0) for cluster in clusters]
        return centroids

    def _hasConverged(self, last_centroids):
        ## convergence is based on distanced between last and current centroids:
        cent_distances = [calcEucDistance(self.centroids[k], last_centroids[k]) for k in range(self.K)]
        return sum(cent_distances) == 0
        
    def _returnClusterLabels(self, clusters):
        #matching the assigned clusters with the cluster labels 
        labels = np.empty(self.num_samp)
        for cidx, cluster in enumerate(self.clusters):
            for samp_idx in cluster:
                labels[samp_idx] = cidx
        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(10,5))

        for cluster in self.clusters:
            dp = self.X[cluster].T
            ax.scatter(*dp)

        for centroid in self.centroids:
            ax.scatter(*centroid, marker='x', color='k', linewidth=2.2)
        plt.show()


def calcEucDistance(A,B):
    return np.sqrt(np.sum(A-B)**2)

def getAccuracy(y_pred, y_test):
    #print(y_pred == y_test)
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc


if __name__ == '__main__':

    pos = ['POINT GUARD', 'SHOOTING GUARD', 'SMALL FORWARD', 'POWER FORWARD', 'CENTER']
    selected_pos = pos[::4] ## select the positions# Process and run LogR:
    #selected_pos = pos[:2]
    print(f'predicting player positions: {selected_pos}')

    drop_feats = ['slug', 'games_played', 'minutes_played']
    chosen_feats = ['total_rebound_percentage', 'assist_percentage', 'steal_percentage','block_percentage', 
                    'three_point_attempt_rate', 'offensive_rebounds', 'blocks', 'assists', 'free_throw_attempt_rate','attempted_three_point_field_goals']

    chosen_feats2 = ['assist_percentage', 'three_point_attempt_rate']

    X_scaled, Y = process_data('nba_data/stats_full_merged.csv', 'positions', selected_pos, feats2drop=drop_feats, chosen_feats=chosen_feats)
    KM = HM_Kmeans(K=len(selected_pos), max_iterations=100, plot_process=False)
    Y_predict = KM.makePredictions(X_scaled)
    accuracy = getAccuracy(Y_predict, Y)
    # Print the cluster labels for each player
    #print("Cluster labels for each player:", Ya_predict)
    print(f'Accuracy for advanced data: {accuracy}')

    l = list(itertools.combinations(chosen_feats, 2))

    # for comb in l:
    #     chosen_feats = list(comb)
    #     X_scaled, Y = process_data('nba_data/stats_full_merged.csv', 'positions', selected_pos, feats2drop = drop_feats, chosen_feats=comb)
    #     KM = HM_Kmeans(K=len(selected_pos), max_iterations=100, plot_process=False)
    #     Y_predict = KM.makePredictions(X_scaled)
    #     accuracy = getAccuracy(Y_predict, Y)
    #     # Print the cluster labels for each player
    #     #print("Cluster labels for each player:", Ya_predict)
    #     print(f'Accuracy for {comb} advanced data: {accuracy}')
    

    # '''toy example'''
    # from sklearn.datasets import make_blobs
    # X,Y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True)
    # num_of_clusters = len(np.unique(Y))
    # print(X.shape, Y)
    # k_toy = HM_Kmeans(K=num_of_clusters, max_iterations=350, plot_process=True)
    # y_predict = k_toy.makePredictions(X)
    # k_toy.plot()
    # print(f'toy example run with no errors')
    # '''end of the toy example'''









            

