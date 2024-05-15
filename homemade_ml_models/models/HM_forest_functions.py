'''ML homemade functions by Jan Stasinski
FORESTS'''
import sklearn
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
import warnings
from collections import Counter


def getAccuracy(y_pred, y_test):
    acc = np.sum(y_pred == y_test) / len(y_test)
    return acc

class SingleNode:
    def __init__(self, feat=None,th=None, lNode=None, rNode=None, *,value=None):
        self.feat = feat
        self.th = th
        self.lNode = lNode
        self.rNode = rNode
        self.value = value

    def leafCheck(self):
        return self.value is not None

class HMDecisionTree:
    def __init__(self, min_split=2, max_depth=50, num_feats=None):
        self.min_split = min_split
        self.max_depth = max_depth
        self.num_feats = num_feats
        self.root=None
    
    def fit(self, X, Y):
        if not self.num_feats: self.num_feats = X.shape[1]
        else:  self.num_feats = min(X.shape[1], self.num_feats)
        self.root = self._growTree(X,Y)

    def _growTree(self, X, Y, depth=0):
        n_samp, n_feats = X.shape
        n_labels = len(np.unique(Y))

        # stopping criteria:
        if (depth >= self.max_depth or n_labels == 1 or n_samp < self.min_split ):
            leaf_val = self._mostCommonLabel(Y)
            return SingleNode(value=leaf_val)
        
        # finding optimal splitting point:
        fidxs = np.random.choice(n_feats, self.num_feats, replace=False)
        opt_feat, opt_thr = self._optSplit(X, Y, fidxs) 

        # make child nodes after splitting
        left_idxs, right_idxs = self._getSplit(X[:,opt_feat], opt_thr)
        # Breaking mechanism: If either subset is empty, return a leaf node with the most common label
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return SingleNode(value=self._mostCommonLabel(Y))

        lNode = self._growTree(X[left_idxs,:], Y[left_idxs], depth+1)
        rNode = self._growTree(X[right_idxs,:], Y[right_idxs], depth+1)
        #return nodes based on the node class above and the features
        return SingleNode(opt_feat, opt_thr, lNode, rNode)
    
    def _optSplit(self, X, Y, fidxs):
        curr_gain = -1
        split_idx= None
        split_thr = None
        for fidx in fidxs:
            Xcol = X[:, fidx]
            possible_thresholds = np.unique(Xcol)
            for t in possible_thresholds:
                gain = self._calcGain(Y, Xcol, t)
                if gain > curr_gain:
                    curr_gain = gain
                    split_idx = fidx
                    split_thr = t
        return split_idx, split_thr
    
    def _calcGain(self,Y, Xcol, thr):
        p_entropy  = self._calcEntropy(Y)
        # get splits
        Lidxs, Ridxs = self._getSplit(Xcol, thr)
        # weighted average entropy of children nodes:
        # no splits no enthropy:
        if len(Lidxs) == 0 or len(Ridxs) ==0:
            return 0
        ch_entropy = (len(Lidxs)/ len(Y)) * self._calcEntropy(Y[Lidxs]) + (len(Ridxs)/ len(Y)) * self._calcEntropy(Y[Ridxs])
        # return information gain:
        return p_entropy - ch_entropy

    def _getSplit(self, Xcol, thr):
        Lidxs = np.argwhere(Xcol <= thr).flatten()
        Ridxs = np.argwhere(Xcol > thr).flatten()
        return Lidxs, Ridxs

    def _calcEntropy(self, Y):
        bin_count = np.bincount(Y)
        Ps = bin_count / len(Y)
        entropy = -np.sum([p * np.log(p) for p in Ps if p>0])
        return entropy

    def _mostCommonLabel(self, Y):
        if len(Y) == 0:
            raise ValueError("Input array Y is empty.")
        count = Counter(Y)
        val = count.most_common(1)[0][0]
        return val

    def makePredictions(self, X):
        prediction = np.array([self._walkDownTree(x, self.root) for x in X])
        return prediction
    
    def _walkDownTree(self, x, node):
        #print(f"Current node feature: {node.feat}, threshold: {node.th}") 
        if node.leafCheck():
            return node.value
        if x[node.feat] <= node.th:
            return self._walkDownTree(x, node.lNode)
        else:
            return self._walkDownTree(x, node.rNode)

        

### Extending this to the RandomForest:
class HMRandomForest:
    def __init__(self, num_trees=30, max_depth=10, min_split=2, num_feats=None ):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_split = min_split
        self.num_feats = num_feats
        self.tree_list = []

    def fit(self, X, Y):
        self.tree_list = []
        for t in range(self.num_trees):
            single_tree = HMDecisionTree(max_depth=self.max_depth,
                         min_split=self.min_split, num_feats=self.num_feats)
            sampleX, sampleY = self._getBootSamples(X, Y)
            single_tree.fit(sampleX, sampleY)
            self.tree_list.append(single_tree)

    def _getBootSamples(self, X, Y):
        num_samples = X.shape[0]
        sidxs = np.random.choice(num_samples, num_samples, replace=True)
        return X[sidxs], Y[sidxs]

    def _mostCommonLabel(self, Y):
        if len(Y) == 0:
            raise ValueError("Input array Y is empty.")
        count = Counter(Y)
        val = count.most_common(1)[0][0]
        return val
        
    def makePredictions(self, X):
        tree_predictions = np.array([t.makePredictions(X) for t in self.tree_list])
        tree_predictions = np.swapaxes(tree_predictions, 0,1)
        most_common_labels = np.array([self._mostCommonLabel(p) for p in tree_predictions])
        return most_common_labels




    
    



