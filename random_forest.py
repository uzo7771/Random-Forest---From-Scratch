
import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath("../Regression-Trees---From-Scratch"))
from regression_trees import RegressionTree


class RandomForest:
    def __init__(self, X, y, max_depth=10, min_samples_split=5, min_samples_leaf = 1,
                  n_features=None, n_trees = 10, bootstrap_sample_percentage=0.5):
        if n_features is None:
            n_features = int(np.sqrt(X.shape[1]))
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.n_trees = n_trees
        self.bootstrap_sample_size = int(np.shape(X)[0]*bootstrap_sample_percentage)
    def fit(self):
        self.feature_importance = np.zeros((self.n_trees, self.X.shape[1]))
        self.list_of_trees = []

        for i in range(self.n_trees):
            observations_indexes = random.choices(range(np.shape(self.X)[0]), k=self.bootstrap_sample_size)
            X_bootstrap = self.X[observations_indexes,:]
            y_bootstrap = self.y[observations_indexes]

            tree = RegressionTree(X_bootstrap, y_bootstrap,  self.max_depth, self.min_samples_split, self.min_samples_leaf, self.n_features)
            tree.fit()
            self.feature_importance[i,:] = tree.feature_importance
            self.list_of_trees.append(tree)
        self.feature_importance = np.mean(self.feature_importance, axis=0)

    def predict(self, X):
        result = np.zeros((self.n_trees, np.shape(X)[0]))
        for i, tree in enumerate(self.list_of_trees):
            result[i,:] = tree.predict(X)
        result = np.mean(result, axis=0)
        return result