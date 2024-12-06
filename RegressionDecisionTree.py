import numpy as np

# base implementation following https://www.youtube.com/watch?v=NxEHSAfFlK8
class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth=depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth=depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_mse = float('inf')
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                mse = self._mean_squared_error(y, X_column, thr)
                if mse < best_mse:
                    best_mse = mse
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _mean_squared_error(self, y, X_column, threshold):

        left_idxs, right_idxs = self._split(X_column, threshold)

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        if n_l == 0 or n_r == 0:
            return float('inf')

        mse_left = np.sum((y[left_idxs] - np.mean(y[left_idxs]))**2)/n_l
        mse_right = np.sum((y[right_idxs] - np.mean(y[right_idxs]))**2)/n_r

        mse = (n_l * mse_left + n_r * mse_right) / n

        return mse

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()

        return left_idxs, right_idxs


    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
