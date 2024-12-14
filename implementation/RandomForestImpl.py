import numpy as np
import pandas as pd
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
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y = y.to_numpy() if isinstance(y, pd.Series) else y
        self.n_features = X.shape[1] // 3 if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # To fix error in house dataset
        if best_feature is None or best_thresh is None:
           return Node(value=np.mean(y))

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

        mse_left = np.sum((y[left_idxs] - np.mean(y[left_idxs])) ** 2) / n_l
        mse_right = np.sum((y[right_idxs] - np.mean(y[right_idxs])) ** 2) / n_r

        mse = (n_l * mse_left + n_r * mse_right) / n

        return mse

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()

        return left_idxs, right_idxs

    def predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        # Convert X and y to numpy arrays if they are pandas objects
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y = y.to_numpy() if isinstance(y, pd.Series) else y
        self.trees = []

        # Resolve the number of features to use in splits
        n_total_features = X.shape[1]
        if isinstance(self.n_features, str):
            if self.n_features == 'sqrt':
                resolved_n_features = int(np.sqrt(n_total_features))
            elif self.n_features == 'log2':
                resolved_n_features = int(np.log2(n_total_features))
            else:
                raise ValueError(f"Unknown n_features value: {self.n_features}")
        else:
            resolved_n_features = self.n_features if self.n_features else n_total_features

        # Train each decision tree
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=resolved_n_features  # Pass the resolved number of features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples // 3, replace=True) 
        return X[idxs], y[idxs]

    def predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        return np.array([np.mean(pred) for pred in tree_preds])

    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        return self

    def get_params(self, deep=True):
        return {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'n_features': self.n_features
        }
