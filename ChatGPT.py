import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split


class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            return self._create_leaf(y)

        best_feature, best_thresh = self._best_split(X, y)
        if best_feature is None:
            return self._create_leaf(y)

        left_idx, right_idx = self._split(X[:, best_feature], best_thresh)
        left_tree = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return {"feature": best_feature, "threshold": best_thresh, "left": left_tree, "right": right_tree}

    def _create_leaf(self, y):
        return np.mean(y)

    def _best_split(self, X, y):
        num_features = X.shape[1]
        best_mse = float("inf")
        best_feature, best_thresh = None, None

        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for thresh in thresholds:
                left_idx, right_idx = self._split(X[:, feature_idx], thresh)
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                mse = self._calculate_mse(y[left_idx], y[right_idx])
                if mse < best_mse:
                    best_mse, best_feature, best_thresh = mse, feature_idx, thresh

        return best_feature, best_thresh

    def _split(self, feature_values, threshold):
        left_idx = np.where(feature_values <= threshold)[0]
        right_idx = np.where(feature_values > threshold)[0]
        return left_idx, right_idx

    def _calculate_mse(self, left_y, right_y):
        mse_left = np.var(left_y) * len(left_y)
        mse_right = np.var(right_y) * len(right_y)
        return (mse_left + mse_right) / (len(left_y) + len(right_y))

    def predict_row(self, node, row):
        if not isinstance(node, dict):
            return node

        if row[node["feature"]] <= node["threshold"]:
            return self.predict_row(node["left"], row)
        else:
            return self.predict_row(node["right"], row)

    def predict(self, X):
        return np.array([self.predict_row(self.tree, row) for row in X])


class RandomForestRegressor:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


data = datasets.load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=1234
)

ml = RandomForestRegressor()
ml.fit(X_train, y_train)
predictions = ml.predict(X_test)

def mse(y_test, y_pred):
    return np.mean((y_test - y_pred) ** 2)

print("Mean Squared Error ChatGPT:", mse(y_test, predictions))

