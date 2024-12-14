from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

from implementation.RandomForestImpl import RandomForest

data = datasets.load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=1234
)

ml = RandomForest()
ml.fit(X_train, y_train)
predictions = ml.predict(X_test)

def mse(y_test, y_pred):
    return np.mean((y_test - y_pred) ** 2)

print("Mean Squared Error own:", mse(y_test, predictions))

ml = RandomForestRegressor()
ml.fit(X_train, y_train)
predictions = ml.predict(X_test)

print("Mean Squared Error sklearn rf:", mse(y_test, predictions))

ml = KNeighborsRegressor()
ml.fit(X_train, y_train)
predictions = ml.predict(X_test)

print("Mean Squared Error sklearn kn:", mse(y_test, predictions))