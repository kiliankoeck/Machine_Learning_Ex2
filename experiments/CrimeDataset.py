import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from implementation.RandomForestImpl import RandomForest as RFCustom
from implementation.ChatGPT import RandomForestRegressor as RFChatGPT
from sklearn.ensemble import RandomForestRegressor as RFSklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

data_file = 'datasets/Communities_Crime/communities.data'
names_file = 'datasets/Communities_Crime/communities.names'

columns = []
with open(names_file, 'r') as f:
    for line in f:
        if "@attribute" in line:
            col_name = line.split(' ')[1].strip()
            columns.append(col_name)

data = pd.read_csv(data_file, header=None, names=columns, na_values='?')

# drop first five columns because they are not predictive
data.drop(columns=data.columns[:5],axis=1, inplace=True)

# Function to clip outliers
def cap_outliers_percentile(df, column):
    lower_bound = df[column].quantile(0.05)
    upper_bound = df[column].quantile(0.95)
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

# Identify numerical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns

# Clip outliers in numerical columns
for column in numerical_cols:
    cap_outliers_percentile(data, column)

X = data.drop(columns=['ViolentCrimesPerPop'])
y = data['ViolentCrimesPerPop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

# Function to evaluate a pipeline
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nEvaluating {model_name}...")
    
    # Training time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Training Time: {training_time:.4f}s, Prediction Time: {prediction_time:.4f}s")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# Evaluate default models
evaluate_model(RFCustom(), X_train, X_test, y_train, y_test, "Custom Random Forest")
evaluate_model(RFChatGPT(), X_train, X_test, y_train, y_test, "ChatGPT Random Forest")
evaluate_model(RFSklearn(), X_train, X_test, y_train, y_test, "Sklearn Random Forest")

# Grid Search for Custom Random Forest
param_grid = {
    'max_depth': [5, 10],        
    'n_trees': [10, 50, 100],     
    'n_features': ['sqrt', 'log2']  
}

def perform_grid_search(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    print(f"\nPerforming Grid Search for {model_name}...")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
    
    # Training with grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Best model evaluation
    best_model = grid_search.best_estimator_
    start_time = time.time()
    y_pred = best_model.predict(X_test)
    prediction_time = time.time() - start_time

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Grid Search completed in {training_time:.2f}s.")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Training Time: {training_time:.4f}s, Prediction Time: {prediction_time:.4f}s")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

perform_grid_search(RFCustom(), param_grid, X_train, y_train, X_test, y_test, "Custom Random Forest")