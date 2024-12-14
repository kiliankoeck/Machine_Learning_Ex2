import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from implementation.RandomForestImpl import RandomForest as RFCustom
from implementation.ChatGPT import RandomForestRegressor as RFChatGPT
from sklearn.ensemble import RandomForestRegressor as RFSklearn

# Load the dataset
file_path = '../datasets/House_Rent/House_Rent_Dataset.csv'
house_rent_data = pd.read_csv(file_path)
house_rent_data = house_rent_data.drop(columns=['Posted On'])


def extract_floors(x):
    current_floor = []
    total_floors = []
    for i in x:
        i = i.replace('Ground', '0').replace('Upper Basement', '-1').replace('Lower Basement', '-2')
        splits = i.split(' out of ')
        current = int(splits[0])
        total = int(splits[1]) if len(splits) > 1 else current
        current_floor.append(current + 2)
        total_floors.append(total + 3)
    return current_floor, total_floors


current_floor, total_floors = extract_floors(house_rent_data['Floor'])
house_rent_data['Current Floor'] = current_floor
house_rent_data['Total Floors'] = total_floors
house_rent_data.drop(columns=['Floor', 'Area Locality'], inplace=True)

nominal_cols = ['Area Type', 'Tenant Preferred', 'City', 'Point of Contact']
ordinal_cols = ['Furnishing Status']
numerical_cols = house_rent_data.drop(columns=nominal_cols + ordinal_cols + ["Rent"]).columns

X = house_rent_data.drop(columns=['Rent'])
y = house_rent_data['Rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_pipeline = Pipeline([('scaler', RobustScaler())])
nominal_pipeline = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
ordinal_pipeline = Pipeline([('encoder', OrdinalEncoder())])

columnTransformer = ColumnTransformer([
    ('numeric', numeric_pipeline, numerical_cols),
    ('ordinal', ordinal_pipeline, ordinal_cols),
    ('nominal', nominal_pipeline, nominal_cols)
])


# Function to clip outliers
def cap_outliers_percentile(df, column):
    lower_bound = df[column].quantile(0.05)
    upper_bound = df[column].quantile(0.95)
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)


# Clip outliers in numerical columns
for column in numerical_cols:
    cap_outliers_percentile(X_train, column)

# Pipelines with default parameters
custom_pipeline = Pipeline([('transformer', columnTransformer), ('model', RFCustom())])
chatgpt_pipeline = Pipeline([('transformer', columnTransformer), ('model', RFChatGPT())])
sklearn_pipeline = Pipeline([('transformer', columnTransformer), ('model', RFSklearn())])

runtimes = []
scores = []


# Function to evaluate default models
def evaluate_pipeline(pipeline, model_name):
    print(f"\nEvaluating {model_name}...")

    # Training time
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Prediction time
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    prediction_time = time.time() - start_time

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    runtimes.append((model_name, "training time", training_time))
    runtimes.append((model_name, "prediction time", prediction_time))
    scores.append((model_name, "mse", mse))
    scores.append((model_name, "r2", r2))

    print(f"Training Time: {training_time:.4f}s, Prediction Time: {prediction_time:.4f}s")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")


# Evaluate models with default parameters
evaluate_pipeline(custom_pipeline, "Custom Random Forest")
evaluate_pipeline(chatgpt_pipeline, "ChatGPT Random Forest")
evaluate_pipeline(sklearn_pipeline, "Sklearn Random Forest")

runtimes = pd.DataFrame(runtimes, columns=['Model', 'Type', 'Time'])
runtimes.to_csv('../results/data/runtimes_housing.csv', index=False)

scores = pd.DataFrame(scores, columns=['Model', 'Type', 'Score'])
scores.to_csv('../results/data/scores_housing.csv', index=False)

# Parameter grid for Grid Search for Custom Random Forest
param_grid = {
    'model__max_depth': [None, 5, 10],
    'model__n_trees': [10, 50, 100],
    'model__n_features': [None, 10, 50, 100],
}


# Perform Grid Search for Custom Random Forest
def perform_grid_search(pipeline, param_grid, model_name):
    print(f"\nPerforming Grid Search for {model_name}...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)

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

    pd.DataFrame(grid_search.cv_results_).to_csv(f'../results/data/grid_search_results_{model_name}_housing.csv',
                                                 index=False)


perform_grid_search(custom_pipeline, param_grid, "Custom Random Forest")
