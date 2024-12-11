import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from RegressionRandomForest import RandomForest
from ChatGPT import RandomForestRegressor
import numpy as np

# File paths
data_file = 'communities+and+crime/communities.data'
names_file = 'communities+and+crime/communities.names'

# Extract column names
columns = []
with open(names_file, 'r') as f:
    for line in f:
        if "attribute" in line:
            col_name = line.split(':')[-1].strip().split()[0]
            clean_name = re.sub(r'\W+', '_', col_name)
            columns.append(clean_name)

# Handle duplicate column names
seen = {}
unique_columns = []
for col in columns:
    if col not in seen:
        seen[col] = 1
        unique_columns.append(col)
    else:
        unique_columns.append(f"{col}_{seen[col]}")
        seen[col] += 1

# Load data
data = pd.read_csv(data_file, header=None, names=unique_columns, na_values='?')

# missing values 
missing_values_summary = data.isnull().sum()
missing_values_summary = missing_values_summary[missing_values_summary > 0].sort_values(ascending=False)
print("Columns with Missing Values:\n")
print(missing_values_summary)

# Handle missing values
# Drop columns with >50% missing values
threshold = 0.5 * len(data)
columns_to_drop = data.columns[data.isnull().sum() > threshold]
data_cleaned = data.drop(columns=columns_to_drop)

# Fill remaining missing values
numerical_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numerical_columns] = data_cleaned[numerical_columns].fillna(data_cleaned[numerical_columns].median())

categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
for col in categorical_columns:
    mode_value = data_cleaned[col].mode()[0]
    data_cleaned[col] = data_cleaned[col].fillna(mode_value)

# Verify that there are no remaining missing values
remaining_missing = data_cleaned.isnull().sum().sum()
print(f"Remaining missing values in the dataset: {remaining_missing}")


categorical_columns = data_cleaned.select_dtypes(include=['object']).columns

# Encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col])
print(f"Categorical columns encoded: {list(categorical_columns)}")

# Handle outliers
# Function to clip outliers at the 5th and 95th percentiles
def cap_outliers_percentile(df, column):
    lower_bound = df[column].quantile(0.05)
    upper_bound = df[column].quantile(0.95)
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

for column in numerical_columns:
    cap_outliers_percentile(data_cleaned, column)

# Scale numerical features
scaler = StandardScaler()
data_cleaned[numerical_columns] = scaler.fit_transform(data_cleaned[numerical_columns])

# Separate features and target
X = data_cleaned.iloc[:, :-1].values  
y = data_cleaned.iloc[:, -1].values   

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Train and evaluate Random Forest models
# Custom Random Forest
rf_model = RandomForest(n_trees=10, max_depth=10, min_samples_split=2)
rf_model.fit(X_train, y_train)
predictions_custom = rf_model.predict(X_test)

# Mean Squared Error function
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print("Mean Squared Error (Custom Random Forest):", mse(y_test, predictions_custom))

# ChatGPT Random Forest
ml = RandomForestRegressor()
ml.fit(X_train, y_train)
predictions_chatgpt = ml.predict(X_test)
print("Mean Squared Error (ChatGPT Random Forest):", mse(y_test, predictions_chatgpt))
