import pandas as pd
import re
from RegressionRandomForest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ChatGPT import RandomForestRegressor
from RegressionRandomForest import RandomForest  
import numpy as np

data_file = 'communities+and+crime\communities.data'
names_file = 'communities+and+crime\communities.names'

columns = []
with open(names_file, 'r') as f:
    for line in f:
        if "attribute" in line:
            col_name = line.split(':')[-1].strip().split()[0]
            clean_name = re.sub(r'\W+', '_', col_name)
            columns.append(clean_name)

seen = {}
unique_columns = []
for col in columns:
    if col not in seen:
        seen[col] = 1
        unique_columns.append(col)
    else:
        unique_columns.append(f"{col}_{seen[col]}")
        seen[col] += 1

data = pd.read_csv(data_file, header=None, names=unique_columns, na_values='?')

# missing values 
missing_values_summary = data.isnull().sum()
missing_values_summary = missing_values_summary[missing_values_summary > 0].sort_values(ascending=False)
print("Columns with Missing Values:\n")
print(missing_values_summary)

# dropping the ones that have 50% missing
threshold = 0.5 * len(data)
columns_to_drop = data.columns[data.isnull().sum() > threshold]
data_cleaned = data.drop(columns=columns_to_drop)

numerical_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numerical_columns] = data_cleaned[numerical_columns].fillna(data_cleaned[numerical_columns].median())

categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
for col in categorical_columns:
    mode_value = data_cleaned[col].mode()[0]
    data_cleaned[col] = data_cleaned[col].fillna(mode_value)

remaining_missing = data_cleaned.isnull().sum().sum()
print(f"Remaining missing values in the dataset: {remaining_missing}")


categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col])

print(f"Categorical columns encoded: {list(categorical_columns)}") 

y = data_cleaned.iloc[:, -1]       
X = data_cleaned.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=1234
)

ml = RandomForestRegressor()
ml.fit(X_train, y_train)
predictions = ml.predict(X_test)

def mse(y_test, y_pred):
    return np.mean((y_test - y_pred) ** 2)

print("Mean Squared Error ChatGPT:", mse(y_test, predictions))
