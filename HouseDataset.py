import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from RegressionRandomForest import RandomForest
from ChatGPT import RandomForestRegressor
import numpy as np

# Load the dataset
file_path = 'archive/House_Rent_Dataset.csv'
house_rent_data = pd.read_csv(file_path)

# Initial inspection
print(house_rent_data.head())
print("Data Information:")
print(house_rent_data.info())
print("\nSummary Statistics:")
print(house_rent_data.describe())
print("\nMissing Values:")
print(house_rent_data.isnull().sum())
print("\nCheck for Duplicates:")
print(house_rent_data.duplicated().sum())

# Drop irrelevant columns
#  Removed 'Posted On', 'Point of Contact', and 'Area Locality' as they don't contribute to predictions.
house_rent_data = house_rent_data.drop(columns=['Posted On', 'Point of Contact', 'Area Locality'])

# Encode categorical columns
label_encoder = LabelEncoder()
columns_to_encode = ['Area Type', 'Furnishing Status', 'Tenant Preferred', 'City']

for col in columns_to_encode:
    house_rent_data[col] = label_encoder.fit_transform(house_rent_data[col])


numerical_columns = ['Rent', 'Size', 'BHK']

# Visualize boxplots before clipping
plt.figure(figsize=(8, len(numerical_columns) * 3))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 1, i)
    plt.boxplot(house_rent_data[column], vert=False, patch_artist=True, boxprops=dict(facecolor="skyblue"))
    plt.title(f'Boxplot of {column} (Before Clipping)')
    plt.xlabel(column)

plt.tight_layout()
plt.savefig('house_outlier.png')
plt.show()

# Function to clip outliers
def cap_outliers_percentile(df, column):
    lower_bound = df[column].quantile(0.05)
    upper_bound = df[column].quantile(0.95)
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

for column in numerical_columns:
    cap_outliers_percentile(house_rent_data, column)

# Visualize boxplots after clipping
plt.figure(figsize=(8, len(numerical_columns) * 3))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 1, i)
    plt.boxplot(house_rent_data[column], vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.title(f'Boxplot of {column} (After Clipping)')
    plt.xlabel(column)

plt.tight_layout()
plt.savefig('house_no_outlier.png')
plt.show()

# Scale numerical features
# Scaling applied only to features, not to the target ('Rent')
scaler = StandardScaler()
house_rent_data[numerical_columns[1:]] = scaler.fit_transform(house_rent_data[numerical_columns[1:]])

print("Scaled Data:")
print(house_rent_data.head())

print("\nSummary Statistics After Scaling:")
print(house_rent_data.describe())

# Separate features and target
X = house_rent_data.drop(columns=['Rent'])
y = house_rent_data['Rent']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForest(n_trees=10, max_depth=10, min_samples_split=2)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
predictions = rf_model.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print("Mean Squared Error (Custom Random Forest):", mse(y_test, predictions))

# Train the ChatGPT Random Forest
ml = RandomForestRegressor()
ml.fit(X_train, y_train)

# Evaluate the ChatGPT Random Forest
predictions = ml.predict(X_test)
print("Mean Squared Error ChatGPT:", mse(y_test, predictions))


