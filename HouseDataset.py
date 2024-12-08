import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from RegressionRandomForest import RandomForest  
import numpy as np
from ChatGPT import RandomForestRegressor

file_path = 'archive/House_Rent_Dataset.csv'  
house_rent_data = pd.read_csv(file_path)

print(house_rent_data.head())
print("Data Information:")
print(house_rent_data.info())
print("\nSummary Statistics:")
print(house_rent_data.describe() )
print("\nMissing Values:")
print(house_rent_data.isnull().sum())
print("\nCheck for Duplicates:")
print(house_rent_data.duplicated().sum())

# label because i think its easier than the hot one (we would have to many columns)
label_encoder = LabelEncoder()
columns_to_encode = ['Area Type', 'Furnishing Status', 'Tenant Preferred', 'City']

for col in columns_to_encode:
    house_rent_data[col] = label_encoder.fit_transform(house_rent_data[col])

house_rent_data_encoded = house_rent_data.drop(columns=['Point of Contact'])

numerical_columns = ['Rent', 'Size', 'BHK']

plt.figure(figsize=(8, len(numerical_columns) * 3))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 1, i)
    plt.boxplot(house_rent_data_encoded[column], vert=False, patch_artist=True, boxprops=dict(facecolor="skyblue"))
    plt.title(f'Boxplot of {column} (Before Clipping)')
    plt.xlabel(column)

plt.tight_layout()
plt.savefig('house_outlier.png')  
plt.show()

def cap_outliers_percentile(df, column): 
    lower_bound = df[column].quantile(0.05)  
    upper_bound = df[column].quantile(0.95)  
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

for column in numerical_columns:
    cap_outliers_percentile(house_rent_data_encoded, column)

# another boxplot for comparison to check if the outliers were removed
plt.figure(figsize=(8, len(numerical_columns) * 3))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 1, i)
    plt.boxplot(house_rent_data_encoded[column], vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.title(f'Boxplot of {column} (After Clipping)')
    plt.xlabel(column)

plt.tight_layout()
plt.savefig('house_no_outlier.png')  # Save the plot
plt.show()


scaler = StandardScaler()

house_rent_data_encoded[numerical_columns] = scaler.fit_transform(house_rent_data_encoded[numerical_columns])

print("Scaled Data:")
print(house_rent_data_encoded.head())

print("\nSummary Statistics After Scaling:")
print(house_rent_data_encoded[numerical_columns].describe())
X = house_rent_data_encoded.drop(columns=['Rent'])
y = house_rent_data_encoded['Rent']


X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

rf_model = RandomForest(n_trees=10, max_depth=10, min_samples_split=2)
rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print("Mean Squared Error (Custom Random Forest):", mse(y_test, predictions))

ml = RandomForestRegressor()
ml.fit(X_train, y_train)
predictions = ml.predict(X_test)
print("Mean Squared Error ChatGPT:", mse(y_test, predictions))

