import pandas as pd
import matplotlib.pyplot as plt

file_path = '../datasets/House_Rent/House_Rent_Dataset.csv'
house_rent_data = pd.read_csv(file_path)

print(house_rent_data.head())
print("Data Information:")
print(house_rent_data.info())
print("\nSummary Statistics:")
print(house_rent_data.describe())
print("\nMissing Values:")
print(house_rent_data.isnull().sum())
print("\nCheck for Duplicates:")
print(house_rent_data.duplicated().sum())

print("\nUnique Values:")
for column in house_rent_data.columns:
    print(f"{column}: {house_rent_data[column].nunique()}")

nominal_cols = ['Area Type', 'Tenant Preferred', 'City']
categorical_cols = ['Furnishing Status']
numerical_cols = house_rent_data.drop(columns=nominal_cols + categorical_cols).columns
# Visualize boxplots before clipping
plt.figure(figsize=(8, len(numerical_cols) * 3))
for i, column in enumerate(numerical_cols, 1):
    plt.subplot(len(numerical_cols), 1, i)
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

for column in numerical_cols:
    cap_outliers_percentile(house_rent_data, column)

# Visualize boxplots after clipping
plt.figure(figsize=(8, len(numerical_cols) * 3))
for i, column in enumerate(numerical_cols, 1):
    plt.subplot(len(numerical_cols), 1, i)
    plt.boxplot(house_rent_data[column], vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.title(f'Boxplot of {column} (After Clipping)')
    plt.xlabel(column)

plt.tight_layout()
plt.savefig('house_no_outlier.png')
plt.show()
