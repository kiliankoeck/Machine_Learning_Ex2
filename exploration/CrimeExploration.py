import pandas as pd

data_file = '../datasets/Communities_Crime/communities.data'
names_file = '../datasets/Communities_Crime/communities.names'

columns = []
with open(names_file, 'r') as f:
    for line in f:
        if "@attribute" in line:
            col_name = line.split(' ')[1].strip()
            columns.append(col_name)

data = pd.read_csv(data_file, header=None, names=columns, na_values='?')

data.drop(columns=data.columns[:5],axis=1, inplace=True)

for col in data.columns:
    print(f"{col} - {data[col].dtype}")

print(data.head())
print("Data Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())
print("\nCheck for Duplicates:")
print(data.duplicated().sum())
