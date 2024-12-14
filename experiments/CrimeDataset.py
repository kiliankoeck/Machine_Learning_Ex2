import pandas as pd
from sklearn.model_selection import train_test_split
from implementation.RandomForestImpl import RandomForest as RFCustom
from implementation.ChatGPT import RandomForestRegressor as RFChatGPT
from sklearn.ensemble import RandomForestRegressor as RFSklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

data_file = '../datasets/Communities_Crime/communities.data'
names_file = '../datasets/Communities_Crime/communities.names'

columns = []
with open(names_file, 'r') as f:
    for line in f:
        if "@attribute" in line:
            col_name = line.split(' ')[1].strip()
            columns.append(col_name)

data = pd.read_csv(data_file, header=None, names=columns, na_values='?')

#drop first five columns because they are not predictive
data.drop(columns=data.columns[:5],axis=1, inplace=True)

X = data.drop(columns=['ViolentCrimesPerPop'])
y = data['ViolentCrimesPerPop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

model_custom = RFCustom()
model_custom.fit(X_train, y_train)
y_pred_custom = model_custom.predict(X_test)

model_chatgpt = RFChatGPT()
model_chatgpt.fit(X_train, y_train)
y_pred_chatgpt = model_chatgpt.predict(X_test)

model_sklearn = RFSklearn()
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)

print("Mean Squared Error own:", mean_squared_error(y_test, y_pred_custom))
print("R2 own:", r2_score(y_test, y_pred_custom))

print("Mean Squared Error chatgpt:", mean_squared_error(y_test, y_pred_chatgpt))
print("R2 chatgpt:", r2_score(y_test, y_pred_chatgpt))

print("Mean Squared Error sklearn rf:", mean_squared_error(y_test, y_pred_sklearn))
print("R2 sklearn rf:", r2_score(y_test, y_pred_sklearn))


