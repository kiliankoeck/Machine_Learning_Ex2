import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from implementation.RandomForestImpl import RandomForest as RFCustom
from implementation.ChatGPT import RandomForestRegressor as RFChatGPT
from sklearn.ensemble import RandomForestRegressor as RFSklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = '../datasets/House_Rent/House_Rent_Dataset.csv'
house_rent_data = pd.read_csv(file_path)

house_rent_data = house_rent_data.drop(columns=['Posted On'])


def extract_floors(x):
    current_floor = []
    total_floors = []
    for i in x:
        i = i.replace('Ground', '0')
        i = i.replace('Upper Basement', '-1')
        i = i.replace('Lower Basement', '-2')
        splits = i.split(' out of ')
        current = int(splits[0])
        if len(splits) == 1:
            total = current
        else:
            total = int(splits[1])
        # + 2 to make all values positive, + 3 because f.ex 1 out of 1 does not include ground and basements in total
        # amount of floors. assumption -> all buildings have basements and ground floor
        current_floor.append(current + 2)
        total_floors.append(total + 3)

    return current_floor, total_floors


current_floor, total_floors = extract_floors(house_rent_data['Floor'])
house_rent_data['Current Floor'] = current_floor
house_rent_data['Total Floors'] = total_floors
# Area Locality deleted for perfomance reasons
house_rent_data.drop(columns=['Floor', 'Area Locality'], inplace=True)

nominal_cols = ['Area Type', 'Tenant Preferred', 'City', 'Point of Contact']
ordinal_cols = ['Furnishing Status']
numerical_cols = house_rent_data.drop(columns=nominal_cols + ordinal_cols + ["Rent"]).columns

X = house_rent_data.drop(columns=['Rent'])
y = house_rent_data['Rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

nominal_pipeline = Pipeline([
    ('encoder', OneHotEncoder()),
    ('dim_reducer', PCA())
])

ordinal_pipeline = Pipeline([
    ('encoder', OrdinalEncoder())
])

columnTransformer = ColumnTransformer([
    ('numeric', numeric_pipeline, numerical_cols),
    ('ordinal', ordinal_pipeline, ordinal_cols),
    ('nominal', nominal_pipeline, nominal_cols)
])

custom_pipeline = Pipeline([
    ('transformer', columnTransformer),
    ('model', RFCustom())
])

chatgpt_pipeline = Pipeline([
    ('transformer', columnTransformer),
    ('model', RFChatGPT())
])

sklearn_pipeline = Pipeline([
    ('transformer', columnTransformer),
    ('model', RFSklearn())
])

custom_pipeline.fit(X_train, y_train)
y_pred_custom = custom_pipeline.predict(X_test)

chatgpt_pipeline.fit(X_train, y_train)
y_pred_chatgpt = chatgpt_pipeline.predict(X_test)

sklearn_pipeline.fit(X_train, y_train)
y_pred_sklearn = sklearn_pipeline.predict(X_test)

print("Mean Squared Error own:", mean_squared_error(y_test, y_pred_custom))
print("R2 own:", r2_score(y_test, y_pred_custom))

print("Mean Squared Error chatgpt:", mean_squared_error(y_test, y_pred_chatgpt))
print("R2 chatgpt:", r2_score(y_test, y_pred_chatgpt))

print("Mean Squared Error sklearn rf:", mean_squared_error(y_test, y_pred_sklearn))
print("R2 sklearn rf:", r2_score(y_test, y_pred_sklearn))
