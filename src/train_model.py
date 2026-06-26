import sqlite3
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

conn = sqlite3.connect("data/ev_stations.db")
df = pd.read_sql_query("SELECT * FROM utilization_forecast", conn)

numeric_cols = ['hour',
                'day_of_week',
                'is_weekend',
                'total_ports',
                'is_fast_charging_hub',
                'fast_charger_ratio',
                'temperature',
                'humidity',
                'wind_speed',]

categorical_cols = ['network', 'weather_condition']

encoder = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

X = df[numeric_cols + categorical_cols]
y = df['utilization_rate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state = 0
)

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.fit_transform(X_test)

model = xgb.XGBRegressor(
    n_estimators = 200,
    max_depth = 5,
    learning_rate = 0.1,
    random_state = 0
)

model.fit(X_train_encoded, y_train)

with open("models/xgboost_ev_model.pkl", 'wb') as f:
    pickle.dump(model, f)

with open("models/encoder.pkl", 'wb') as f:
    pickle.dump(encoder, f)