import sqlite3
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb
import pickle

def load_data(db_path = 'data/ev_stations.db'):
    conn = sqlite3.connect("data/ev_stations.db")
    df = pd.read_sql_query("SELECT * FROM utilization_forecast", conn)
    conn.close()
    return df


def prepare_features(df):
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

    return X_train_encoded, X_test_encoded, y_train, y_test, encoder

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators = 200,
        max_depth = 5,
        learning_rate = 0.1,
        random_state = 0
    )

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}

def save_artifacts(model, encoder, model_path = 'models/xgboost_ev_model.pkl',
                   encoder_path = 'models/encoder.pkl'):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
        

def run():
    print("Loading data...")
    df = load_data()

    print("Preparing features...")
    X_train, X_test, y_train, y_test, encoder = prepare_features(df)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving artifiacts...")
    save_artifacts(model, encoder)

if __name__ == "__main__":
    run()