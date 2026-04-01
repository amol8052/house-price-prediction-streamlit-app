# src/model.py
import os
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=["MedHouseVal"])
    y = data.frame["MedHouseVal"]
    return X, y

def train_and_save_model(save_path="models/house_rf.joblib"):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)

    print(f"Saved model to {save_path}")
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    train_and_save_model()
