import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data CSV
df_train = pd.read_csv('namadataset_preprocessing/data_train_scaled.csv')
df_test = pd.read_csv('namadataset_preprocessing/data_test_scaled.csv')

# Load Scaler
try:
    scaler_y = joblib.load('namadataset_preprocessing/scaler_y.pkl')
except FileNotFoundError:
    print("ERROR filenya gk ada")
    exit()

#PREPARE FEATURES & TARGET
TARGET_COL = 'Target_Power'

# Pisah X dan y untuk Train
X_train = df_train.drop(columns=[TARGET_COL])
y_train = df_train[TARGET_COL].values

# Pisah X dan y untuk Test
X_test = df_test.drop(columns=[TARGET_COL])
y_test = df_test[TARGET_COL].values

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

#SETUP MLFLOW
mlflow.set_experiment("Solar_Panel_Forecasting_Comparison")
mlflow.autolog()

print("Starting training with Autolog enabled...")

# --- LINEAR REGRESSION ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train.ravel())
print("Linear Regression trained.")

# --- XGBOOST ---
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train.ravel())
print("XGBoost trained.")

# Save for interference using joblib (required for serving criteria)
joblib.dump(xgb_model, 'model_xgboost.pkl')
print(f"XGBoost model saved locally to model_xgboost.pkl")