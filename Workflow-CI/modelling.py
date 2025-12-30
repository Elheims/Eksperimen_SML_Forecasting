import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import os

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, 'data_train_scaled.csv')
    test_path = os.path.join(script_dir, 'data_test_scaled.csv')

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
except FileNotFoundError:
    print(f"Error: File not found at {train_path} or {test_path}")
    exit()

TARGET_COL = 'Target_Power'

# Pisah X dan y
X_train = df_train.drop(columns=[TARGET_COL])
y_train = df_train[TARGET_COL]
X_test = df_test.drop(columns=[TARGET_COL])
y_test = df_test[TARGET_COL]

print(f"Train shape: {X_train.shape}")


mlflow.set_experiment("Solar_Panel_Forecasting")


#XGBOOST
mlflow.xgboost.autolog(log_models=True)

with mlflow.start_run(run_name="XGBoost_Autolog"):
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=67)
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)