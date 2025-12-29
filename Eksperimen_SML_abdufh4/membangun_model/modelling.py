import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# 1. LOAD DATA
try:
    df_train = pd.read_csv('data_train_scaled.csv')
    df_test = pd.read_csv('data_test_scaled.csv')
except FileNotFoundError:
    print("Error: File data_train_scaled.csv atau data_test_scaled.csv.")
    exit()

TARGET_COL = 'Target_Power'

# Pisah X dan y
X_train = df_train.drop(columns=[TARGET_COL])
y_train = df_train[TARGET_COL]
X_test = df_test.drop(columns=[TARGET_COL])
y_test = df_test[TARGET_COL]

print(f"Data siap. Train shape: {X_train.shape}")

# 2. SETUP MLFLOW
mlflow.set_experiment("Solar_Panel_Forecasting_Basic")

# 3. TRAINING DENGAN AUTOLOG

#XGBOOST
mlflow.xgboost.autolog(log_models=True)

with mlflow.start_run(run_name="XGBoost_Autolog"):
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=67)
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

#Linear Regression
mlflow.sklearn.autolog(log_models=True)

with mlflow.start_run(run_name="Linear_Regression_Autolog"):
    lr_model = LinearRegression()
    
    lr_model.fit(X_train, y_train)
    
    lr_model.predict(X_test)