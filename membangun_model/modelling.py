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
df_train = pd.read_csv('data_train_scaled.csv')
df_test = pd.read_csv('data_test_scaled.csv')

# Load Scaler
try:
    scaler_y = joblib.load('scaler_y.pkl')
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

#DEFINISI FUNGSI TRAINING
def train_eval_log(model, model_name, X_train, y_train, X_test, y_test, scaler_y):
    
    with mlflow.start_run(run_name=model_name):
        print(f"\nMemulai Training: {model_name}")
        
        # Training
        model.fit(X_train, y_train.ravel())
        
        # Prediksi (Skala 0-1)
        pred_scaled = model.predict(X_test)
        
        # Inverse Transform 
        pred_final = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
        y_test_final = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        # Hitung Error
        mae = mean_absolute_error(y_test_final, pred_final)
        rmse = np.sqrt(mean_squared_error(y_test_final, pred_final))
        r2 = r2_score(y_test_final, pred_final)
        
        print(f"Hasil {model_name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Logging ke MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        if "XGBoost" in model_name:
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("learning_rate", model.learning_rate)
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
            
        print(f"logging {model_name} ke MLflow.")

#EKSEKUSI MODEL

# Linear Regression
lr_model = LinearRegression()
train_eval_log(lr_model, "Linear_Regression", X_train, y_train, X_test, y_test, scaler_y)

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
train_eval_log(xgb_model, "XGBoost", X_train, y_train, X_test, y_test, scaler_y)