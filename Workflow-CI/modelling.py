import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from optuna.integration.mlflow import MLflowCallback
import optuna.visualization.matplotlib as plot_mpl
import math
import os

def plot_actual_vs_predicted(y_true, y_pred, title):
    # Ensure inputs are numpy arrays for consistent slicing
    if hasattr(y_true, 'values'):
        y_true = y_true.values
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Line Chart (100 samples) 
    limit = 100
    ax1.plot(range(len(y_true[:limit])), y_true[:limit], label='Actual', color='blue', alpha=0.7)
    ax1.plot(range(len(y_pred[:limit])), y_pred[:limit], label='Predicted', color='red', alpha=0.7, linestyle='--')
    ax1.set_title(f'{title} - Actual vs Predicted (First {limit} Samples)')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Power Output')
    ax1.legend()
    ax1.grid(True)
    
    # Scatter Plot for overall correlation
    ax2.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    ax2.set_title(f'{title} - Scatter Plot')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# Check file paths
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

with mlflow.start_run(run_name="XGBoost_Untuned"):
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=67)
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    mlflow.xgboost.log_model(xgb_model, name="model")
    #Log Prediction Plot
    xgb_preds = xgb_model.predict(X_test)
    
    # Calculate Metrics for Untuned XGBoost
    rmse_untuned = math.sqrt(mean_squared_error(y_test, xgb_preds))
    mae_untuned = mean_absolute_error(y_test, xgb_preds)
    r2_untuned = r2_score(y_test, xgb_preds)
    
    print(f"Untuned XGBoost Test Metrics: RMSE={rmse_untuned:.4f}, MAE={mae_untuned:.4f}, R2={r2_untuned:.4f}")
    
    mlflow.log_metric("eval_rmse", rmse_untuned)
    mlflow.log_metric("eval_mae", mae_untuned)
    mlflow.log_metric("eval_r2", r2_untuned)

    fig_xgb = plot_actual_vs_predicted(y_test, xgb_preds, "Untuned XGBoost")
    mlflow.log_figure(fig_xgb, "xgboost_untuned_actual_vs_predicted.png")
    plt.close(fig_xgb)


# Linear Regression
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run(run_name="LinearRegression"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    mlflow.sklearn.log_model(lr_model, name="model")
    
    # Evaluate on Test/Validation Set
    y_pred = lr_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Linear Regression Test Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    mlflow.log_metric("eval_rmse", rmse)
    mlflow.log_metric("eval_mae", mae)
    mlflow.log_metric("eval_r2", r2)
    
    # Log Prediction Plot
    fig_lr = plot_actual_vs_predicted(y_test, y_pred, "Linear Regression")
    mlflow.log_figure(fig_lr, "linear_regression_actual_vs_predicted.png")
    plt.close(fig_lr)


# XGBOOST Tuned with Optuna
mlflow.xgboost.autolog(disable=True)

with mlflow.start_run(run_name="XGBoost_Option_Tune"):
    
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=62)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        
        model.fit(
            X_train_opt, y_train_opt,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_val)
        return math.sqrt(mean_squared_error(y_val, preds))

    print("Starting Optuna Hyperparameter Tuning...")
    
    # Setup MLflow Callback
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="rmse",
        mlflow_kwargs={"nested": True}
    )
    
    # Run the optimization
    storage_name = "sqlite:///{}/optuna.db".format(script_dir)
    study = optuna.create_study(direction='minimize', storage=storage_name, study_name="xgboost_optuna", load_if_exists=True)
    study.optimize(objective, n_trials=200, callbacks=[mlflow_callback])

    print("Best params:", study.best_params)
    print("Best CV RMSE:", study.best_value)
    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_cv_rmse", study.best_value)
    
    # Visualizations
    try:
        # Plot optimization history
        ax_history = plot_mpl.plot_optimization_history(study)
        if isinstance(ax_history, matplotlib.axes.Axes):
            fig_history = ax_history.get_figure()
        else:
            fig_history = ax_history
        mlflow.log_figure(fig_history, "optimization_history.png")
        
        # Plot param importances
        ax_importance = plot_mpl.plot_param_importances(study)
        if isinstance(ax_importance, matplotlib.axes.Axes):
            fig_importance = ax_importance.get_figure()
        else:
            fig_importance = ax_importance
        mlflow.log_figure(fig_importance, "param_importances.png")
        
        # Plot slice
        ax_slice = plot_mpl.plot_slice(study)
        if isinstance(ax_slice, matplotlib.axes.Axes):
            fig_slice = ax_slice.get_figure()
        elif isinstance(ax_slice, numpy.ndarray):
             fig_slice = ax_slice.flatten()[0].get_figure()
        else:
            fig_slice = ax_slice
        mlflow.log_figure(fig_slice, "slice_plot.png")
    
    except Exception as e:
        print(f"Vizualization failed: {e}")

    
    # Train best model
    best_params = study.best_params
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False) # Evaluate 
    # Evaluate
    y_pred_tune = best_model.predict(X_test)
    rmse_tune = math.sqrt(mean_squared_error(y_test, y_pred_tune))
    mae_tune = mean_absolute_error(y_test, y_pred_tune)
    r2_tune = r2_score(y_test, y_pred_tune)
    
    mlflow.log_metric("eval_rmse", rmse_tune)
    mlflow.log_metric("eval_mae", mae_tune)
    mlflow.log_metric("eval_r2", r2_tune)
    
    # Log Prediction Plot
    fig_xgb = plot_actual_vs_predicted(y_test, y_pred_tune, "Tuned XGBoost")
    mlflow.log_figure(fig_xgb, "xgboost_actual_vs_predicted.png")
    plt.close(fig_xgb)
    
    print(f"Tuned XGBoost (Optuna) Test Metrics: RMSE={rmse_tune:.4f}, MAE={mae_tune:.4f}, R2={r2_tune:.4f}")
    
    mlflow.xgboost.log_model(best_model, name="model")

# ==========================================
# FINAL EVALUATION TABLE
# ==========================================
print("\n" + "="*30)
print("FINAL MODEL EVALUATION")
print("="*30)

results = {
    'Model': ['Linear Regression', 'XGBoost (Untuned)', 'XGBoost (Tuned)'],
    'RMSE': [rmse, rmse_untuned, rmse_tune],
    'MAE': [mae, mae_untuned, mae_tune],
    'R2 Score': [r2, r2_untuned, r2_tune]
}

df_results = pd.DataFrame(results)
try:
    print(df_results.to_markdown(index=False, floatfmt=".4f"))
except ImportError:
    print(df_results.to_string(index=False, float_format="%.4f"))

# Optional: Log results table to MLflow
with mlflow.start_run(run_name="Final_Evaluation_Summary"):
    df_results.to_csv("final_evaluation_results.csv", index=False)
    mlflow.log_artifact("final_evaluation_results.csv")
