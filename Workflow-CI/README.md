# Solar Panel Power Forecasting

This project implements machine learning models to forecast solar panel power output. It includes validation using both **Linear Regression** and **XGBoost**, with advanced hyperparameter tuning via **Optuna**.

## Features

- **Models**: Linear Regression & XGBoost Regressor.
- **Experiment Tracking**: Integrated with **MLflow** to track metrics (RMSE, MAE, R2), parameters, and artifacts.
- **Hyperparameter Tuning**: Automated tuning using **Optuna** with `MLflowCallback` for real-time tracking.
- **Visualization**: Auto-generated plots (Optimization History, Parameter Importance) saved to MLflow.
- **Optuna Dashboard**: Interactive dashboard to explore tuning results.

## Installation

1.  Navigate to the project directory:

    ```bash
    cd Workflow-CI
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training & Tuning

Run the main script to train the models and perform Optuna hyperparameter tuning:

```bash
python modelling.py
```

This will:

- Train a base Linear Regression model.
- Train a base XGBoost model.
- Run 50 trials of Optuna optimization for XGBoost.
- Log all results to a local MLflow backend (`mlflow.db`).

### 2. Viewing Results (MLflow UI)

To see the experiment results, metrics, and artifacts:

```bash
mlflow ui
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 3. Optuna Dashboard

To visualize the hyperparameter search (Parallel Coordinates, Importance, etc.):

```bash
python run_dashboard.py
```

Then open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

## Project Structure

- `modelling.py`: Main script for training and tuning.
- `requirements.txt`: Python dependencies.
- `run_dashboard.py`: Portable launcher for the Optuna Dashboard.
- `mlflow.db`: SQLite database for MLflow tracking.
- `optuna.db`: SQLite database for Optuna study storage.
