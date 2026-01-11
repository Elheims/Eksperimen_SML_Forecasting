# Prediksi Daya Panel Surya

Proyek ini mengimplementasikan model pembelajaran mesin untuk memprediksi output daya panel surya. Proyek ini menggunakan model **Regresi Linier** dan **XGBoost**, dengan penyesuaian hiperparameter lanjutan melalui **Optuna**.

## Fitur

- **Model**: Regresi Linier & Regresi XGBoost.
- **Pelacakan Eksperimen**: Terintegrasi dengan **MLflow** untuk melacak metrik (RMSE, MAE, R2), parameter, dan artefak.
- **Penyesuaian Hiperparameter**: Penyesuaian otomatis menggunakan **Optuna** dengan `MLflowCallback` untuk pelacakan real-time.
- **Visualisasi**: Grafik otomatis (Riwayat Optimasi, Pentingnya Parameter) disimpan ke MLflow.
- **Dashboard Optuna**: Dashboard interaktif untuk mengeksplorasi hasil penyesuaian.

## Instalasi

1. Navigasi ke direktori proyek:

   ```bash
   cd Workflow-CI
   ```

2. Instal dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

## Penggunaan

### 1. Pelatihan & Penyesuaian

Jalankan skrip utama untuk melatih model dan melakukan penyesuaian hiperparameter Optuna:

```bash
python modelling.py
```

Ini akan:

- Melatih model Regresi Linier dasar.
- Melatih model XGBoost dasar.
- Menjalankan 50 percobaan optimasi Optuna untuk XGBoost.
- Mencatat semua hasil ke backend MLflow lokal (`mlflow.db`).

### 2. Melihat Hasil (Antarmuka Pengguna MLflow)

Untuk melihat hasil eksperimen, metrik, dan artefak:

```bash
mlflow ui
```

Kemudian buka [http://127.0.0.1:5000](http://127.0.0.1:5000) di browser Anda.

**Untuk Melihat Plot Prediksi (Aktual vs Prediksi):**

1.  Klik nama run eksperimen (contoh: `XGBoost_Option_Tune` atau `LinearRegression`).
2.  Klik tab **"Artifacts"**.
3.  Cari file gambar berikut:
    - `linear_regression_actual_vs_predicted.png`
    - `xgboost_untuned_actual_vs_predicted.png`
    - `xgboost_actual_vs_predicted.png`

Ini akan menampilkan perbandingan visual antara prediksi model dan data aktual.

### 3. Dashboard Optuna

Untuk memvisualisasikan pencarian hiperparameter (Parallel Coordinates, Importance, dll.):

```bash
python run_dashboard.py
```

Kemudian buka [http://127.0.0.1:8080](http://127.0.0.1:8080) di browser Anda.

## Struktur Proyek

- `modelling.py`: Skrip utama untuk pelatihan dan penyetelan.
- `requirements.txt`: Ketergantungan Python.
- `run_dashboard.py`: Peluncur portabel untuk Dashboard Optuna.
- `mlflow.db`: Database SQLite untuk pelacakan MLflow.
- `optuna.db`: Database SQLite untuk penyimpanan studi Optuna.

Translated with DeepL.com (free version)

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

**To View Prediction Plots (Actual vs Predicted):**

1.  Click on the experiment run name (e.g., `XGBoost_Option_Tune` or `LinearRegression`).
2.  Click the **"Artifacts"** tab.
3.  Looks for the following image files:
    - `linear_regression_actual_vs_predicted.png`
    - `xgboost_untuned_actual_vs_predicted.png`
    - `xgboost_actual_vs_predicted.png`

These plots provide a visual comparison between model predictions and actual data.

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
