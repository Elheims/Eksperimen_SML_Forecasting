import pandas as pd
import joblib
import mlflow.pyfunc
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Gauge, Counter
import time
from sklearn.preprocessing import MinMaxScaler

# PROMETHEUS METRICS
PREDICTION_LATENCY = Gauge('solar_prediction_latency_seconds', 'Time taken for prediction')
REQUEST_COUNT = Counter('solar_prediction_request_count', 'Total prediction requests')
PREDICTION_VALUE = Gauge('solar_prediction_value_watts', 'Predicted solar power in Watts')

app = Flask(__name__)

# LOAD MODEL & SCALER
try:
    model_path = '../Workflow-CI/MLProject/model_xgboost.pkl'
    scaler_path = '../Workflow-CI/MLProject/namadataset_preprocessing/scaler_y.pkl'

    model = joblib.load(model_path) 
    scaler = joblib.load(scaler_path)
    
    # INIT INPUT SCALER
    # Load raw data
    raw_data_path = '../Eksperimen_SML_Elheims/namadataset_raw/powergeneration.csv'
    df_raw = pd.read_csv(raw_data_path)
    
    # Select features
    feature_cols = ['Temp', 'WindSpeed', 'SkyCover', 'Visibility', 'Humidity', 'IsDaylight']
    X_raw = df_raw[feature_cols]
    
    scaler_X = MinMaxScaler()
    scaler_X.fit(X_raw)
    print("Input Scaler (X) initialized successfully.")
    
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error loading: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data, index=[0])
        
        # START TIMER
        start_time = time.time()

        # REORDER & SCALE INPUTS
        df = df[feature_cols]
        input_scaled = scaler_X.transform(df)

        prediction_scaled = model.predict(input_scaled)
        prediction_final = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
        
        result = float(prediction_final[0][0])
        
        # STOP TIMER & RECORD METRICS
        end_time = time.time()
        latency = end_time - start_time
        
        PREDICTION_LATENCY.set(latency)
        REQUEST_COUNT.inc()
        PREDICTION_VALUE.set(result)
        
        return jsonify({
            "status": "success",
            "prediction_power": result,
            "latency_seconds": latency
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Start Prometheus metrics server on port 8000
    start_http_server(8000)
    # Start Flask app on port 5001
    app.run(host='0.0.0.0', port=5001)