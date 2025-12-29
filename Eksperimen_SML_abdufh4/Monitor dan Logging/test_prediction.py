import requests
import json

url = 'http://localhost:5001/predict'
import time

# Initial values
temp = 20.0
wind_speed = 0.0
sky_cover = 0.0
visibility = 0.0
humidity = 20.0

print("Starting linear data simulation (Press Ctrl+C to stop)...")

while True:
    # Linear increments
    temp += 0.5
    wind_speed += 0.2
    sky_cover += 1.0
    visibility += 0.5
    humidity += 1.0
    
    # Reset if too high
    if temp > 40: temp = 20
    if sky_cover > 100: sky_cover = 0
    if humidity > 100: humidity = 20

    payload = {
        "Temp": temp,
        "WindSpeed": wind_speed,
        "SkyCover": sky_cover,
        "Visibility": visibility,
        "Humidity": humidity,
        "IsDaylight": 1
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Sent: Temp={temp:.1f}, Sky={sky_cover:.1f} -> Status: {response.status_code}, Res: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        
    time.sleep(2)
