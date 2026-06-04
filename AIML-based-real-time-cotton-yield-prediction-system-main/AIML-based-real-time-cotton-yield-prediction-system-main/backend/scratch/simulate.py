import requests
import time
import random

print("Simulation started...")
while True:
    try:
        payload = {
            'temperature': random.uniform(28, 35),
            'humidity': random.uniform(50, 70),
            'soil_moisture': random.uniform(40, 60),
            'ndvi': random.uniform(0.65, 0.85)
        }
        requests.post('http://127.0.0.1:5000/predict', json=payload)
    except Exception as e:
        pass
    time.sleep(4)
