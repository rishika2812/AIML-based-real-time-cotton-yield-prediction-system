import requests
import time
import json

BASE_URL = 'http://127.0.0.1:5000/predict'

scenarios = [
    {
        "name": "Excellent Conditions",
        "data": {'temperature': 28, 'humidity': 55, 'soil_moisture': 50, 'ndvi': 0.72, 'ndre': 0.45, 'savi': 0.68, 'evi': 0.61, 'gndvi': 0.58}
    },
    {
        "name": "Heat & Water Stress",
        "data": {'temperature': 40, 'humidity': 35, 'soil_moisture': 20, 'ndvi': 0.68, 'ndre': 0.40, 'savi': 0.60, 'evi': 0.55, 'gndvi': 0.52}
    },
    {
        "name": "Low Chlorophyll/Nitrogen",
        "data": {'temperature': 30, 'humidity': 60, 'soil_moisture': 50, 'ndvi': 0.55, 'ndre': 0.20, 'savi': 0.55, 'evi': 0.50, 'gndvi': 0.45}
    }
]

for s in scenarios:
    print(f"Testing Scenario: {s['name']}")
    try:
        res = requests.post(BASE_URL, json=s['data'])
        print(json.dumps(res.json(), indent=2))
        print("-" * 30)
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(5)
