import requests
import json

url = "http://127.0.0.1:5000/get_indices"
data = {
    "lat": 17.3850,
    "lon": 78.4867,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
