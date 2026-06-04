import requests, json

# Test 1: Good conditions
r1 = requests.post('http://127.0.0.1:5000/predict', json={
    'temperature': 28, 'humidity': 60, 'soil_moisture': 55
})
d1 = r1.json()

# Test 2: Stress conditions
r2 = requests.post('http://127.0.0.1:5000/predict', json={
    'temperature': 38, 'humidity': 40, 'soil_moisture': 20
})
d2 = r2.json()

print("=== GOOD CONDITIONS (28°C, 60%H, 55%Soil) ===")
for k, v in d1['input_features_used'].items():
    print(f"  {k}: {v}")
print(f"  Health: {d1['crop_health_status']}")
print(f"  Reason: {d1['health_reason']}")

print()

print("=== STRESS CONDITIONS (38°C, 40%H, 20%Soil) ===")
for k, v in d2['input_features_used'].items():
    print(f"  {k}: {v}")
print(f"  Health: {d2['crop_health_status']}")
print(f"  Reason: {d2['health_reason']}")
