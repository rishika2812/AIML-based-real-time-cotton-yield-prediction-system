import ee
import flask
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime

# ==========================================
# 1. GEE Authentication and Initialization
# ==========================================
try:
    print("Initializing Google Earth Engine...")
    # Trigger authentication flow. 
    # Note: Requires manual browser interaction on first run.
    # ee.Authenticate() 
    ee.Initialize(project='gen-lang-client-0050929624')
    print("GEE Initialized successfully with project: gen-lang-client-0050929624")
except Exception as e:
    print(f"GEE Initialization failed: {e}")
    print("Try running 'ee.Authenticate()' manually in a terminal first.")

app = Flask(__name__)

# ==========================================
# 2. Paths and Model Loading
# ==========================================
# The model.pkl is the Random Forest model trained for Biomass
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler_X.pkl'

model = None
scaler = None

def load_assets():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        print(f"Error loading assets: {e}")

# ==========================================
# 3. GEE Index Calculation Logic
# ==========================================
def get_sentinel_indices(lat, lon, start_date, end_date):
    """
    Fetches the most recent Sentinel-2 image and calculates 5 vegetation indices.
    """
    point = ee.Geometry.Point([lon, lat])
    
    # Fetch Sentinel-2 Surface Reflectance imagery
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(point)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .sort('system:time_start', False)) # Get most recent
    
    image = collection.first()
    
    if image is None:
        return None, "No image found for given date range and cloud cover filter."

    # Sentinel-2 Band Mapping:
    # B2: Blue, B3: Green, B4: Red, B5: Red Edge 1, B8: NIR
    b2 = image.select('B2')
    b3 = image.select('B3')
    b4 = image.select('B4')
    b5 = image.select('B5')
    b8 = image.select('B8')

    # NDVI = (B8 - B4) / (B8 + B4)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # NDRE = (B8 - B5) / (B8 + B5)
    ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE')
    
    # SAVI = ((B8 - B4) / (B8 + B4 + 0.5)) * 1.5
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {'NIR': b8, 'RED': b4}
    ).rename('SAVI')
    
    # EVI = 2.5 * ((B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1))
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': b8, 'RED': b4, 'BLUE': b2}
    ).rename('EVI')
    
    # GNDVI = (B8 - B3) / (B8 + B3)
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')

    # Combine indices into a single image
    indices_img = ee.Image([ndvi, ndre, savi, evi, gndvi])
    
    # Extract values at the given point
    stats = indices_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10
    ).getInfo()
    
    return stats, None

# ==========================================
# 4. API Endpoints
# ==========================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "GEE Backend Running", "project": "gen-lang-client-0050929624"})

@app.route('/get_indices', methods=['POST'])
def get_indices():
    """
    Returns the 5 vegetation indices for a point and date range.
    """
    data = request.get_json()
    
    # Missing field validation
    required = ['lat', 'lon', 'start_date', 'end_date']
    if not all(k in data for k in required):
        return jsonify({"error": f"Missing input fields. Required: {required}"}), 400
    
    try:
        stats, error = get_sentinel_indices(
            data['lat'], data['lon'], 
            data['start_date'], data['end_date']
        )
        
        if error:
            return jsonify({"error": error}), 404
            
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Combines GEE satellite indices with hardware data for a biomass prediction.
    """
    data = request.get_json()
    
    # Validation
    required = ['lat', 'lon', 'start_date', 'end_date', 'temperature', 'humidity', 'soil_moisture']
    if not all(k in data for k in required):
        return jsonify({"error": f"Missing input fields. Required: {required}"}), 400

    if model is None or scaler is None:
        return jsonify({"error": "ML model or scaler not loaded on server."}), 500

    try:
        # 1. Fetch indices from GEE
        stats, error = get_sentinel_indices(
            data['lat'], data['lon'], 
            data['start_date'], data['end_date']
        )
        
        if error:
            return jsonify({"error": f"GEE Error: {error}"}), 404

        # 2. Extract values and combine with hardware data
        # Exact order: [NDVI, NDRE, SAVI, EVI, GNDVI, temperature, humidity, soil_moisture]
        feature_list = [
            stats['NDVI'],
            stats['NDRE'],
            stats['SAVI'],
            stats['EVI'],
            stats['GNDVI'],
            float(data['temperature']),
            float(data['humidity']),
            float(data['soil_moisture'])
        ]
        
        # 3. Model Prediction
        features_np = np.array([feature_list])
        scaled_features = scaler.transform(features_np)
        prediction = model.predict(scaled_features)
        
        # 4. Return result
        return jsonify({
            "prediction_kg_ha": round(float(prediction[0]), 2),
            "indices_used": stats,
            "hardware_data_used": {
                "temp": data['temperature'],
                "hum": data['humidity'],
                "soil": data['soil_moisture']
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_assets()
    # Run on all interfaces on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
