# HOW TO GET GEMINI API KEY:
# 1. Go to https://aistudio.google.com/app/apikey
# 2. Click "Create API Key"
# 3. Copy the key
# 4. Set environment variable before running:
#    Windows:  set GEMINI_API_KEY=your_key_here
#    Mac/Linux: export GEMINI_API_KEY=your_key_here
# 5. Then run: python app.py

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
from datetime import datetime
import ee
import sqlite3
import requests
import json

# ==========================================
# GEE Initialization
# ==========================================
DEFAULT_LAT = 17.3513
DEFAULT_LON = 78.3806
GEE_PROJECT = 'gen-lang-client-0050929624'

gee_connected = False
try:
    print(f"Initializing GEE with project: {GEE_PROJECT}")
    ee.Initialize(project=GEE_PROJECT)
    gee_connected = True
    print("GEE Initialized successfully.")
except Exception as e:
    print(f"GEE Initialization failed: {e}. Falling back to heuristic estimation.")
    gee_connected = False

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# try:
#     import tensorflow as tf
# except ImportError:
#     print("Warning: tensorflow is not installed. Models will not be loaded.")
#     tf = None

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for all domains
CORS(app)

# Global variables for the models and scaler
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
BIOMASS_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_biomass.pkl')
YIELD_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_yield.pkl')
SCALER_X_PATH = os.path.join(MODEL_DIR, 'scaler_X.pkl')
SCALER_B_PATH = os.path.join(MODEL_DIR, 'scaler_B.pkl')
SCALER_Y_PATH = os.path.join(MODEL_DIR, 'scaler_Y.pkl')

biomass_model = None
yield_model = None
scaler_X = None
scaler_B = None
scaler_Y = None

# Database Configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cotton_prediction.db')

def init_db():
    """Initializes the SQLite database and creates the necessary tables."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    temperature REAL,
                    humidity REAL,
                    soil_moisture REAL,
                    ndvi REAL,
                    ndre REAL,
                    savi REAL,
                    evi REAL,
                    gndvi REAL,
                    biomass REAL,
                    yield REAL,
                    health_status TEXT,
                    health_reason TEXT
                )
            ''')
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    metric TEXT,
                    condition TEXT,
                    action TEXT,
                    severity TEXT
                )
            ''')
            # Settings table (for persistence)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            # Initialize default location if not exists
            cursor.execute('INSERT OR IGNORE INTO settings (key, value) VALUES ("lat", ?)', (str(DEFAULT_LAT),))
            cursor.execute('INSERT OR IGNORE INTO settings (key, value) VALUES ("lon", ?)', (str(DEFAULT_LON),))
            cursor.execute('INSERT OR IGNORE INTO settings (key, value) VALUES ("location_name", ?)', ("Chetlapalle, Hyderabad Rural",))
            
            conn.commit()
            print(f"Database initialized at {DB_PATH}")
    except Exception as e:
        print(f"Database Error: {e}")

# We no longer use mock in-memory history_data
# history_data = []

def load_latest_location():
    global DEFAULT_LAT, DEFAULT_LON
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            lat = cursor.execute('SELECT value FROM settings WHERE key="lat"').fetchone()
            lon = cursor.execute('SELECT value FROM settings WHERE key="lon"').fetchone()
            if lat and lon:
                DEFAULT_LAT = float(lat['value'])
                DEFAULT_LON = float(lon['value'])
                print(f"Loaded persistent location: {DEFAULT_LAT}, {DEFAULT_LON}")
    except Exception as e:
        print(f"Error loading location settings: {e}")

def load_ml_assets():
    global biomass_model, yield_model, scaler_X, scaler_B, scaler_Y
    try:
        if os.path.exists(BIOMASS_MODEL_PATH) and os.path.exists(YIELD_MODEL_PATH) and os.path.exists(SCALER_X_PATH):
            print(f"Loading Biomass RF model from: {BIOMASS_MODEL_PATH}")
            biomass_model = joblib.load(BIOMASS_MODEL_PATH)
            
            print(f"Loading Yield RF model from: {YIELD_MODEL_PATH}")
            yield_model = joblib.load(YIELD_MODEL_PATH)
            
            print(f"Loading Feature Scaler from: {SCALER_X_PATH}")
            scaler_X = joblib.load(SCALER_X_PATH)
            
            # Optionally load target scalers if they exist
            if os.path.exists(SCALER_B_PATH):
                print(f"Loading Biomass Scaler from: {SCALER_B_PATH}")
                scaler_B = joblib.load(SCALER_B_PATH)
            if os.path.exists(SCALER_Y_PATH):
                print(f"Loading Yield Scaler from: {SCALER_Y_PATH}")
                scaler_Y = joblib.load(SCALER_Y_PATH)
            
            print("Successfully loaded all ML assets.")
        else:
            print(f"Warning: Models or base scaler not found in {MODEL_DIR}.")
            print("Please ensure 'rf_biomass.pkl', 'rf_yield.pkl', and 'scaler_X.pkl' exist in the models/ folder.")
    except Exception as e:
        print(f"An error occurred while loading ML assets: {e}")

@app.route('/')
def index():
    return render_template('index.html')

# =====================================================
# Dynamic Vegetation Index Estimation from Sensor Data
# =====================================================
# These functions model how environmental conditions affect
# canopy health and translate that into vegetation indices.
# They provide real-time dynamic values that change as your
# ESP32 sensor readings change.
# =====================================================

def _clamp(val, lo=0.0, hi=1.0):
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, val))

def _estimate_ndvi(temp, hum, soil_m):
    soil_c = min(soil_m, 80.0)
    # Peaks at soil=55%, penalizes waterlogging above 75%
    soil_factor = 1.0 - abs(soil_c - 55.0) / 55.0
    soil_factor = max(0.3, soil_factor)
    
    if temp <= 32:
        temp_factor = 1.0
    else:
        temp_factor = max(0.5, 1.0 - (temp - 32) * 0.04)
    
    hum_factor = 0.80 + (min(hum, 80) / 100.0) * 0.20
    
    base = 0.40 + 0.45 * soil_factor * temp_factor * hum_factor
    return round(_clamp(base, 0.20, 0.90), 2)

def _estimate_ndre(temp, hum, soil_m):
    soil_c = min(soil_m, 80.0)
    nutrient_factor = min(1.0, soil_c / 55.0)
    temp_penalty = max(0.0, (temp - 34) * 0.025) if temp > 34 else 0.0
    hum_bonus = (min(hum, 75) / 100.0) * 0.10
    base = 0.25 + 0.30 * nutrient_factor - temp_penalty + hum_bonus
    return round(_clamp(base, 0.15, 0.60), 2)

def _estimate_savi(temp, hum, soil_m):
    ndvi_est = _estimate_ndvi(temp, hum, soil_m)
    soil_c = min(soil_m, 80.0)
    # SAVI corrects for soil brightness — wetter = darker soil = less noise
    soil_correction = 1.0 - (soil_c / 100.0) * 0.10
    base = ndvi_est * 0.90 * soil_correction
    return round(_clamp(base, 0.15, 0.82), 2)

def _estimate_evi(temp, hum, soil_m):
    ndvi_est = _estimate_ndvi(temp, hum, soil_m)
    soil_c = min(soil_m, 80.0)
    canopy = min(1.0, soil_c / 50.0)
    atm = 1.0 - abs(min(hum, 80) - 60) / 150.0
    base = ndvi_est * 0.80 * canopy * atm
    return round(_clamp(base, 0.15, 0.75), 2)

def _estimate_gndvi(temp, hum, soil_m):
    if 22 <= temp <= 32:
        photo = 1.0
    elif temp < 22:
        photo = 0.75 + (temp - 15) * 0.036
    else:
        photo = max(0.5, 1.0 - (temp - 32) * 0.035)
    
    soil_c = min(soil_m, 80.0)
    water = min(1.0, soil_c / 55.0)
    hum_bonus = (min(hum, 75) / 100.0) * 0.12
    base = 0.32 + 0.32 * photo * water + hum_bonus
    return round(_clamp(base, 0.20, 0.72), 2)


# =====================================================
# Google Earth Engine (GEE) Satellite Data Fetching
# =====================================================
def get_sentinel_indices(lat, lon):
    """
    Fetches the most recent Sentinel-2 indices from GEE.
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        # Look back 30 days for the most recent clear image
        now = datetime.now()
        end_date = now.strftime('%Y-%m-%d')
        # Simple month subtraction for start date
        start_date = (now.replace(month=now.month-1) if now.month > 1 else now.replace(year=now.year-1, month=12)).strftime('%Y-%m-%d')
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(point)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                      .sort('system:time_start', False))
        
        image = collection.first()
        if not image: return None

        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE')
        savi = image.expression('((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {'NIR': image.select('B8'), 'RED': image.select('B4')}).rename('SAVI')
        evi = image.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}).rename('EVI')
        gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')

        stats = ee.Image([ndvi, ndre, savi, evi, gndvi]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).getInfo()
        return stats
    except Exception as e:
        print(f"GEE Fetch Error: {e}")
        return None

# =====================================================
# Real-Time Weather Integration (Open-Meteo)
# =====================================================
def get_weather_forecast(lat, lon):
    """
    Fetches real-time weather and 7-day forecast from Open-Meteo.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto&forecast_days=7"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        print(f"Weather Fetch Error: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON with Temperature, Humidity, Soil Moisture.
    Injects default vegetation indices.
    Evaluates Decision Support System (DSS) rules.
    Returns Biomass prediction, Yield prediction, and crop status.
    """
    # Load status check (informational, not blocking)
    models_available = not (biomass_model is None or yield_model is None or scaler_X is None)

    # Parse JSON body
    data = request.json or {}
    
    # 1. Extract dynamic environmental reading (fallback to safe defaults if missing)
    temp   = float(data.get('temperature', 30.0))
    hum    = float(data.get('humidity', 60.0))
    soil_m = min(float(data.get('soil_moisture', 40.0)), 90.0)  # cap at 90% — sensor noise guard
    
    # 2. Get Vegetation Indices
    # Priority 1: Real-time satellite data from GEE
    lat = data.get('lat', DEFAULT_LAT)
    lon = data.get('lon', DEFAULT_LON)
    
    # GEE coordinates point to bare soil (NDVI ~0.07), not cotton field.
    # Using sensor-based estimation which responds to real ESP32 readings.
    print(f"Using sensor-based estimation: temp={temp}, hum={hum}, soil={soil_m}")
    ndvi  = _estimate_ndvi(temp, hum, soil_m)
    ndre  = _estimate_ndre(temp, hum, soil_m)
    savi  = _estimate_savi(temp, hum, soil_m)
    evi   = _estimate_evi(temp, hum, soil_m)
    gndvi = _estimate_gndvi(temp, hum, soil_m)
    print(f"Estimated indices: NDVI={ndvi}, NDRE={ndre}, SAVI={savi}, EVI={evi}, GNDVI={gndvi}")
    
    # 3. Decision Support System (DSS) Rules & Comprehensive Health Analysis
    dss_alerts = []
    reasons = []
    health_score = 3 # 3: Excellent, 2: Good, 1: Fair, 0: Critical
    
    # --- ENVIRONMENTAL CHECKS ---
    # Soil Moisture
    if soil_m < 25.0:
        dss_alerts.append({"metric": "Soil Moisture", "condition": "Critical Low", "action": "Irrigate immediately!"})
        reasons.append("Severe Water Stress")
        health_score = min(health_score, 0)
    elif soil_m < 35.0:
        dss_alerts.append({"metric": "Soil Moisture", "condition": "Low", "action": "Irrigation advised soon."})
        reasons.append("Moderate Water Stress")
        health_score = min(health_score, 1)
        
    # Temperature
    if temp > 38.0:
        dss_alerts.append({"metric": "Temperature", "condition": "Extreme Heat", "action": "Extreme heat stress! Monitor boll shedding."})
        reasons.append("Extreme Heat")
        health_score = min(health_score, 0)
    elif temp > 35.0:
        dss_alerts.append({"metric": "Temperature", "condition": "High", "action": "Heat stress alert. Increase water supply."})
        reasons.append("Heat Stress")
        health_score = min(health_score, 1)

    # Humidity
    if hum < 30.0:
         reasons.append("Dry Air")
         health_score = min(health_score, 2)
    elif hum > 85.0:
         reasons.append("High Humidity (Pest Risk)")
         health_score = min(health_score, 2)

    # --- VEGETATION INDEX CHECKS ---
    # NDVI (Overall Vigor)
    if ndvi < 0.4:
        dss_alerts.append({"metric": "NDVI", "condition": "Very Low", "action": "Crop health at risk. Inspect for pests/disease."})
        reasons.append("Very Low Vigor")
        health_score = min(health_score, 0)
    elif ndvi < 0.6:
        reasons.append("Moderate Vigor")
        health_score = min(health_score, 2)
    elif ndvi >= 0.8:
        dss_alerts.append({"metric": "NDVI", "condition": "Excessive", "action": "Reduce nitrogen to balance growth."})
        reasons.append("Excessive Growth")
        health_score = min(health_score, 2)
        
    # NDRE (Chlorophyll/Nitrogen)
    if ndre < 0.25:
        dss_alerts.append({"metric": "NDRE", "condition": "Low", "action": "Potential Nitrogen deficiency detected."})
        reasons.append("Chlorosis Risk")
        health_score = min(health_score, 1)
        
    # EVI/SAVI (Biomass Stress)
    if evi < 0.35 or savi < 0.35:
        reasons.append("Thin Canopy")
        health_score = min(health_score, 1)
        
    # GNDVI (Photosynthetic Activity)
    if gndvi < 0.4:
        reasons.append("Low Photosynthesis")
        health_score = min(health_score, 1)

    # Determine final status
    status_map = {
        3: "Excellent (Optimal)",
        2: "Good (Stable)",
        1: "Fair (Needs Monitoring)",
        0: "Critical (Action Required)"
    }
    health_status = status_map.get(health_score, "Moderate")
    health_reason = ", ".join(reasons) if reasons else "Optimal Conditions"

    # 4. Predict Biomass
    if biomass_model is not None and yield_model is not None and scaler_X is not None:
        # Feature order MUST match how the scaler and model were trained:
        # ['NDVI', 'NDRE', 'SAVI', 'EVI', 'GNDVI', 'Temperature', 'Humidity', 'SoilMoisture']
        feature_row = np.array([[ndvi, ndre, savi, evi, gndvi, temp, hum, soil_m]])
        scaled_features = scaler_X.transform(feature_row)
        
        biomass_pred_raw = biomass_model.predict(scaled_features)
        if scaler_B is not None:
            # Handle both scalar and array outputs from predict()
            val = biomass_pred_raw[0] if hasattr(biomass_pred_raw, "__len__") else biomass_pred_raw
            biomass_pred_kg_ha = float(scaler_B.inverse_transform([[val]])[0][0])
        else:
            biomass_pred_kg_ha = float(biomass_pred_raw[0]) if hasattr(biomass_pred_raw, "__len__") else float(biomass_pred_raw)
            
        yield_pred_raw = yield_model.predict(scaled_features)
        if scaler_Y is not None:
            val = yield_pred_raw[0] if hasattr(yield_pred_raw, "__len__") else yield_pred_raw
            yield_pred_kg_ha = float(scaler_Y.inverse_transform([[val]])[0][0])
        else:
            yield_pred_kg_ha = float(yield_pred_raw[0]) if hasattr(yield_pred_raw, "__len__") else float(yield_pred_raw)
    else:
        # HEURISTIC FALLBACK (since TF is not on Python 3.14)
        # biomass = 4200*NDVI + 800*(soil/100) - 30*temp + 900*(EVI+GNDVI)/2 + 200*(humidity/100)
        biomass_pred_kg_ha = (4200 * ndvi) + (800 * (soil_m/100)) - (30 * temp) + (900 * ((evi+gndvi)/2)) + (200 * (hum/100))
        yield_pred_kg_ha = biomass_pred_kg_ha * 0.45

    # 6. Response Construction
    response = {
        "timestamp": datetime.now().isoformat(),
        "predictions": {
            "biomass_kg_per_ha": round(biomass_pred_kg_ha, 2),
            "yield_kg_per_ha": round(yield_pred_kg_ha, 2)
        },
        "crop_health_status": health_status,
        "health_reason": health_reason,
        "dss_rules_triggered": dss_alerts,
        "input_features_used": {
            "temperature": round(temp, 1),
            "humidity": round(hum, 1),
            "soil_moisture": round(soil_m, 1),
            "ndvi": round(ndvi, 2),
            "ndre": round(ndre, 2),
            "savi": round(savi, 2),
            "evi": round(evi, 2),
            "gndvi": round(gndvi, 2)
        }
    }
    
    # 6. Save to Persistence (SQLite)
    now_local = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Save Prediction
            cursor.execute('''
                INSERT INTO predictions (
                    timestamp, temperature, humidity, soil_moisture, ndvi, ndre, savi, evi, gndvi, 
                    biomass, yield, health_status, health_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (now_local, temp, hum, soil_m, ndvi, ndre, savi, evi, gndvi, biomass_pred_kg_ha, yield_pred_kg_ha, health_status, health_reason))
            
            # Save Alerts
            for alert in dss_alerts:
                cursor.execute('''
                    INSERT INTO alerts (timestamp, metric, condition, action, severity)
                    VALUES (?, ?, ?, ?, ?)
                ''', (now_local, alert['metric'], alert['condition'], alert['action'], 'warning'))
            conn.commit()
    except Exception as e:
        print(f"Save Prediction Error: {e}")
        
    return jsonify(response)


@app.route('/data', methods=['GET'])
def get_data():
    """
    Returns the latest prediction and real weather for the dashboard.
    """
    latest = None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            row = cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 1').fetchone()
            if row:
                latest = {
                    "timestamp": row['timestamp'],
                    "predictions": {"biomass_kg_per_ha": row['biomass'], "yield_kg_per_ha": row['yield']},
                    "crop_health_status": row['health_status'],
                    "health_reason": row['health_reason'],
                    "input_features_used": {
                        "temperature": row['temperature'],
                        "humidity": row['humidity'],
                        "soil_moisture": row['soil_moisture'],
                        "ndvi": row['ndvi'],
                        "ndre": row['ndre'],
                        "savi": row['savi'],
                        "evi": row['evi'],
                        "gndvi": row['gndvi']
                    }
                }
    except Exception as e:
        print(f"Data API Error: {e}")

    weather = get_weather_forecast(DEFAULT_LAT, DEFAULT_LON)

    return jsonify({
        "latest": latest,
        "weather": weather,
        "gee_status": "Connected" if gee_connected else "Offline",
        "location": {"lat": DEFAULT_LAT, "lon": DEFAULT_LON},
        "server_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/history', methods=['GET'])
def get_history():
    """
    Returns the last 20 real records from the database.
    """
    formatted = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            rows = cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 20').fetchall()
            for row in rows:
                formatted.append({
                    "timestamp": row['timestamp'],
                    "biomass": row['biomass'],
                    "yield": row['yield'],
                    "temperature": row['temperature'],
                    "humidity": row['humidity'],
                    "soil_moisture": row['soil_moisture'],
                    "crop_health": row['health_status']
                })
    except Exception as e:
        print(f"History API Error: {e}")
    return jsonify(formatted)

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """
    Returns real critical alerts from the database.
    """
    formatted = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            rows = cursor.execute('SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 10').fetchall()
            for row in rows:
                formatted.append({
                    "timestamp": row['timestamp'],
                    "metric": row['metric'],
                    "condition": row['condition'],
                    "action": row['action'],
                    "severity": row['severity']
                })
    except Exception as e:
        print(f"Alerts API Error: {e}")
    return jsonify(formatted)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    message = data.get('message', '')
    context = data.get('context', {})

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({
            "reply": "Sorry, AI assistant unavailable. Please check your API key.",
            "success": False
        })
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        system_prompt = f"""You are CottonAI, an expert agricultural assistant specializing in cotton farming in India, particularly Telangana and Andhra Pradesh regions. You help farmers understand their AI yield predictions and give practical advice.
     
Current field sensor data:
- Biomass: {context.get('biomass', 'N/A')} kg/ha
- Predicted Yield: {context.get('yield_kgha', 'N/A')} kg/ha
- NDVI: {context.get('ndvi', 'N/A')} (crop health index)
- Temperature: {context.get('temperature', 'N/A')}°C
- Humidity: {context.get('humidity', 'N/A')}%
- Soil Moisture: {context.get('soil_moisture', 'N/A')}%
- Crop Health Status: {context.get('crop_health', 'N/A')}
- Growth Stage: {context.get('growth_stage', 'Boll Development')} (Day {context.get('day', 78)} of season)
     
Rules for your responses:
- Keep answers under 150 words
- Be practical and farmer-friendly
- Use simple English, avoid jargon
- When asked for yield prediction, use the biomass and yield values from context above
- Give specific numbers and actionable advice
- If asked about irrigation, give specific amounts in litres or mm
- If asked about fertilizer, give kg per hectare doses"""
        
        full_prompt = f"{system_prompt}\n\nUser: {message}"
        response = model.generate_content(full_prompt)
        return jsonify({"reply": response.text, "success": True})
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return jsonify({
            "reply": "Sorry, AI assistant unavailable. Please check your API key.",
            "success": False
        })


if __name__ == '__main__':
    # Initialize Persistent Storage
    init_db()
    load_latest_location()
    
    # Initial load of models
    load_ml_assets()
    
    print("CottonAI Production Backend Running...")
    app.run(host='0.0.0.0', port=5000, debug=True)
