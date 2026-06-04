#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ==========================================
// 1. Wifi & Server Configuration
// ==========================================
const char* ssid = "AndroidAP_1281";
const char* password = "Khaapr@2006";

// Replace with your Flask Server's local IP address (e.g. 192.168.1.10)
const char* serverUrl = "http://10.159.96.88:5000/predict";

// ==========================================
// 2. Hardware Pin Setup
// ==========================================
// DHT11 setup
#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Soil Moisture sensor setup
#define SOIL_MOISTURE_PIN 34

// Calibration Values manually determined for your specific soil moisture sensor
// ESP32 has a 12-bit ADC so values range from 0 - 4095
const int dryValue = 4095;   // Sensor in dry air
const int wetValue = 1500;   // Sensor submerged in water

// Timer variables
unsigned long lastTime = 0;
// Set timer to 30000ms (30 seconds)
unsigned long timerDelay = 30000;

void setup() {
  Serial.begin(115200);
  
  // Initialize DHT sensor
  dht.begin();
  
  // Initialize WiFi connection
  WiFi.mode(WIFI_STA); // Set to Station mode
  WiFi.disconnect();    // Reset previous connections
  delay(100);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  int attempts = 0;
  while(WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(1000);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("");
    Serial.print("Super! Connected to WiFi. Node IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("");
    Serial.println("WiFi Connection Failed! Please check:");
    Serial.println("1. Is your hotspot/router name (SSID) EXACTLY 'AndroidAP_1281'?");
    Serial.println("2. Is your password EXACTLY 'Khaapr@2006'?");
    Serial.println("3. Is your phone/laptop close to the ESP32?");
  }
}

void loop() {
  // Send an HTTP POST request every timerDelay (30 seconds)
  if ((millis() - lastTime) > timerDelay) {
    
    // Ensure we are still connected to WiFi
    if(WiFi.status() == WL_CONNECTED) {
      HTTPClient http;

      // 1. READ SENSORS
      float t = dht.readTemperature();
      float h = dht.readHumidity();
      int soilAnalog = analogRead(SOIL_MOISTURE_PIN);
      
      // Convert 12-bit analog soil reading to percentage (0 - 100%)
      // Because typically wet = lower resistance = lower ADC value
      int soilMoisturePercent = map(soilAnalog, dryValue, wetValue, 0, 100);
      soilMoisturePercent = constrain(soilMoisturePercent, 0, 100);

      // Check if DHT read failed
      if (isnan(h) || isnan(t)) {
        Serial.println("Failed to read from DHT sensor!");
        return; // Skip rest of the loop and try again later
      }

      // Print debug values
      Serial.println("---------------------------------");
      Serial.print("Temperature:   "); Serial.print(t); Serial.println(" °C");
      Serial.print("Humidity:      "); Serial.print(h); Serial.println(" %");
      Serial.print("Soil Moisture: "); Serial.print(soilMoisturePercent); Serial.println(" %");

      // 2. CONSTRUCT JSON PAYLOAD
      /* 
       * Size 200 is sufficient for standard payloads in ArduinoJson v6. 
       * If you use ArduinoJson v7, you can simply use: JsonDocument jsonDoc;
       */
      JsonDocument jsonDoc;
      jsonDoc["temperature"] = t;
      jsonDoc["humidity"] = h;
      jsonDoc["soil_moisture"] = soilMoisturePercent;
      
      // Vegetation indices (NDVI, NDRE, SAVI, EVI, GNDVI) are now
      // dynamically estimated by the backend using the real sensor
      // readings below, so we no longer send static values.

      String jsonString;
      serializeJson(jsonDoc, jsonString);

      // 3. SEND POST REQUEST TO FLASK BACKEND
      http.begin(serverUrl);
      http.addHeader("Content-Type", "application/json");
      
      Serial.println("Sending data to Cloud/Server...");
      int httpResponseCode = http.POST(jsonString);

      // 4. READ & PRINT RESPONSE
      if (httpResponseCode > 0) {
        Serial.print("HTTP Success Code: ");
        Serial.println(httpResponseCode);
        
        String payload = http.getString();
        Serial.println("Dashboard DSS Response:");
        Serial.println(payload);
      } else {
        Serial.print("Prediction API Error Code: ");
        Serial.println(httpResponseCode);
        Serial.println(http.errorToString(httpResponseCode).c_str());
      }
      
      // Free network resources
      http.end(); 
      Serial.println("---------------------------------\n");
      
    } else {
      Serial.println("WiFi Disconnected. Reconnecting...");
      WiFi.reconnect();
    }
    
    // Reset timer
    lastTime = millis();
  }
}
