import numpy as np
import pandas as pd
import os

def generate_data(n_samples=600):
    # Set seed for reproducibility
    np.random.seed(42)

    # 1. Independent features
    # Temperature varies from 20 to 45 °C
    temperature = np.random.normal(loc=30.0, scale=4.0, size=n_samples)
    temperature = np.clip(temperature, 20.0, 45.0)

    # Humidity varies from 30% to 90%
    humidity = np.random.normal(loc=60.0, scale=12.0, size=n_samples)
    humidity = np.clip(humidity, 30.0, 90.0)

    # Soil moisture varies from 10% to 60%
    soil_moisture = np.random.normal(loc=35.0, scale=10.0, size=n_samples)
    soil_moisture = np.clip(soil_moisture, 10.0, 60.0)

    # 2. Vegetation Indices
    # Base NDVI (Normalized Difference Vegetation Index) varying from 0.2 to 0.9
    ndvi = np.random.normal(loc=0.6, scale=0.15, size=n_samples)
    ndvi = np.clip(ndvi, 0.2, 0.9)

    # Other indices are generally correlated with NDVI in agricultural context
    ndre = ndvi * 0.8 + np.random.normal(0, 0.05, n_samples)
    savi = ndvi * 1.1 + np.random.normal(0, 0.05, n_samples)
    evi = ndvi * 1.5 + np.random.normal(0, 0.1, n_samples)
    gndvi = ndvi * 0.9 + np.random.normal(0, 0.05, n_samples)
    
    # Clip to realistic bounds
    ndre = np.clip(ndre, 0.1, 0.8)
    savi = np.clip(savi, 0.1, 1.0)
    evi = np.clip(evi, 0.1, 1.5)
    gndvi = np.clip(gndvi, 0.1, 0.9)

    # 3. Biomass (kg/ha)
    # Biomass is heavily dependent on plant health (NDVI) and water availability (SoilMoisture)
    # Base biomass + (NDVI effect) + (SoilMoisture effect) + noise
    biomass = 1500 + (ndvi * 4000) + (soil_moisture * 60) + np.random.normal(0, 300, n_samples)
    
    # 4. Yield (kg/ha)
    # Cotton yield is a fraction of total biomass (Harvest Index) + some variability
    # Ideal conditions result in better yield
    harvest_index = np.random.normal(loc=0.35, scale=0.05, size=n_samples)
    harvest_index = np.clip(harvest_index, 0.2, 0.5)
    
    yield_data = biomass * harvest_index + np.random.normal(0, 100, n_samples)
    yield_data = np.clip(yield_data, 300, 3500) # clip to realistic yield limits

    # Create DataFrame
    data = pd.DataFrame({
        'NDVI': np.round(ndvi, 3),
        'NDRE': np.round(ndre, 3),
        'SAVI': np.round(savi, 3),
        'EVI': np.round(evi, 3),
        'GNDVI': np.round(gndvi, 3),
        'Temperature': np.round(temperature, 1),
        'Humidity': np.round(humidity, 1),
        'SoilMoisture': np.round(soil_moisture, 1),
        'Biomass': np.round(biomass, 1),
        'Yield': np.round(yield_data, 1)
    })

    return data

if __name__ == "__main__":
    df = generate_data()
    
    # Save to the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'final_dataset.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Successfully generated 600 rows of synthetic cotton data!")
    print(f"Dataset saved to: {output_path}")
    print("\nFirst few rows:")
    print(df.head())
