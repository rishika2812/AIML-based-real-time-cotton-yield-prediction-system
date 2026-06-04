import ee
from datetime import datetime

GEE_PROJECT = 'gen-lang-client-0050929624'

def verify_gee():
    print(f"Checking GEE with project: {GEE_PROJECT}")
    try:
        ee.Initialize(project=GEE_PROJECT)
        print("GEE Initialized successfully.")
        
        # Test a simple query (NDVI at a point)
        point = ee.Geometry.Point([78.4867, 17.3850]) 
        image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(point) \
                    .filterDate('2024-01-01', '2025-01-01') \
                    .sort('CLOUDY_PIXEL_PERCENTAGE') \
                    .first()
        
        if image:
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            stats = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).getInfo()
            print(f"GEE Data Fetch Success! NDVI: {stats}")
        else:
            print("GEE Connection OK, but no imagery found.")
            
    except Exception as e:
        print(f"GEE ERROR: {e}")

if __name__ == "__main__":
    verify_gee()
