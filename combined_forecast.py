from datetime import datetime
import joblib
from api_client import OpenWeatherAPI
from weather_model_enhanced import forecast_next_7_days
from config import PUNE_COORDINATES

from datetime import datetime
from api_client import OpenWeatherAPI
from weather_model_enhanced import forecast_next_7_days
from config import PUNE_COORDINATES

def combine_forecasts(model_forecast, api_forecast):
    """
    Combine forecasts from our model and OpenWeather API
    using weighted averaging based on each model's historical performance
    """
    # Weights based on R² scores (can be adjusted)
    weights = {
        'temp': {'model': 0.6, 'api': 0.4},  # Our model is very good with temperature
        'rain': {'model': 0.3, 'api': 0.7},  # API might be better with precipitation
        'wind': {'model': 0.5, 'api': 0.5}   # Equal weights for wind
    }
    
    combined = []
    for i in range(min(len(model_forecast), len(api_forecast))):
        model_day = model_forecast[i]
        api_day = api_forecast.iloc[i]
        
        # Combine predictions using weights
        temp = (weights['temp']['model'] * model_day['temp'] +
                weights['temp']['api'] * api_day['temp'])
        rain = (weights['rain']['model'] * model_day['prcp'] +
                weights['rain']['api'] * api_day['rain'])
        wind = (weights['wind']['model'] * model_day['wspd'] +
                weights['wind']['api'] * api_day['wind'])
        
        combined.append({
            'date': api_day['date'],
            'temperature': round(temp, 1),
            'rainfall': round(rain, 1),
            'wind_speed': round(wind, 1)
        })
    
    return combined

def get_weather_forecast(city="Pune", lat=None, lon=None):
    """
    Get the combined 7-day forecast for a given city.
    For Pune, it combines the local model and the OpenWeather API.
    For other cities, it uses only the OpenWeather API.
    """
    weather_api = OpenWeatherAPI()
    
    if city.lower() == "pune" and lat is None and lon is None:
        print("Fetching and combining forecasts for Pune...")
        coordinates = PUNE_COORDINATES
        api_forecast = weather_api.get_5_day_forecast(coordinates["lat"], coordinates["lon"])
        if api_forecast is None:
            print("Failed to get OpenWeather forecast for Pune!")
            return None
        
        model_forecast = forecast_next_7_days(datetime.now())
        combined_forecast = combine_forecasts(model_forecast, api_forecast)
        return combined_forecast
    else:
        print(f"Fetching forecast for {city} from OpenWeather API...")
        if lat is not None and lon is not None:
            coordinates = {"lat": lat, "lon": lon}
        else:
            coordinates = weather_api.get_coordinates_for_city(city)
            
        if coordinates:
            api_forecast = weather_api.get_5_day_forecast(coordinates["lat"], coordinates["lon"])
            if api_forecast is None:
                return None
            # Since we don't have a local model for other cities, we return the API forecast directly
            # We need to rename the columns to match the format of the combined forecast
            api_forecast = api_forecast.rename(columns={'temp': 'temperature', 'rain': 'rainfall', 'wind': 'wind_speed'})
            return api_forecast.to_dict('records')
        else:
            print(f"Could not find coordinates for {city}.")
            return None
