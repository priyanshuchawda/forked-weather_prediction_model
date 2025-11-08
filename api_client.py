import requests
from datetime import datetime
import pandas as pd
from config import get_api_key

class OpenWeatherAPI:
    def __init__(self):
        self.api_key = get_api_key()
        if not self.api_key:
            raise ValueError("API key for OpenWeather is not set in the environment variables.")
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_coordinates_for_city(self, city_name=None, lat=None, lon=None):
        """Get coordinates (lat, lon) for a given city name or use provided coordinates"""
        if lat is not None and lon is not None:
            return {"lat": lat, "lon": lon}
        
        if city_name is None:
            return None

        endpoint = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": city_name,
            "limit": 1,
            "appid": self.api_key,
        }
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if data:
                return {"lat": data[0]["lat"], "lon": data[0]["lon"]}
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching coordinates: {e}")
            return None

    def get_current_weather(self, lat, lon):
        """Get current weather from OpenWeather API"""
        endpoint = f"{self.base_url}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            weather_data = {
                'date': datetime.fromtimestamp(data['dt']).date(),
                'temp': data['main']['temp'],
                'rain': data.get('rain', {}).get('1h', 0),
                'wind': data['wind']['speed'] * 3.6
            }
            return weather_data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather: {e}")
            return None

    def get_5_day_forecast(self, lat, lon):
        """Get 5-day forecast from OpenWeather API"""
        endpoint = f"{self.base_url}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",  # Use Celsius
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Process forecast data
            forecasts = []
            for item in data['list']:
                dt = datetime.fromtimestamp(item['dt'])
                temp = item['main']['temp']
                rain = item.get('rain', {}).get('3h', 0)  # Rain in mm for 3 hours
                wind = item['wind']['speed'] * 3.6  # Convert m/s to km/h

                forecasts.append({
                    'date': dt.date(),
                    'temp': temp,
                    'rain': rain,
                    'wind': wind
                })

            # Group by date and calculate daily averages
            df = pd.DataFrame(forecasts)
            daily = df.groupby('date').agg({
                'temp': 'mean',
                'rain': 'sum',
                'wind': 'mean'
            }).reset_index()

            return daily

        except requests.exceptions.RequestException as e:
            print(f"Error fetching OpenWeather data: {e}")
            return None
