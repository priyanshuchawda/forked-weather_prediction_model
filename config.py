import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    """Get the OpenWeather API key from the environment variables."""
    return os.getenv("OPENWEATHER_API_KEY")

def get_gemini_api_key():
    """Get the Gemini API key from the environment variables."""
    return os.getenv("GEMINI_API_KEY")

PUNE_COORDINATES = {"lat": 18.5204, "lon": 73.8567}
