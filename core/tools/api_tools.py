# ==== core/tools/api_tools.py ====
"""
Agentic Tools: External Data APIs

Functions that fetch real-time information from free or local external sources.
"""

import requests
from datetime import datetime
from core.utils.logger import get_logger

logger = get_logger(__name__)

def get_current_time() -> str:
    """Fetch the exact current date and time."""
    now = datetime.now()
    return now.strftime("The current date and time is %Y-%m-%d %H:%M:%S.")

def get_weather(location: str) -> str:
    """Fetch the current weather for a specific location using a free API.

    Args:
        location: The name of the city (e.g., "London", "New York").

    Returns:
        A string describing the current weather.
    """
    try:
        # Step 1: Geocode the location (Free Geocoding API)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        
        if not geo_data.get("results"):
            return f"Could not find coordinates for {location}."
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        city = geo_data["results"][0].get("name", location)

        # Step 2: Fetch weather at coordinates (Free Open-Meteo API)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_resp = requests.get(weather_url, timeout=10)
        w_resp.raise_for_status()
        w_data = w_resp.json()
        
        current = w_data.get("current_weather", {})
        temp = current.get("temperature", "Unknown")
        wind = current.get("windspeed", "Unknown")
        
        return f"The current weather in {city} is {temp}°C with wind speeds of {wind} km/h."
        
    except Exception as exc:
        logger.error("Failed to get weather for %s: %s", location, exc)
        return f"Failed to retrieve weather for {location} due to an API error."
