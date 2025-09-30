import os
import requests
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

# loading the API key and target city
ow_api_key = os.getenv("OPENWEATHER_API_KEY")
target_city = os.getenv("TARGET_CITY", "Karachi")

# the coordinates of Karachi
lat, lon = 24.8607, 67.0011
city_coords = {"Karachi": {"lat": lat, "lon": lon}}

def fetch_weather(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={target_city}&appid={ow_api_key}"
    r = requests.get(url)
    r.raise_for_status() # for error handling
    return r.json()

def fetch_pollution_data(city: str):
    coords = city_coords[city]
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={coords['lat']}&lon={coords['lon']}&appid={ow_api_key}"
    r = requests.get(url)
    r.raise_for_status() # for error handling
    return r.json()

# Get the OpenWeatherMap's AQI for the city as well
def fetch_aqi_index(city: str):
    pollution_data = fetch_pollution_data(city)
    aqi_index = pollution_data["list"][0]["main"]["aqi"]
    return {"city": city, "aqi_index": aqi_index}

def save_json(data:dict, prefix: str = "raw_data"):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"raw_data/{prefix}___{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")
    return filename


# main
if __name__ == "__main__":
    weather_data = fetch_weather(target_city)
    pollution_data = fetch_pollution_data(target_city)
    aqi_data = fetch_aqi_index(target_city)

    # Combine all data into one dictionary
    combined_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "weather": weather_data,
        "pollution": pollution_data,
        "aqi_index": aqi_data
    }

    # Save the combined data to a JSON file
    save_json(combined_data, prefix = "karachi_weather_data")