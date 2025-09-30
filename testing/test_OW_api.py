import os
import requests
from dotenv import load_dotenv

load_dotenv()  # this loads the .env file
open_weather_key = os.getenv("OPENWEATHER_API_KEY")

print(open_weather_key)

# testing with the coordinates of Karachi
lat, lon = 24.8607, 67.0011

url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={open_weather_key}"


response = requests.get(url)
data = response.json()

print("Raw JSON data: \n------------\n", data)

print("\nParsed output:")
air = data["list"][0]
components = air["components"]
aqi = air["main"]["aqi"]

print(f"AQI index: {aqi} (1=Good, 5=Very Poor)")
print("Pollutant concentrations (μg/m³):")
for pollutant, value in components.items():
    print(f"  {pollutant}: {value}")