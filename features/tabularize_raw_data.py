import json
import pandas as pd

def tabularize_raw_data(json_filepath: str) -> pd.DataFrame:
    with open(json_filepath, "r") as file:
        data = json.load(file)

        # extracting the city and timestamp
        city = data.get("city")
        timestamp = data.get("timestamp")

        # getting the weather data
        weather = data.get("weather", {})
        main_weather = weather.get("main", {})

        # getting wind data
        wind = weather.get("wind", {})

        # getting pollution data
        pollution = data.get("pollution", {})
        pollution_components = pollution.get("list", [{}])[0].get("components", {})
        aqi_index = data.get("owm_aqi_index")

        # creating a pd row
        row = {
        "timestamp_utc": timestamp,
        "city": city,

        "temp": main_weather.get("temp"),
        "temp_feels_like": main_weather.get("feels_like"),
        "humidity": main_weather.get("humidity"),
        "pressure": main_weather.get("pressure"),

        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "wind_gust": wind.get("gust"),

        "co": pollution_components.get("co"),
        "no": pollution_components.get("no"),
        "no2": pollution_components.get("no2"),
        "o3": pollution_components.get("o3"),
        "so2": pollution_components.get("so2"),
        "pm2_5": pollution_components.get("pm2_5"),
        "pm10": pollution_components.get("pm10"),
        "nh3": pollution_components.get("nh3"),
        
        "owm_aqi_index": aqi_index,
    }
        
        return pd.DataFrame([row])