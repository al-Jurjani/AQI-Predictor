import json
import os
import random
from datetime import datetime, timedelta

# ----------------------------
# CONFIGURATION
# ----------------------------
SOURCE_FILE = (
    "raw_data/karachi_weather_data___20251015_161703.json"  # your original JSON
)
OUTPUT_DIR = "backfilled_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1️⃣ Perturbation helpers
# ----------------------------


def perturb_value(v, scale=0.1, min_val=0, max_val=None):
    """Perturb a numeric value slightly."""
    if not isinstance(v, (int, float)):
        return v
    delta = random.uniform(-scale, scale) * v
    new_val = v + delta
    if max_val is not None:
        new_val = min(max_val, new_val)
    if min_val is not None:
        new_val = max(min_val, new_val)
    return new_val


def perturb_pollution(pollution, scale=0.15):
    new_pollution = json.loads(json.dumps(pollution))  # deep copy
    for item in new_pollution.get("list", []):
        comps = item.get("components", {})
        for k, v in comps.items():
            comps[k] = perturb_value(v, scale, min_val=0)
    return new_pollution


def perturb_weather(weather, temp_scale=0.02, humidity_scale=0.1, wind_scale=0.15):
    new_weather = json.loads(json.dumps(weather))
    main = new_weather.get("main", {})
    wind = new_weather.get("wind", {})

    if "temp" in main:
        main["temp"] = perturb_value(main["temp"], temp_scale)
    if "feels_like" in main:
        main["feels_like"] = perturb_value(main["feels_like"], temp_scale)
    if "temp_min" in main:
        main["temp_min"] = perturb_value(main["temp_min"], temp_scale)
    if "temp_max" in main:
        main["temp_max"] = perturb_value(main["temp_max"], temp_scale)
    if "humidity" in main:
        main["humidity"] = perturb_value(
            main["humidity"], humidity_scale, min_val=0, max_val=100
        )

    if "speed" in wind:
        wind["speed"] = perturb_value(wind["speed"], wind_scale, min_val=0)

    return new_weather


def perturb_aqi(ow_aqi_obj):
    """Randomly shift AQI index slightly (±1 step but clamp between 1–5)."""
    new_aqi = json.loads(json.dumps(ow_aqi_obj))
    base = new_aqi.get("ow_aqi_index", 3)
    new_aqi["ow_aqi_index"] = int(min(5, max(1, base + random.choice([-1, 0, 1]))))
    return new_aqi


# ----------------------------
# 2️⃣ Backfill generator
# ----------------------------
def generate_backfill(source_file, days=7):
    with open(source_file, "r") as f:
        base = json.load(f)

    for i in range(1, days + 1):
        backfill_date = datetime.utcnow() - timedelta(days=i)

        # perturb key fields
        weather_sim = perturb_weather(base["weather"])
        pollution_sim = perturb_pollution(base["pollution"])
        ow_aqi_sim = perturb_aqi(base["ow_aqi_index"])

        # shift internal timestamps
        pollution_sim["list"][0]["dt"] = int(
            datetime.utcfromtimestamp(pollution_sim["list"][0]["dt"]).timestamp()
            - i * 86400
        )
        weather_sim["dt"] = int(
            datetime.utcfromtimestamp(weather_sim["dt"]).timestamp() - i * 86400
        )

        # construct full JSON (same schema)
        simulated = {
            "timestamp": backfill_date.isoformat(),
            "city": base["city"],
            "weather": weather_sim,
            "pollution": pollution_sim,
            "ow_aqi_index": ow_aqi_sim,
        }

        out_name = (
            f"karachi_backfilled_weather_data__{backfill_date.strftime('%Y%m%d')}.json"
        )
        out_path = os.path.join(OUTPUT_DIR, out_name)

        with open(out_path, "w") as out:
            json.dump(simulated, out, indent=2)

        print(f"💾 Created {out_name}")


# ----------------------------
# 3️⃣ Runner
# ----------------------------
if __name__ == "__main__":
    generate_backfill(SOURCE_FILE, days=7)
