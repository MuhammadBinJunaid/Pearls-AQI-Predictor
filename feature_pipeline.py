"""
AQI Feature Pipeline — multi-city (Pakistan), Open-Meteo-based
=================================================================
Fetches pollutant data (Open-Meteo Air Quality API) and weather data
(Open-Meteo Historical/Forecast Weather API) for multiple Pakistani
cities. NO API KEY REQUIRED for either endpoint. Engineers features
and stores them in the Hopsworks Feature Store.

Two modes:
  --mode live      -> fetch today's hourly data for all configured cities (run hourly)
  --mode backfill  -> fetch N days of real historical data per city (years available)

Usage:
    python feature_pipeline.py --mode live
    python feature_pipeline.py --mode backfill --days 730   # ~2 years
"""

import os
import argparse
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT")

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# 19 Pakistani cities with coordinates (lat, lon)
PAKISTAN_CITIES = {
    "karachi":     (24.8607, 67.0011),
    "lahore":      (31.5497, 74.3436),
    "islamabad":   (33.6844, 73.0479),
    "rawalpindi":  (33.5651, 73.0169),
    "faisalabad":  (31.4504, 73.1350),
    "multan":      (30.1575, 71.5249),
    "peshawar":    (34.0151, 71.5249),
    "quetta":      (30.1798, 66.9750),
    "sialkot":     (32.4945, 74.5229),
    "gujranwala":  (32.1877, 74.1945),
    "hyderabad":   (25.3960, 68.3578),
    "sukkur":      (27.7052, 68.8574),
    "bahawalpur":  (29.3956, 71.6836),
    "sargodha":    (32.0836, 72.6711),
    "sheikhupura": (31.7167, 73.9850),
    "larkana":     (27.5590, 68.2120),
    "gujrat":      (32.5740, 74.0789),
    "mardan":      (34.1989, 72.0404),
    "kasur":       (31.1156, 74.4502),
}

cities_env = os.environ.get("AQI_CITIES")
CITIES = [c.strip() for c in cities_env.split(",")] if cities_env else list(PAKISTAN_CITIES.keys())


# ---------------------------------------------------------------------
# STEP 1: FETCH RAW DATA (Open-Meteo, no key needed)
# ---------------------------------------------------------------------
def fetch_air_quality(city: str, start_date: str, end_date: str) -> dict:
    """Fetch hourly pollutant + US AQI data for one city over a date range."""
    lat, lon = PAKISTAN_CITIES[city]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_weather(city: str, start_date: str, end_date: str, use_archive: bool) -> dict:
    """Fetch hourly weather data for one city over a date range.
    Uses the archive endpoint for past dates, forecast endpoint for today/recent."""
    lat, lon = PAKISTAN_CITIES[city]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    url = WEATHER_ARCHIVE_URL if use_archive else WEATHER_FORECAST_URL
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------
# STEP 2: COMPUTE FEATURES — merge AQ + weather hourly series into rows
# ---------------------------------------------------------------------
def build_feature_rows(city: str, aq_json: dict, weather_json: dict) -> list:
    """Merge hourly air quality and weather series (matched by timestamp) into feature rows."""
    aq_hourly = aq_json.get("hourly", {})
    w_hourly = weather_json.get("hourly", {})

    aq_times = aq_hourly.get("time", [])
    w_times = w_hourly.get("time", [])

    weather_by_time = {
        t: {
            "temperature": w_hourly["temperature_2m"][i],
            "humidity": w_hourly["relative_humidity_2m"][i],
            "pressure": w_hourly["surface_pressure"][i],
            "wind_speed": w_hourly["wind_speed_10m"][i],
        }
        for i, t in enumerate(w_times)
    }

    rows = []
    prev_aqi = None
    for i, t in enumerate(aq_times):
        dt = datetime.fromisoformat(t)
        us_aqi = aq_hourly.get("us_aqi", [None] * len(aq_times))[i]
        aqi = float(us_aqi) if us_aqi is not None else np.nan

        w = weather_by_time.get(t, {})

        row = {
            "city": city,
            "timestamp": pd.Timestamp(dt),
            "aqi": aqi,
            "pm25": aq_hourly.get("pm2_5", [None] * len(aq_times))[i] or np.nan,
            "pm10": aq_hourly.get("pm10", [None] * len(aq_times))[i] or np.nan,
            "o3": aq_hourly.get("ozone", [None] * len(aq_times))[i] or np.nan,
            "no2": aq_hourly.get("nitrogen_dioxide", [None] * len(aq_times))[i] or np.nan,
            "so2": aq_hourly.get("sulphur_dioxide", [None] * len(aq_times))[i] or np.nan,
            "co": aq_hourly.get("carbon_monoxide", [None] * len(aq_times))[i] or np.nan,
            "temperature": w.get("temperature", np.nan),
            "humidity": w.get("humidity", np.nan),
            "pressure": w.get("pressure", np.nan),
            "wind_speed": w.get("wind_speed", np.nan),
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
            "day": dt.day,
            "month": dt.month,
            "aqi_change_rate": (aqi - prev_aqi) if (prev_aqi is not None and not np.isnan(aqi)) else 0.0,
        }
        prev_aqi = aqi if not np.isnan(aqi) else prev_aqi
        rows.append(row)

    return rows


# ---------------------------------------------------------------------
# STEP 3: STORE IN HOPSWORKS FEATURE STORE
# ---------------------------------------------------------------------
def store_features(df: pd.DataFrame):
    """Push a DataFrame of feature rows (any number of cities) into Hopsworks."""
    import hopsworks

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name="aqi_features",
        version=4,  # v4: Open-Meteo based, no API key needed, full Pakistan coverage
        description="Hourly AQI + weather features for forecasting, 19 Pakistani cities (Open-Meteo)",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )
    feature_group.insert(df)
    print(f"Inserted {len(df)} row(s) across {df['city'].nunique()} cities into 'aqi_features' (v4).")


# ---------------------------------------------------------------------
# MODES
# ---------------------------------------------------------------------
def run_live():
    """Fetch today's hourly readings per configured city and store them."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    all_rows = []

    for city in CITIES:
        try:
            aq = fetch_air_quality(city, today, today)
            weather = fetch_weather(city, today, today, use_archive=False)
            rows = build_feature_rows(city, aq, weather)
            all_rows.extend(rows)
            latest_aqi = rows[-1]["aqi"] if rows else "n/a"
            print(f"  ok {city}: {len(rows)} hourly readings today, latest AQI {latest_aqi}")
        except Exception as e:
            print(f"  FAILED {city}: {e}")
        time.sleep(0.5)

    if not all_rows:
        print("No cities fetched successfully -- nothing to store.")
        return

    df = pd.DataFrame(all_rows)
    print(df.tail())
    store_features(df)


def run_backfill(days: int):
    """
    Pulls `days` days of real historical data per city from Open-Meteo
    (both Air Quality and Historical Weather APIs -- no key required)
    and stores it in Hopsworks. Fetches in ~90-day chunks per city.
    """
    end_dt = datetime.utcnow().date()
    start_dt = end_dt - timedelta(days=days)

    all_rows = []
    chunk_days = 90

    for city in CITIES:
        print(f"\nBackfilling {city} from {start_dt} to {end_dt}...")
        cursor = start_dt
        city_row_count = 0
        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
            cs, ce = cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
            try:
                aq = fetch_air_quality(city, cs, ce)
                weather = fetch_weather(city, cs, ce, use_archive=True)
                rows = build_feature_rows(city, aq, weather)
                all_rows.extend(rows)
                city_row_count += len(rows)
                print(f"  {cs} to {ce}: {len(rows)} hourly readings")
            except Exception as e:
                print(f"  FAILED chunk {cs} to {ce}: {e}")
            cursor = chunk_end
            time.sleep(0.5)
        print(f"  -> {city_row_count} total historical readings for {city}")

    if not all_rows:
        print("No historical rows collected -- nothing to store.")
        return

    df = pd.DataFrame(all_rows)
    print(f"\nTotal historical rows across all cities: {len(df)}")
    store_features(df)


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "backfill"], default="live")
    parser.add_argument("--days", type=int, default=730, help="days of history to backfill (default ~2 years)")
    args = parser.parse_args()

    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        raise SystemExit("Missing HOPSWORKS_API_KEY / HOPSWORKS_PROJECT environment variables.")

    print(f"Cities configured ({len(CITIES)}): {', '.join(CITIES)}")

    if args.mode == "live":
        run_live()
    else:
        run_backfill(args.days)
