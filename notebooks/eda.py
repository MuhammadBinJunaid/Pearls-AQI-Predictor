"""
Exploratory Data Analysis — AQI Predictor
============================================
Pulls the accumulated feature data from Hopsworks and analyzes it:
- Basic shape / missing values
- AQI distribution and hazard-band breakdown
- Trends by hour / day of week / city
- Correlations between pollutants, weather, and AQI

Run with:
    python notebooks/eda.py
Outputs a few PNG charts into notebooks/eda_outputs/ and prints a summary.
"""

import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

OUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    import hopsworks

    project = hopsworks.login(
        api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
        project=os.environ.get("HOPSWORKS_PROJECT"),
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=4)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["city", "timestamp"]).reset_index(drop=True)


def basic_summary(df: pd.DataFrame):
    print("=" * 60)
    print("SHAPE & MISSING VALUES")
    print("=" * 60)
    print(f"Total rows: {len(df):,}")
    print(f"Cities: {df['city'].nunique()} -> {sorted(df['city'].unique())}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\nMissing values per column:")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")
    if missing.sum() == 0:
        print("  None -- clean dataset.")


def aqi_distribution(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("AQI DISTRIBUTION")
    print("=" * 60)
    print(df["aqi"].describe().round(1))

    bands = [
        (0, 50, "Good"), (51, 100, "Moderate"), (101, 150, "Unhealthy (Sensitive)"),
        (151, 200, "Unhealthy"), (201, 300, "Very Unhealthy"), (301, 500, "Hazardous"),
    ]
    print("\nAQI band breakdown (% of readings):")
    for lo, hi, label in bands:
        pct = ((df["aqi"] >= lo) & (df["aqi"] <= hi)).mean() * 100
        print(f"  {label:25s} ({lo}-{hi}): {pct:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    df["aqi"].hist(bins=40, ax=ax, color="#4FA8D8", edgecolor="white")
    ax.set_title("AQI Distribution — All Cities, All Readings")
    ax.set_xlabel("AQI")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_distribution.png"), dpi=120)
    plt.close()


def city_comparison(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("AVERAGE AQI BY CITY")
    print("=" * 60)
    city_avg = df.groupby("city")["aqi"].mean().sort_values(ascending=False).round(1)
    print(city_avg)

    fig, ax = plt.subplots(figsize=(9, 6))
    city_avg.plot(kind="barh", ax=ax, color="#E8974A")
    ax.set_title("Average AQI by City")
    ax.set_xlabel("Average AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "avg_aqi_by_city.png"), dpi=120)
    plt.close()


def time_trends(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("AQI BY HOUR OF DAY (all cities combined)")
    print("=" * 60)
    hourly = df.groupby("hour")["aqi"].mean().round(1)
    print(hourly)

    fig, ax = plt.subplots(figsize=(9, 5))
    hourly.plot(kind="line", marker="o", ax=ax, color="#4FA8D8")
    ax.set_title("Average AQI by Hour of Day")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Average AQI")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_by_hour.png"), dpi=120)
    plt.close()

    print("\nAQI BY DAY OF WEEK (0=Mon .. 6=Sun)")
    dow = df.groupby("day_of_week")["aqi"].mean().round(1)
    print(dow)


def correlations(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("CORRELATION WITH AQI")
    print("=" * 60)
    numeric_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                     "temperature", "humidity", "pressure", "wind_speed", "hour"]
    corr = df[numeric_cols].corr()["aqi"].drop("aqi").sort_values(ascending=False)
    print(corr.round(3))

    fig, ax = plt.subplots(figsize=(8, 6))
    corr.plot(kind="barh", ax=ax, color="#4CAF7D")
    ax.set_title("Correlation of Features with AQI")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlations.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    print("Loading data from Hopsworks...")
    df = load_data()

    basic_summary(df)
    aqi_distribution(df)
    city_comparison(df)
    time_trends(df)
    correlations(df)

    print("\n" + "=" * 60)
    print(f"Charts saved to: {OUT_DIR}")
    print("=" * 60)
