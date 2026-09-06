"""
Simple Local Model — First Proof That This Data Can Predict AQI
====================================================================
Loads data from Hopsworks, builds a 24-hours-ahead AQI target per city,
does a proper CHRONOLOGICAL train/test split (never random-shuffle time
series data), trains a plain Ridge Regression, and evaluates with
RMSE, MAE, and R2 together.

This is intentionally simple -- no Hopsworks Model Registry, no other
model types yet. The goal is just to prove the data is learnable before
building the full training pipeline.

Run with:
    python notebooks/train_simple_model.py
"""

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

FORECAST_HORIZON_HOURS = 24  # predicting AQI 24 hours ahead
FEATURE_COLS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day_of_week", "day", "month", "aqi_change_rate",
]


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


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 'target_aqi' column = AQI value FORECAST_HORIZON_HOURS ahead,
    computed separately per city so we never leak across city boundaries."""
    df = df.copy()
    df["target_aqi"] = df.groupby("city")["aqi"].shift(-FORECAST_HORIZON_HOURS)
    # rows near the end of each city's series have no future value -- drop them
    df = df.dropna(subset=["target_aqi"]).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Split by TIME, not randomly. All rows before the cutoff timestamp go to
    train; everything after goes to test. This is the correct way to
    validate time-series models -- random shuffling would leak future
    information into training and give a falsely optimistic score.
    """
    cutoff = df["timestamp"].quantile(train_frac)
    train = df[df["timestamp"] <= cutoff]
    test = df[df["timestamp"] > cutoff]
    return train, test


if __name__ == "__main__":
    print("Loading data from Hopsworks...")
    df = load_data()

    print(f"Building {FORECAST_HORIZON_HOURS}h-ahead target per city...")
    df = build_target(df)
    print(f"Rows after target creation: {len(df):,}")

    # fill the small amount of missing pollutant data with column median
    for col in ["o3", "no2", "so2"]:
        df[col] = df[col].fillna(df[col].median())

    train_df, test_df = chronological_split(df, train_frac=0.8)
    print(f"\nChronological split:")
    print(f"  Train: {len(train_df):,} rows, {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"  Test:  {len(test_df):,} rows, {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    X_train, y_train = train_df[FEATURE_COLS], train_df["target_aqi"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target_aqi"]

    print("\nTraining Ridge Regression...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n" + "=" * 50)
    print(f"RESULTS -- Ridge Regression, {FORECAST_HORIZON_HOURS}h-ahead AQI forecast")
    print("=" * 50)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R2:   {r2:.3f}")

    print("\nFeature coefficients (higher magnitude = more influence):")
    coefs = pd.Series(model.coef_, index=FEATURE_COLS).sort_values(key=abs, ascending=False)
    print(coefs.round(3))

    print("\nSample predictions vs actual (first 10 test rows):")
    sample = test_df[["city", "timestamp"]].copy()
    sample["actual"] = y_test.values
    sample["predicted"] = preds.round(1)
    print(sample.head(10).to_string(index=False))
