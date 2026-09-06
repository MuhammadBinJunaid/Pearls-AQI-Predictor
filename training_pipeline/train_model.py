"""
AQI Training Pipeline — multi-model, multi-horizon
=====================================================
Fetches historical (features, target) from the Hopsworks Feature Store,
trains and evaluates THREE model types (Ridge Regression, Random Forest,
small TensorFlow dense network) for THREE forecast horizons (24h, 48h,
72h -- i.e. the next 3 days), picks the best model per horizon by RMSE,
and stores it in the Hopsworks Model Registry.

Validation is always CHRONOLOGICAL (train on earlier data, test on
later data) -- never randomly shuffled, since this is time-series data.
Evaluation always reports RMSE, MAE, and R2 together, never a single
metric alone.

Run with:
    python training_pipeline/train_model.py
"""

import os
import json
import shutil
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

load_dotenv()

HORIZONS_HOURS = [24, 48, 72]  # next 3 days
FEATURE_COLS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day_of_week", "day", "month", "aqi_change_rate",
]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
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
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    for col in ["o3", "no2", "so2"]:
        df[col] = df[col].fillna(df[col].median())
    return df


def build_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Target = AQI `horizon_hours` ahead, computed per city (no cross-city leakage)."""
    df = df.copy()
    df["target_aqi"] = df.groupby("city")["aqi"].shift(-horizon_hours)
    return df.dropna(subset=["target_aqi"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    cutoff = df["timestamp"].quantile(train_frac)
    return df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
def train_ridge(X_train, y_train):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_tensorflow(X_train, y_train, X_val, y_val):
    import tensorflow as tf
    from tensorflow.keras import layers, Sequential, callbacks

    model = Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early_stop = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=256, verbose=0,
        callbacks=[early_stop],
    )
    return model


def evaluate(model, X_test, y_test, is_keras=False):
    preds = model.predict(X_test, verbose=0).flatten() if is_keras else model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------
def save_to_registry(model_path: str, model_name: str, metrics: dict, is_keras: bool):
    import hopsworks

    project = hopsworks.login(
        api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
        project=os.environ.get("HOPSWORKS_PROJECT"),
    )
    mr = project.get_model_registry()

    py_model = mr.python.create_model(
        name=model_name,
        metrics={"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]},
        description=f"AQI forecast model ({model_name}) -- RMSE {metrics['rmse']:.2f}",
    )
    py_model.save(model_path)
    print(f"  Saved '{model_name}' to Hopsworks Model Registry (RMSE {metrics['rmse']:.2f})")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data from Hopsworks...")
    df_raw = load_data()

    results_summary = []

    for horizon in HORIZONS_HOURS:
        print(f"\n{'=' * 60}")
        print(f"HORIZON: {horizon}h ahead")
        print("=" * 60)

        df = build_target(df_raw, horizon)
        train_df, test_df = chronological_split(df)

        X_train_raw = train_df[FEATURE_COLS]
        y_train = train_df["target_aqi"]
        X_test_raw = test_df[FEATURE_COLS]
        y_test = test_df["target_aqi"]

        # feature scaling -- required for Ridge and TensorFlow to behave sensibly;
        # Random Forest doesn't need it but scaling doesn't hurt it either.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # hold out a small validation slice from the END of train for TensorFlow early stopping
        val_cut = int(len(X_train) * 0.9)
        X_tr, X_val = X_train[:val_cut], X_train[val_cut:]
        y_tr, y_val = y_train.values[:val_cut], y_train.values[val_cut:]

        candidates = {}

        print("Training Ridge Regression...")
        ridge = train_ridge(X_train, y_train)
        candidates["ridge"] = (ridge, evaluate(ridge, X_test, y_test), False)

        print("Training Random Forest...")
        rf = train_random_forest(X_train, y_train)
        candidates["random_forest"] = (rf, evaluate(rf, X_test, y_test), False)

        print("Training TensorFlow (small dense net)...")
        tf_model = train_tensorflow(X_tr, y_tr, X_val, y_val)
        candidates["tensorflow"] = (tf_model, evaluate(tf_model, X_test, y_test, is_keras=True), True)

        print(f"\nResults for {horizon}h horizon:")
        for name, (_, metrics, _) in candidates.items():
            print(f"  {name:15s} RMSE={metrics['rmse']:6.2f}  MAE={metrics['mae']:6.2f}  R2={metrics['r2']:.3f}")

        best_name = min(candidates, key=lambda n: candidates[n][1]["rmse"])
        best_model, best_metrics, best_is_keras = candidates[best_name]
        print(f"\n  -> BEST: {best_name} (RMSE {best_metrics['rmse']:.2f})")

        # save locally
        horizon_dir = os.path.join(MODEL_DIR, f"aqi_model_{horizon}h")
        if os.path.exists(horizon_dir):
            shutil.rmtree(horizon_dir)
        os.makedirs(horizon_dir)

        if best_is_keras:
            best_model.save(os.path.join(horizon_dir, "model.keras"))
        else:
            joblib.dump(best_model, os.path.join(horizon_dir, "model.joblib"))
        joblib.dump(scaler, os.path.join(horizon_dir, "scaler.joblib"))
        with open(os.path.join(horizon_dir, "metadata.json"), "w") as f:
            json.dump({
                "horizon_hours": horizon,
                "best_model_type": best_name,
                "is_keras": best_is_keras,
                "feature_cols": FEATURE_COLS,
                "metrics": best_metrics,
                "all_candidates": {n: m for n, (_, m, _) in candidates.items()},
            }, f, indent=2)

        print(f"  Saved locally to {horizon_dir}")

        # push to Hopsworks Model Registry
        try:
            save_to_registry(
                horizon_dir,
                model_name=f"aqi_forecast_{horizon}h",
                metrics=best_metrics,
                is_keras=best_is_keras,
            )
        except Exception as e:
            print(f"  WARNING: could not push to Model Registry ({e}). Model is saved locally regardless.")

        results_summary.append({"horizon": horizon, "best_model": best_name, **best_metrics})

    print("\n" + "=" * 60)
    print("FINAL SUMMARY -- best model per horizon")
    print("=" * 60)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
