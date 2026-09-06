# AQI Predictor — Agent Instructions

## Project Overview
Multi-city AQI forecasting system for Pakistan using Open-Meteo (no API key) + Hopsworks Feature Store/Model Registry. Three pipelines: feature extraction, model training, Streamlit dashboard.

## Environment Setup
```bash
# Required env vars (in .env or exported)
HOPSWORKS_API_KEY=...
HOPSWORKS_PROJECT=...
AQI_CITIES=karachi,lahore,islamabad  # optional, defaults to all 19 Pakistan cities
# OPENWEATHER_KEY present in .env but NOT used (Open-Meteo requires no key)
```

## Commands

### Feature Pipeline (hourly data ingestion)
```bash
# Live mode: fetch today's hourly data for all cities (run hourly)
python feature_pipeline.py --mode live

# Backfill mode: fetch N days of historical data (~90-day chunks per city)
python feature_pipeline.py --mode backfill --days 730
```
- Stores to Hopsworks feature group `aqi_features` v4 (primary key: city + timestamp)

### Training Pipeline (daily retrain)
```bash
python training_pipeline/train_model.py
```
- Loads from `aqi_features` v4, chronological split (80/20), no shuffle
- Trains 3 models × 3 horizons (24h/48h/72h): Ridge, Random Forest, small TF dense net
- Picks best per horizon by RMSE, saves locally + pushes to Hopsworks Model Registry as `aqi_forecast_{24,48,72}h`
- Feature columns: `aqi,pm25,pm10,o3,no2,so2,co,temperature,humidity,pressure,wind_speed,hour,day_of_week,day,month,aqi_change_rate`
- Fills `o3`, `no2`, `so2` NaN with median before training

### Dashboard
```bash
streamlit run app/dashboard.py
```
- Caches full feature table (30 min TTL) and all models (1 hr TTL) at session start
- Falls back to demo data if Hopsworks unavailable
- Computes SHAP explanations from cached 24h model (TreeExplainer for RF, coef_ for Ridge, KernelExplainer for TF)
- Loads everything upfront before rendering (fixes empty-box flash)

### Test Hopsworks Connection
```bash
python test_hopsworks_connection.py
```

## CI/CD (GitHub Actions)
- `.github/workflows/feature_pipeline.yml`: Runs hourly (`0 * * * *`), live mode
- `.github/workflows/training_pipeline.yml`: Runs daily at 03:00 UTC
- Both use `python-version: '3.11'`, install deps from requirements.txt, require secrets
- Note: CI installs subset of deps; `shap`, `matplotlib` not in workflow installs (dashboard only)

## Key Architecture Notes
- **No API keys needed** for Open-Meteo endpoints (air quality + weather archive/forecast)
- 19 hardcoded Pakistan cities in `feature_pipeline.py:40-60`; override via `AQI_CITIES` env
- Feature group version is **v4** (Open-Meteo based) — update if schema changes
- Models saved locally in `training_pipeline/trained_models/aqi_model_{24,48,72}h/` with `model.keras`/`.joblib`, `scaler.joblib`, `metadata.json`
- Dashboard loads **latest model version** from registry per horizon (not hardcoded)
- Time-series validation: always chronological split, target = AQI shifted by horizon hours per city

## Common Gotchas
- `.env` is gitignored — must be created locally
- `venv/` is gitignored — recreate with `pip install -r requirements.txt`
- Feature pipeline uses `time.sleep(0.5)` between city requests to respect API
- Backfill fetches in 90-day chunks (Open-Meteo limit)
- Dashboard loads everything upfront before rendering (fixes empty-box flash)
- If `o3`, `no2`, `so2` have NaN, they're filled with median (both training + dashboard)