# AQI Predictor — Pakistan Air Quality Forecasting

Multi-city AQI forecasting system for Pakistan using **Open-Meteo** (no API key required) + **Hopsworks** Feature Store/Model Registry.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Feature Pipeline│────▶│  Hopsworks       │◀───│ Training Pipeline│
│ (hourly ingest) │     │  Feature Store   │     │ (daily retrain) │
└─────────────────┘     │  aqi_features v4 │     └────────┬────────┘
                        └────────┬─────────┘              │
                                 │                        │
                                 ▼                        ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Hopsworks       │     │  Streamlit      │
                        │  Model Registry  │────▶│  Dashboard      │
                        │  aqi_forecast    │     │  (real-time)    │
                        └──────────────────┘     └─────────────────┘
```

### Three Pipelines

| Pipeline | Schedule | Command | Purpose |
|----------|----------|---------|---------|
| **Feature Pipeline** | Hourly (GitHub Actions) | `python feature_pipeline.py --mode live` | Fetches today's hourly AQI + weather from Open-Meteo for 19 Pakistan cities, engineers features, stores in Hopsworks Feature Store |
| **Training Pipeline** | Daily 03:00 UTC (GitHub Actions) | `python training_pipeline/train_model.py` | Loads features, trains 3 models × 3 horizons (24h/48h/72h), picks best by RMSE, pushes to Model Registry |
| **Dashboard** | On-demand | `streamlit run app/dashboard.py` | Loads cached features + models, shows 3-day forecasts with SHAP explanations for all cities |

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Hopsworks account (free tier works)
- `.env` file with credentials (see below)

### 2. Environment Setup
```bash
# Clone and enter project
cd aqi-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
Create `.env` in project root:
```env
HOPSWORKS_API_KEY=your_api_key_here
HOPSWORKS_PROJECT=your_project_name
AQI_CITIES=karachi,lahore,islamabad  # optional: defaults to all 19 cities
```
> **Note**: `OPENWEATHER_KEY` in .env is ignored — Open-Meteo requires no API key.

### 4. Test Connection
```bash
python test_hopsworks_connection.py
```

### 5. Run Pipelines Manually
```bash
# Feature pipeline — live mode (today's data)
python feature_pipeline.py --mode live

# Feature pipeline — backfill (historical data, ~90-day chunks)
python feature_pipeline.py --mode backfill --days 730

# Training pipeline — retrains all 3 horizons
python training_pipeline/train_model.py

# Dashboard
streamlit run app/dashboard.py
```

---

## Performance Notes (Important)

### Current Bottlenecks
| Component | Time | Issue |
|-----------|------|-------|
| Feature table load | ~33s | Downloads 333K rows (all 19 cities × 2 years) |
| 24h model (TF) load | ~26s | TensorFlow model initialization |
| 48h/72h models | ~1s | Cached, fast |
| **Total dashboard startup** | **~60s** | Loads everything upfront before rendering |

### Optimizations Applied
- Dashboard now loads **only latest row per city** for prediction (not full history)
- Models cached for 1 hour via `@st.cache_resource`
- Feature table cached for 30 min via `@st.cache_data`

---

## Model Performance (Latest Training)

| Horizon | Best Model | RMSE | MAE | R² |
|---------|------------|------|-----|-----|
| 24h | TensorFlow | 22.28 | ~16 | ~0.65 |
| 48h | TensorFlow | 29.04 | ~21 | ~0.52 |
| 72h | TensorFlow | 30.26 | ~22 | ~0.48 |

> **Interpretation**: RMSE ~22-30 on AQI scale (0-500). For context: AQI bands are 0-50 (Good), 51-100 (Moderate), 101-150 (Unhealthy for Sensitive), 151-200 (Unhealthy), 201-300 (Very Unhealthy), 301+ (Hazardous). Errors of ~20-30 AQI points are usable for band-level forecasting but not precise point predictions.

---

## Project Structure
```
aqi-predictor/
├── feature_pipeline.py           # Hourly data ingestion from Open-Meteo
├── training_pipeline/
│   ├── train_model.py            # Daily retraining (3 models × 3 horizons)
│   └── trained_models/           # Local model artifacts (gitignored)
├── app/
│   └── dashboard.py              # Streamlit dashboard with SHAP
├── notebooks/                    # EDA and experimentation
├── .github/workflows/            # CI/CD (hourly feature, daily training)
├── requirements.txt
├── test_hopsworks_connection.py
├── AGENTS.md                     # Agent instructions
└── .env                          # Credentials (gitignored)
```

---

## Key Technical Details

- **Data Source**: Open-Meteo Air Quality API + Historical/Forecast Weather API (no API key)
- **Cities**: 19 major Pakistan cities (hardcoded in `feature_pipeline.py:40-60`)
- **Feature Group**: `aqi_features` v4 (PK: city + timestamp)
- **Models**: Ridge, Random Forest, TensorFlow dense net — best per horizon by RMSE
- **Validation**: Chronological split (80/20), target = AQI shifted by horizon hours per city
- **NaN Handling**: `o3`, `no2`, `so2` filled with median (both training + dashboard)
- **CI/CD**: GitHub Actions (hourly feature, daily training), requires `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` secrets

---

## Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | `pip install -r requirements.txt` in venv |
| Hopsworks auth fails | Check `.env` has correct `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` |
| Dashboard slow (>60s) | First load downloads models; subsequent loads use cache (1hr TTL) |
| No data in dashboard | Run feature pipeline first: `python feature_pipeline.py --mode live` |
| Backfill fails | Open-Meteo limits to ~90 days per request; script handles chunking |

---

## License
MIT