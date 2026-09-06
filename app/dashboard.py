"""
Pearls AQI Predictor — Dashboard
=================================
Loads the latest features + trained models from Hopsworks ONCE per
session, computes 3-day AQI forecasts for all cities from that cached
data, and displays it with SHAP explanations and hazard alerts.

Run with:
    streamlit run app/dashboard.py
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# DESIGN SYSTEM — atmospheric / instrument-panel identity
# ---------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --ink: #0E1620;
    --haze: #16212C;
    --haze-2: #1D2A38;
    --sky: #4FA8D8;
    --text: #E7EEF3;
    --muted: #8CA0AE;
    --good: #4CAF7D;
    --moderate: #E8C547;
    --sensitive: #E8974A;
    --unhealthy: #E0563F;
    --very-unhealthy: #A85FBE;
    --hazardous: #7A2E3D;
}

.stApp {
    background: linear-gradient(180deg, var(--ink) 0%, #0A0F16 100%);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sky);
    margin-bottom: 0.3rem;
}

.headline {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    line-height: 1.1;
    color: var(--text);
    margin: 0 0 0.2rem 0;
}

.subhead {
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 1.6rem;
}

.card {
    background: var(--haze);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}

.metric-number {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 3.2rem;
    line-height: 1;
    text-shadow: 0 0 24px currentColor;
}

.metric-label {
    color: var(--muted);
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

.band-tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.25rem 0.7rem;
    border-radius: 100px;
    text-transform: uppercase;
}

.alert-banner {
    background: linear-gradient(90deg, rgba(122,46,61,0.35), rgba(122,46,61,0.08));
    border-left: 3px solid var(--hazardous);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    margin-bottom: 1.4rem;
}

.day-card {
    background: var(--haze-2);
    border-radius: 12px;
    padding: 1.1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
}

.day-label {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.day-number {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 2rem;
}

hr.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 2rem 0 1.4rem 0;
}

div[data-baseweb="select"] > div {
    background: var(--haze) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--sky) !important;
    font-size: 0.85rem !important;
}

button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--muted) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--sky) !important;
    border-bottom-color: var(--sky) !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: var(--sky) !important;
}

.health-card {
    background: linear-gradient(135deg, var(--haze) 0%, var(--haze-2) 100%);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    border-left: 3px solid var(--sky);
    margin-top: 1rem;
}

.leaderboard-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: var(--haze-2);
    transition: transform 0.15s ease;
}

.leaderboard-row:hover {
    transform: translateX(4px);
}

.rank-badge {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.08);
    margin-right: 0.8rem;
}

@media (max-width: 900px) {
    .metric-number { font-size: 2.2rem !important; }
    .headline { font-size: 1.8rem !important; }
}

.block-container {
    padding-top: 2rem !important;
    max-width: 1200px;
}

.live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--good); display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(76,175,125,0.6); }
    70% { box-shadow: 0 0 0 8px rgba(76,175,125,0); }
    100% { box-shadow: 0 0 0 0 rgba(76,175,125,0); }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# AQI SCALE — the real EPA/AQICN breakpoints, used as the design signature
# ---------------------------------------------------------------------
AQI_BANDS = [
    (0, 50, "Good", "var(--good)"),
    (51, 100, "Moderate", "var(--moderate)"),
    (101, 150, "Unhealthy (Sensitive)", "var(--sensitive)"),
    (151, 200, "Unhealthy", "var(--unhealthy)"),
    (201, 300, "Very Unhealthy", "var(--very-unhealthy)"),
    (301, 500, "Hazardous", "var(--hazardous)"),
]


def band_for(aqi: float):
    for lo, hi, label, color in AQI_BANDS:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "var(--hazardous)"


def generate_aqi_story(city_name, current_aqi, forecast, top_feature_name):
    delta = forecast[0] - current_aqi
    direction = "worsening" if delta > 0 else "improving"
    return (
        f"{city_name}'s air quality is {direction} over the next 24 hours, "
        f"driven mainly by {top_feature_name}. Expect AQI to shift from "
        f"{current_aqi:.0f} to around {forecast[0]:.0f} by tomorrow."
    )


def parse_city_from_text(text: str):
    """Extract city name from user text (simple fuzzy match)."""
    text_lower = text.lower()
    for city in CITY_OPTIONS:
        if city.lower() in text_lower:
            return city
    return None


def answer_aqi_question(question: str, city: str, current_aqi: float, forecast: list, features_df: pd.DataFrame = None) -> str:
    """Answer AQI questions from the user."""
    q_lower = question.lower()
    
    # Current AQI questions
    if any(x in q_lower for x in ["current aqi", "how is the air", "what's the aqi", "air quality now"]):
        label, _ = band_for(current_aqi)
        return f"📊 {city}'s current AQI is **{current_aqi:.0f}** ({label}). This is based on live air quality data updated every hour."
    
    # Forecast questions
    if any(x in q_lower for x in ["forecast", "tomorrow", "next 24", "what will it be"]):
        return f"🔮 AQI forecast for {city}: **24h: {forecast[0]:.0f}** | **48h: {forecast[1]:.0f}** | **72h: {forecast[2]:.0f}**. These are model predictions based on historical patterns."
    
    # Health/activity questions
    if any(x in q_lower for x in ["health", "exercise", "activity", "outdoor", "can i", "safe", "should i"]):
        label, _ = band_for(current_aqi)
        if current_aqi <= 50:
            return f"🟢 Air quality is **{label}**. It's safe to enjoy outdoor activities, including exercise!"
        elif current_aqi <= 100:
            return f"🟡 Air quality is **{label}**. Most activities are fine, but sensitive groups should be cautious during peak hours (10am-4pm)."
        elif current_aqi <= 150:
            return f"🟠 Air quality is **{label}**. Sensitive groups (children, elderly, asthma) should limit outdoor exertion. Others can exercise but may want to reduce intensity."
        elif current_aqi <= 200:
            return f"🔴 Air quality is **{label}**. Everyone should limit outdoor activity. Keep windows closed and use air purifiers if available."
        else:
            return f"🟣 Air quality is **{label}** — hazardous! Stay indoors, keep windows/doors closed. This is not a time for outdoor activity."
    
    # Pollution contributor questions
    if any(x in q_lower for x in ["cause", "what's causing", "why is it", "pollutant", "pollution"]):
        return f"🔬 The main contributors to {city}'s air quality are tracked across 7 pollutants: PM2.5, PM10, O3, NO2, SO2, CO, and more. Weather patterns (wind, humidity, temperature) also play a major role in daily changes."
    
    # Default fallback
    return f"💭 I can help with AQI predictions, health advice, and air quality info for {city}. Try asking: *'Is it safe to exercise?'*, *'What's the forecast?'*, or *'Why is the air bad today?'*"


FEATURE_COLS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day_of_week", "day", "month", "aqi_change_rate",
]
HORIZONS = [24, 48, 72]
FRIENDLY_NAMES = {
    "aqi": "Current AQI", "pm25": "PM2.5", "pm10": "PM10", "o3": "Ozone (O3)",
    "no2": "Nitrogen Dioxide (NO2)", "so2": "Sulfur Dioxide (SO2)", "co": "Carbon Monoxide (CO)",
    "temperature": "Temperature", "humidity": "Humidity", "pressure": "Pressure",
    "wind_speed": "Wind Speed", "hour": "Hour of Day", "day_of_week": "Day of Week",
    "day": "Day", "month": "Month", "aqi_change_rate": "AQI Change Rate",
}

DEFAULT_CITIES = [
    "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan",
    "Peshawar", "Quetta", "Sialkot", "Gujranwala", "Hyderabad", "Sukkur",
    "Bahawalpur", "Sargodha", "Sheikhupura", "Larkana", "Gujrat", "Mardan", "Kasur",
]
cities_env = os.environ.get("AQI_CITIES")
CITY_OPTIONS = [c.strip().title() for c in cities_env.split(",")] if cities_env else DEFAULT_CITIES


# ---------------------------------------------------------------------
# ONE shared Hopsworks connection, cached for the whole session.
# ---------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def get_project():
    import hopsworks
    return hopsworks.login(
        api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
        project=os.environ.get("HOPSWORKS_PROJECT"),
    )


# ---------------------------------------------------------------------
# ONE full read of the feature table, cached. Every city then just
# filters this in-memory DataFrame -- no repeated downloads.
# ---------------------------------------------------------------------
@st.cache_data(ttl=1800)
def load_full_features():
    """
    Reads the full feature table once (proven to work reliably, same as
    the training pipeline uses) and caches it for 30 minutes. The earlier
    slowness wasn't from reading the full table -- it was from reading it
    ~20 separate times, once per city. Calling it exactly once here (and
    caching the result) is what actually fixes performance.
    """
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=4)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ["o3", "no2", "so2"]:
        df[col] = df[col].fillna(df[col].median())
    return df.sort_values(["city", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# ONE load of all trained models, cached. Always fetches the LATEST
# version per horizon (not hardcoded to version 1).
# ---------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def load_models():
    import joblib
    import json

    project = get_project()
    mr = project.get_model_registry()

    models = {}
    for horizon in HORIZONS:
        model_name = f"aqi_forecast_{horizon}h"
        all_versions = mr.get_models(model_name)
        if not all_versions:
            raise RuntimeError(f"No model found in registry for '{model_name}'")
        latest = max(all_versions, key=lambda m: m.version)
        model_dir = latest.download()

        with open(os.path.join(model_dir, "metadata.json")) as f:
            meta = json.load(f)

        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

        if meta["is_keras"]:
            import tensorflow as tf
            model = tf.keras.models.load_model(os.path.join(model_dir, "model.keras"))
        else:
            model = joblib.load(os.path.join(model_dir, "model.joblib"))

        models[horizon] = {"model": model, "scaler": scaler, "meta": meta, "version": latest.version}

    return models


def predict_for_city(df: pd.DataFrame, models: dict, city_slug: str):
    """Pure in-memory computation -- no network calls. Uses the already-
    cached full DataFrame and already-cached models."""
    city_df = df[df["city"] == city_slug].sort_values("timestamp").tail(1)
    if city_df.empty:
        raise ValueError(f"No stored data for {city_slug}")

    current_aqi = float(city_df["aqi"].iloc[0])
    latest_features = city_df[FEATURE_COLS]

    forecast = []
    rmse_by_horizon = {}
    for horizon in HORIZONS:
        entry = models[horizon]
        X_scaled = entry["scaler"].transform(latest_features)
        if entry["meta"]["is_keras"]:
            pred = float(entry["model"].predict(X_scaled, verbose=0).flatten()[0])
        else:
            pred = float(entry["model"].predict(X_scaled)[0])
        forecast.append(max(0.0, pred))
        rmse_by_horizon[horizon] = entry["meta"]["metrics"]["rmse"]

    return current_aqi, forecast, rmse_by_horizon


def health_recommendation(aqi: float, hour_of_day: int) -> tuple:
    """Returns (headline, detail, icon) for the current AQI + time of day."""
    if aqi <= 50:
        return ("Good day to be outside", "Air quality is fine for all outdoor activities, including exercise.", "🟢")
    elif aqi <= 100:
        if 10 <= hour_of_day <= 16:
            return ("Fine for most people", "Air is Moderate right now, and typically peaks around midday -- unusually sensitive individuals may prefer early morning or evening for outdoor exercise.", "🟡")
        return ("Good time for outdoor activity", "Air quality is Moderate and this is typically a lower-pollution part of the day.", "🟡")
    elif aqi <= 150:
        return ("Sensitive groups should take care", "Children, older adults, and people with asthma or heart conditions should limit prolonged outdoor exertion today.", "🟠")
    elif aqi <= 200:
        return ("Limit outdoor exertion", "Everyone may begin to experience effects; sensitive groups should avoid outdoor activity, especially around midday.", "🔴")
    else:
        return ("Stay indoors if possible", "Air quality is Very Unhealthy to Hazardous -- avoid outdoor activity and keep windows closed.", "🟣")


def render_trend_chart(features_df: pd.DataFrame, city_slug: str, current_aqi: float, forecast: list, rmse_by_horizon: dict):
    """Renders historical + forecast trend chart with confidence band."""
    # last 24h actual
    city_hist = features_df[features_df["city"] == city_slug].sort_values("timestamp").tail(24)
    hist_times = city_hist["timestamp"].tolist()
    hist_values = city_hist["aqi"].tolist()

    # forecast points continue from "now"
    forecast_times = [datetime.now() + timedelta(days=i + 1) for i in range(3)]
    forecast_values = forecast

    # RMSE band around the forecast
    upper = [f + (rmse_by_horizon.get(h) or 0) for f, h in zip(forecast_values, HORIZONS)]
    lower = [max(0, f - (rmse_by_horizon.get(h) or 0)) for f, h in zip(forecast_values, HORIZONS)]

    fig = go.Figure()

    # actual history
    fig.add_trace(go.Scatter(
        x=hist_times, y=hist_values, mode="lines",
        line=dict(color="#4FA8D8", width=2), name="Actual",
    ))

    # forecast line, connected to the last actual point
    fig.add_trace(go.Scatter(
        x=[hist_times[-1]] + forecast_times, y=[hist_values[-1]] + forecast_values,
        mode="lines+markers", line=dict(color="#E8974A", width=2, dash="dot"),
        marker=dict(size=7), name="Forecast",
    ))

    # confidence band
    fig.add_trace(go.Scatter(
        x=forecast_times + forecast_times[::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(232,151,74,0.15)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        plot_bgcolor="#16212C", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E7EEF3", family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10), height=340,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="AQI"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------
# LOAD EVERYTHING ONCE, UP FRONT -- this is what fixes both the
# slowness and the empty-box flash: all the slow work happens before
# any layout is drawn, so the page renders complete in one pass.
# (FIX 4: Themed loading spinner)
# ---------------------------------------------------------------------
load_error = None
with st.spinner("Reading the atmosphere over Pakistan..."):
    try:
        features_df = load_full_features()
        models = load_models()
        data_source = "live"
    except Exception as e:
        load_error = str(e)
        features_df = None
        models = None
        data_source = "demo"

# ---------------------------------------------------------------------
# HEADER + CITY SELECTOR
# ---------------------------------------------------------------------
st.markdown('<div class="eyebrow">Pearls AQI Predictor · Pakistan Air Quality Network</div>', unsafe_allow_html=True)
city = st.selectbox("City", CITY_OPTIONS, label_visibility="collapsed")
city_slug = city.lower().replace(" ", "-")

if data_source == "live":
    try:
        current_aqi, forecast, rmse_by_horizon = predict_for_city(features_df, models, city_slug)
        source = "live"
    except Exception as e:
        load_error = str(e)
        source = "demo"
else:
    source = "demo"

if source == "demo":
    if load_error:
        st.warning(f"Could not load live data/models, showing demo values. Error: {load_error}")
    seed = sum(ord(c) for c in city_slug)
    current_aqi = 60 + (seed % 180)
    forecast = [
        max(10, current_aqi + ((seed * 3) % 40) - 20),
        max(10, current_aqi + ((seed * 7) % 60) - 30),
        max(10, current_aqi + ((seed * 11) % 50) - 25),
    ]
    rmse_by_horizon = {h: None for h in HORIZONS}

band_label, band_color = band_for(current_aqi)

# Compute SHAP importances early (needed for story generation)
is_real_shap = False
importances_df = None
if data_source == "live":
    try:
        import shap
        entry = models[24]
        city_df = features_df[features_df["city"] == city_slug].sort_values("timestamp").tail(50)
        X = city_df[FEATURE_COLS]
        X_scaled = entry["scaler"].transform(X)
        
        if entry["meta"]["best_model_type"] == "random_forest":
            explainer = shap.TreeExplainer(entry["model"])
            shap_values = explainer.shap_values(X_scaled)
            importances = np.abs(shap_values).mean(axis=0)
        elif hasattr(entry["model"], "coef_"):
            importances = np.abs(entry["model"].coef_)
        else:
            background = X_scaled[:20]
            explainer = shap.KernelExplainer(
                lambda x: entry["model"].predict(x, verbose=0).flatten(), background
            )
            shap_values = explainer.shap_values(X_scaled[-1:], nsamples=100)
            importances = np.abs(shap_values).flatten()
        
        importances_df = pd.DataFrame({
            "feature": [FRIENDLY_NAMES.get(c, c) for c in FEATURE_COLS],
            "importance": importances,
        }).sort_values("importance")
        is_real_shap = True
    except Exception as e:
        pass

if importances_df is None:
    importances_df = pd.DataFrame({
        "feature": ["PM2.5", "Humidity", "Hour of Day", "PM10", "Wind Speed", "AQI Change Rate"],
        "importance": [0.34, 0.21, 0.16, 0.14, 0.09, 0.06],
    }).sort_values("importance")

# ---------------------------------------------------------------------
# HEADLINE
# ---------------------------------------------------------------------
st.markdown(f'<div class="headline">Air over {city}, next 3 days</div>', unsafe_allow_html=True)
live_indicator = '<span class="live-dot"></span>' if source == "live" else ""
st.markdown(
    f'<div class="subhead">{live_indicator}Updated {datetime.now().strftime("%b %d, %Y · %H:%M")} '
    f'{"— live from Hopsworks" if source == "live" else "— demo data, see warning above"}</div>',
    unsafe_allow_html=True,
)

if data_source == "live" and importances_df is not None and len(importances_df) > 0:
    top_feature = importances_df.sort_values("importance", ascending=False).iloc[0]["feature"]
    story = generate_aqi_story(city, current_aqi, forecast, top_feature)
    st.markdown(f'<div class="subhead" style="font-style:italic;">{story}</div>', unsafe_allow_html=True)

if max([current_aqi] + forecast) >= 151:
    st.markdown(
        '<div class="alert-banner">⚠ HAZARD ALERT — forecasted AQI reaches unhealthy levels within '
        'the next 3 days. Sensitive groups should limit prolonged outdoor exertion.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# MAIN LAYOUT — current reading + vertical AQI scale signature
# ---------------------------------------------------------------------
col_main, col_scale = st.columns([2.4, 1])

with col_main:
    # FIX 3: Animated counter for current reading
    current_html = (
        '<div class="card">'
        '<div class="eyebrow">Current reading</div>'
        f'<div class="metric-number" id="aqi-counter" style="color:{band_color}">{current_aqi:.0f}</div>'
        '<div class="metric-label">AQI</div>'
        f'<span class="band-tag" style="background:{band_color}22; color:{band_color}; '
        f'border:1px solid {band_color}55;">{band_label}</span>'
        '</div>'
        '<script>'
        '(function() {'
        f'    let el = window.parent.document.getElementById("aqi-counter");'
        '    if (!el) return;'
        f'    let target = {current_aqi:.0f};'
        '    let current = 0;'
        '    let step = Math.max(1, Math.round(target / 30));'
        '    let timer = setInterval(function() {'
        '        current += step;'
        '        if (current >= target) { current = target; clearInterval(timer); }'
        '        el.innerText = current;'
        '    }, 20);'
        '})()'
        '</script>'
    )
    st.markdown(current_html, unsafe_allow_html=True)

with col_scale:
    scale_rows = ""
    for lo, hi, label, color in reversed(AQI_BANDS):
        active = lo <= current_aqi <= hi
        opacity = "1" if active else "0.35"
        marker = " ← today" if active else ""
        scale_rows += (
            f'<div style="display:flex; align-items:center; gap:0.6rem; margin:0.35rem 0; opacity:{opacity};">'
            f'<div style="width:10px; height:10px; border-radius:3px; background:{color};"></div>'
            f'<div style="font-family:JetBrains Mono, monospace; font-size:0.75rem; color:var(--text);">'
            f'{lo}–{hi} {label}{marker}</div></div>'
        )
    scale_html = (
        '<div class="card" style="height:100%;">'
        '<div class="eyebrow">AQI scale</div>'
        f'{scale_rows}'
        '</div>'
    )
    st.markdown(scale_html, unsafe_allow_html=True)

# ===== TABBED NAVIGATION ===== 
st.markdown('<hr class="divider">', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 3-Day Forecast", "📈 Trend", "🌍 All Cities", "🔬 Why This Forecast", "🤖 AQI Copilot"])

# ===== TAB 1: 3-Day Forecast =====
with tab1:
    st.markdown('<div class="eyebrow">Next 3 days ahead</div>', unsafe_allow_html=True)
    
    # Forecast day cards
    forecast_cols = st.columns(3)
    for i, (col, horizon, pred_aqi) in enumerate(zip(forecast_cols, HORIZONS, forecast)):
        label, color = band_for(pred_aqi)
        delta = pred_aqi - current_aqi
        direction = "Worsening" if delta > 0 else "Improving"
        direction_color = "var(--unhealthy)" if delta > 0 else "var(--good)"
        arrow = "↑" if delta > 0 else "↓"
        rmse_val = rmse_by_horizon.get(horizon)
        rmse_text = f"±{rmse_val:.1f}" if rmse_val else "N/A"
        
        with col:
            st.markdown(
                f'<div class="day-card" style="border-left:3px solid {color};">'
                f'<div class="day-label">{horizon}h forecast</div>'
                f'<div class="day-number" style="color:{color};">{pred_aqi:.0f}</div>'
                f'<div style="font-size:0.7rem; color:{direction_color}; margin-top:0.4rem;">{arrow} {direction} {abs(delta):.1f}</div>'
                f'<div style="color:var(--muted); font-size:0.8rem; margin-top:0.5rem;">Confidence: {rmse_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    
    st.caption("Forecasts are model estimates based on historical patterns, not medical or health advice.")
    
    # Health recommendation card
    hour_of_day = datetime.now().hour
    headline, detail, icon = health_recommendation(current_aqi, hour_of_day)
    st.markdown(
        f'<div class="health-card">'
        f'<div style="font-weight:600; font-size:1.1rem; margin-bottom:0.5rem;">{icon} {headline}</div>'
        f'<div style="color:var(--muted); font-size:0.9rem;">{detail}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ===== TAB 2: Trend =====
with tab2:
    st.markdown('<div class="eyebrow">Historical + forecast trends</div>', unsafe_allow_html=True)
    
    if data_source == "live" and features_df is not None:
        try:
            fig = render_trend_chart(features_df, city_slug, current_aqi, forecast, rmse_by_horizon)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Last 24h actual (blue), 3-day forecast (orange dashed), confidence band (orange ±{list(rmse_by_horizon.values())[0]:.1f} AQI).")
        except Exception as e:
            st.info(f"Trend chart unavailable: {e}")
    else:
        st.info("Live data not available for trend visualization.")

# ===== TAB 3: All Cities =====
with tab3:
    st.markdown('<div class="eyebrow">All monitored cities</div>', unsafe_allow_html=True)
    
    compare_cols_per_row = 5
    for i in range(0, len(CITY_OPTIONS), compare_cols_per_row):
        row_cities = CITY_OPTIONS[i:i + compare_cols_per_row]
        row_cols = st.columns(compare_cols_per_row)
        for c, col in zip(row_cities, row_cols):
            c_slug = c.lower().replace(" ", "-")
            try:
                if data_source == "live":
                    c_aqi, _, _ = predict_for_city(features_df, models, c_slug)
                else:
                    raise RuntimeError("demo mode")
            except Exception:
                seed = sum(ord(ch) for ch in c_slug)
                c_aqi = 60 + (seed % 180)
            c_label, c_color = band_for(c_aqi)
            is_selected = (c == city)
            with col:
                st.markdown(
                    f'<div class="day-card" style="{"border:2px solid " + c_color + ";" if is_selected else ""}">'
                    f'<div class="day-label">{c}</div>'
                    f'<div class="day-number" style="color:{c_color}; font-size:1.3rem;">{c_aqi:.0f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    
    # Leaderboard section
    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
    col_worst, col_improved = st.columns(2)
    
    if data_source == "live" and features_df is not None:
        # Build leaderboard data
        city_aqi_map = {}
        for c in CITY_OPTIONS:
            c_slug = c.lower().replace(" ", "-")
            try:
                c_aqi, _, _ = predict_for_city(features_df, models, c_slug)
                city_aqi_map[c] = c_aqi
            except:
                seed = sum(ord(ch) for ch in c_slug)
                city_aqi_map[c] = 60 + (seed % 180)
        
        # Worst air today
        with col_worst:
            st.markdown('<div class="eyebrow">Worst air today</div>', unsafe_allow_html=True)
            worst_sorted = sorted(city_aqi_map.items(), key=lambda x: x[1], reverse=True)[:5]
            for idx, (city_name, aqi_val) in enumerate(worst_sorted, 1):
                _, c_color = band_for(aqi_val)
                st.markdown(
                    f'<div class="leaderboard-row" style="display:flex; align-items:center; gap:0.8rem; padding:0.8rem; '
                    f'background:rgba(255,255,255,0.02); border-radius:8px; margin:0.4rem 0;">'
                    f'<div class="rank-badge" style="background:rgba(255,255,255,0.08); width:28px; height:28px; '
                    f'display:flex; align-items:center; justify-content:center; border-radius:50%; font-weight:600;">{idx}</div>'
                    f'<div style="flex:1;">{city_name}</div>'
                    f'<div style="font-weight:700; color:{c_color}; font-size:1.1rem;">{aqi_val:.0f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        
        # Most improved today
        with col_improved:
            st.markdown('<div class="eyebrow">Most improved today</div>', unsafe_allow_html=True)
            # Use aqi_change_rate if available
            most_improved = sorted(city_aqi_map.items(), key=lambda x: x[1])[:5]
            for idx, (city_name, aqi_val) in enumerate(most_improved, 1):
                _, c_color = band_for(aqi_val)
                st.markdown(
                    f'<div class="leaderboard-row" style="display:flex; align-items:center; gap:0.8rem; padding:0.8rem; '
                    f'background:rgba(255,255,255,0.02); border-radius:8px; margin:0.4rem 0;">'
                    f'<div class="rank-badge" style="background:rgba(255,255,255,0.08); width:28px; height:28px; '
                    f'display:flex; align-items:center; justify-content:center; border-radius:50%; font-weight:600;">{idx}</div>'
                    f'<div style="flex:1;">{city_name}</div>'
                    f'<div style="font-weight:700; color:{c_color}; font-size:1.1rem;">{aqi_val:.0f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ===== TAB 4: Why This Forecast =====
with tab4:
    st.markdown('<div class="eyebrow">Feature importance (24h model)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # importances_df is already computed early, just display it here
    st.bar_chart(importances_df.set_index("feature"), horizontal=True, color="#4FA8D8")
    if not is_real_shap:
        st.caption("Demo values shown.")

    st.markdown('</div>', unsafe_allow_html=True)

# ===== TAB 5: AQI Copilot =====
with tab5:
    st.markdown('<div class="eyebrow">Ask about air quality</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:var(--muted); font-size:0.9rem; margin-bottom:1.4rem;">'
        'Ask me anything about air quality, health, forecasts, or pollution. I\'ll give you personalized insights for your city.'
        '</div>',
        unsafe_allow_html=True,
    )
    
    # Initialize session state for chat history
    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []
    
    # Chat history display
    if st.session_state.copilot_history:
        st.markdown('<div style="margin-bottom:1.4rem;">', unsafe_allow_html=True)
        for msg in st.session_state.copilot_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="card" style="background:rgba(79, 168, 216, 0.1); border-left:3px solid var(--sky);">'
                    f'<div style="color:var(--sky); font-weight:600; margin-bottom:0.4rem;">You</div>'
                    f'<div style="color:var(--text);">{msg["content"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="card" style="background:rgba(76, 175, 125, 0.05); border-left:3px solid var(--good);">'
                    f'<div style="color:var(--good); font-weight:600; margin-bottom:0.4rem;">AQI Copilot</div>'
                    f'<div style="color:var(--text);">{msg["content"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input section
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Your question",
            placeholder="e.g., 'Is it safe to exercise?', 'What's the forecast?'",
            label_visibility="collapsed",
            key="aqi_question",
        )
    
    with col_btn:
        ask_button = st.button("Send", use_container_width=True, type="primary")
    
    # Process question
    if ask_button and user_question.strip():
        # Add user question to history
        st.session_state.copilot_history.append({"role": "user", "content": user_question})
        
        # Generate answer
        answer = answer_aqi_question(user_question, city, current_aqi, forecast, features_df)
        st.session_state.copilot_history.append({"role": "assistant", "content": answer})
        
        # Rerun to show new message immediately
        st.rerun()
    
    # Clear history button
    if st.session_state.copilot_history:
        if st.button("Clear chat history", use_container_width=True, help="Reset the conversation"):
            st.session_state.copilot_history = []
            st.rerun()
    
    # Help section
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Quick topics</div>', unsafe_allow_html=True)
    quick_topics = [
        ("🏃 Health & Exercise", "Is it safe to exercise today?"),
        ("🔮 Forecast", "What's the 24-hour forecast?"),
        ("🌡️ Pollution", "What's causing the pollution?"),
        ("📊 Current AQI", "How is the air quality now?"),
    ]
    
    cols = st.columns(2)
    for idx, (emoji_title, question) in enumerate(quick_topics):
        with cols[idx % 2]:
            if st.button(emoji_title, use_container_width=True, key=f"quick_{idx}"):
                st.session_state.copilot_history.append({"role": "user", "content": question})
                answer = answer_aqi_question(question, city, current_aqi, forecast, features_df)
                st.session_state.copilot_history.append({"role": "assistant", "content": answer})
                st.rerun()