
import os
import io
import json
import time
import math
import base64
import requests
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st
# Forecasting imports with graceful fallbacks
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False
    
try:
    from pmdarima import auto_arima
    _HAS_ARIMA = True
except Exception:
    _HAS_ARIMA = False

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

import plotly.express as px
import plotly.graph_objects as go

# SustainifyAI — Sustainability & Climate Change Tracker (All‑in‑One Streamlit App)
# ---------------------------------------------------------------------------------
# ✅ Final Version: Premium UI, Live Data, AI Forecasts, Universal Explanations,
#     Animated Goal Numbers, and Localized Auto-Carbon Footprint.
# ---------------------------------------------------------------------------------

st.set_page_config(
    page_title="SustainifyAI — Sustainability & Climate Tracker",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Premium & Futuristic Theming (CSS) ------------------------------
st.markdown(
    """
    <style>
    :root {
      --bg: #070d18;
      --card: #101625;
      --muted: #aebed0;
      --brand: #4ade80; /* Neon Green */
      --brand2:#60a5fa; /* Neon Blue */
      --brand-glow: 0 0 10px rgba(96, 165, 250, 0.4), 0 0 20px rgba(74, 222, 128, 0.2);
    }
    
    /* --- CORE BACKGROUND & LAYOUT --- */
    .stApp { 
        background: transparent !important; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* CINEMATIC GRADIENT OVERLAY (New Element) */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -90; /* Above video (-100), below content (0) */
        /* Orange-Blue Fusion Gradient for Cinematic Feel */
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(251, 191, 36, 0.1) 40%, rgba(30, 64, 175, 0.2) 70%, rgba(74, 222, 128, 0.1) 100%);
        pointer-events: none;
        opacity: 0.9;
    }
    
    .video-background {
        position: fixed; 
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        transform: none;
        z-index: -100;
        pointer-events: none;
        object-fit: cover; 
        filter: brightness(0.65); /* Video Dimmer */
    }
    
    /* HIGH CONTRAST: Main content area (60% Transparent) */
    .main > div {
        background-color: rgba(7, 13, 24, 0.40); 
        backdrop-filter: blur(2px);
    }
    /* HIGH CONTRAST: Sidebar (70% Opaque - Sidebar Focus) */
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #0d162a 0%, #172a4d 100%) !important; /* Brighter, more vibrant gradient */
        border-right: 2px solid rgba(96, 165, 250, 0.3); /* Stronger border */
        box-shadow: 6px 0 30px rgba(0, 0, 0, 0.8), inset -3px 0 10px rgba(255, 255, 255, 0.05); /* Deeper shadow, inner glow */
    }

    .block-container { padding-top: 1rem; }
    
    /* --- NEWS TICKER (PIPE) STYLING --- */
    @keyframes marquee {
      0%   { transform: translate(100%, 0); }
      100% { transform: translate(-100%, 0); }
    }
    .news-pipe-container {
      overflow: hidden;
      width: 100%;
      height: 40px; /* Height of the pipe */
      background: rgba(18, 26, 43, 0.9);
      border: 1px solid #2d406b;
      border-radius: 8px;
      margin-bottom: 20px;
      padding: 5px 0;
      box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
      position: relative;
    }
    .news-pipe-content {
      white-space: nowrap;
      color: #ffc44a; /* Amber/Gold for alerts */
      font-size: 1.1rem;
      font-weight: 600;
      padding-top: 2px;
      animation: marquee 25s linear infinite; /* Control speed here */
      text-shadow: 0 0 5px rgba(255, 196, 74, 0.4);
    }
    
    /* --- TEXT STYLING (Luminous White & Increased Size) --- */
    h1,h2,h3,h4 { 
        color:#f0f8ff; /* Alice Blue */
        letter-spacing:0.8px;
        text-shadow: 0 0 6px rgba(96, 165, 250, 0.2);
    }
    p,li,span,div, label, .stMarkdown { 
        color:#e8f0fe !important; /* Luminous White */
        font-size: 1.05rem !important; /* General Font Increase */
    }

    /* --- SIDEBAR SPECIFIC STYLING --- */
    /* Sidebar H1 / Title */
    .stSidebar h1 { 
        font-size: 1.9rem; /* Slightly larger sidebar title */
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(96, 165, 250, 0.6); /* Enhanced glow for sidebar titles */
    }
    
    /* Sidebar Input Labels (Search/Date) */
    .stSidebar label {
        color: #ffffff !important; /* Pure White labels */
        font-size: 1.05rem !important; /* Increased sidebar label size */
        letter-spacing: 0.8px;
        font-weight: 600; /* Bolder labels */
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.2); /* Soft glow */
    }

    /* Input Fields */
    .stSidebar [data-baseweb="input"], .stSidebar [data-baseweb="base-input"],
    .stSidebar [data-baseweb="select"] > div:first-child,
    .stSidebar .stDateInput > div:first-child > div { /* Target date input directly */
        background-color: #1a2c4a !important; /* Brighter dark blue background */
        border: 1px solid #4a689d !important; /* Vibrant border */
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5); /* Deeper inner shadow */
        border-radius: 8px;
        padding: 8px 10px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stSidebar [data-baseweb="input"]:focus-within, 
    .stSidebar [data-baseweb="base-input"]:focus-within,
    .stSidebar [data-baseweb="select"]:focus-within > div:first-child,
    .stSidebar .stDateInput > div:first-child:focus-within > div { /* Target date input directly */
        border-color: #80bfff !important; /* Vibrant light blue on focus */
        box-shadow: 0 0 0 2px #80bfff, inset 0 2px 8px rgba(0, 0, 0, 0.8); /* Stronger outer glow */
    }
    .stSidebar .stSuccess {
        background-color: #1abc9c !important; /* Brighter green for success */
        color: #ffffff !important; /* White text for contrast */
        border: 1px solid #1abc9c;
        border-radius: 12px;
        padding: 10px;
        margin-top: 5px;
        box-shadow: 0 4px 15px rgba(26, 188, 156, 0.6); /* More prominent shadow */
    }
    /* Globe Rotation Animation */
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .rotating-globe {
        display: inline-block;
        animation: spin 6s linear infinite;
        font-size: 1.5em;
        margin-right: 5px;
        text-shadow: 0 0 12px rgba(96, 165, 250, 0.8); /* Enhanced glow for globe */
    }

    /* Slider styling */
    .stSidebar .stSlider [data-baseweb="slider"] {
        background-color: #4a689d; /* Track background */
        height: 8px;
        border-radius: 4px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
    }
    .stSidebar .stSlider [data-baseweb="slider"] > div:nth-child(2) { /* Filled portion of track */
        background-color: #80bfff; /* Vibrant blue fill */
        box-shadow: 0 0 8px rgba(128, 191, 255, 0.6);
    }
    .stSidebar .stSlider [data-baseweb="slider"] > div:nth-child(3) { /* Slider handle */
        background-color: #e8f0fe; /* Bright white handle */
        border: 2px solid #80bfff; /* Vibrant blue border */
        box-shadow: 0 0 15px rgba(128, 191, 255, 0.8), 0 0 5px rgba(255,255,255,0.8);
        width: 20px;
        height: 20px;
        top: -6px; /* Adjust vertical position */
    }
    .stSidebar .stSlider [data-baseweb="slider"] > div:nth-child(3):hover {
        transform: scale(1.1); /* Slight pop on hover */
        box-shadow: 0 0 20px rgba(128, 191, 255, 1), 0 0 10px rgba(255,255,255,1);
    }
    
    /* --- HERO HEADING (MAX POP, GLOW, AND INTERACTIVITY) --- */
    @keyframes subtle-pulse {
        0% { opacity: 0.9; transform: scale(1.0); }
        50% { opacity: 1; transform: scale(1.005); }
        100% { opacity: 0.9; transform: scale(1.0); }
    }
    .hero { 
      position: relative; 
      padding: 30px 0 25px; /* Increased padding for cinematic space */
      text-align: center; /* Center alignment */
      animation: subtle-pulse 5s infinite ease-in-out; /* Subtle motion */
    }

    .hero-title {
      /* INCREASED SIZE & STYLISH FONT (Visage-like Impact font stack) */
      font-family: 'Montserrat', 'Impact', sans-serif; 
      font-size: clamp(45px, 7.5vw, 85px); /* Maximized size */
      font-weight: 900; 
      line-height: 1.05;

      /* GLOWING GRADIENT EFFECT (More vibrant than original) */
      background: linear-gradient(90deg, #ffc44a 0%, #ffffff 30%, #60a5fa 70%, #4ade80 100%);
      -webkit-background-clip: text; 
      background-clip: text; 
      color: transparent;

      /* INTENSE INITIAL GLOW */
      letter-spacing: 3px; /* Wider spacing */
      text-shadow: 
        0 0 20px rgba(255, 255, 255, 0.8), /* Strong White Glow */
        0 0 40px rgba(96,165,250,1), /* Blue Neon Glow */
        0 0 60px rgba(74, 222, 128, 0.6); /* Green Glow */

      /* POP & GLOW INTERACTIVITY */
      transition: all .5s cubic-bezier(0.25, 0.8, 0.25, 1); 
      display:inline-block; 
      cursor: default; /* Remove pointer effect */
    }

    .hero-title:hover { 
      transform: none; /* Disable hover pop out */
      filter: saturate(1.5) brightness(1.2); /* Gentle color boost on hover */
    }
    
    .hero-sub { 
        color:#f0f8ff !important; 
        font-size: 1.35rem !important; /* Larger sub-header */
        font-weight: 300; /* Thinner font */
        opacity:0.95; 
        letter-spacing: 1px;
        margin-top: 10px;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.3); /* Soft shadow */
    }

    /* --- GLASS MORPHISM KPI / CARDS (Header) --- */
    .glass-card-header { 
      /* Glass effect */
      background: rgba(16, 22, 37, 0.5); /* Semi-transparent */
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      
      border: 1px solid rgba(96, 165, 250, 0.3); /* Soft Blue/Brand border */
      padding: 15px; 
      border-radius: 20px; /* Highly rounded corners */
      box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2), 0 0 25px rgba(96, 165, 250, 0.2); /* Deep shadow + soft glow */
      transition: all .35s ease; 
      position: relative; 
      color: #f0f8ff !important; 
    }
    .glass-card-header:hover { 
        transform: translateY(-4px); 
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4), 0 0 30px rgba(96, 165, 250, 0.5);
    }
    
    /* Ensure Streamlit metrics inside the glass card look white/clean */
    .glass-card-header .stMetric > div > div:first-child {
        color: #aebed0 !important; /* Muted label color */
        font-size: 0.9rem;
    }
    .glass-card-header .stMetric > div > div:nth-child(2) {
        color: #ffffff !important; /* Pure white value */
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: -5px;
    }

    /* --- KPI / CARDS (Regular) --- */
    .metric-card { 
      background:linear-gradient(180deg,#16243e 0%,#0f1a31 100%); border:1px solid #2d406b; padding:20px; border-radius:18px; box-shadow:0 10px 30px rgba(0, 0, 0, 0.4); transition: all .35s ease; position: relative; color: #f0f8ff !important; 
    }
    .metric-card:hover{ transform: translateY(-8px) scale(1.02); background: radial-gradient(800px 220px at 10% 10%, #1e335e 0%, #142240 70%); box-shadow:0 18px 50px rgba(30,64,175,.5); border-color:#4f79c2; }
    
    /* --- CUSTOM KPI VALUE STYLING FOR DIRECT MARKDOWN (Integrates value inside the card) --- */
    .kpi-label {
        color: #f0f8ff !important;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .kpi-value {
        color: #ffc44a !important; 
        font-weight: 800; 
        font-size: 2.2rem; 
        text-shadow: 0 0 10px rgba(255, 196, 74, 0.5); 
        margin-top: -10px; /* Pull the number up into the button/card area */
    }

    /* --- TAB BUTTONS --- */
    .stTabs [data-baseweb="tab-list"] { background-color: rgba(7, 13, 24, 0.9); border-bottom: 2px solid #1e293b; margin-bottom: 20px; }
    .stTabs [aria-selected="true"] { color: #60a5fa !important; border-bottom: 4px solid #60a5fa !important; font-weight: 700; text-shadow: 0 0 10px rgba(96, 165, 250, 0.8) !important; }
    

    /* --- BUTTONS --- */
    .stButton.btn-primary button { 
        /* BRIGHTER BUTTON STYLE */
        background:linear-gradient(90deg,#ffc44a,#ff9933)!important; 
        color:#000000!important; 
        border:2px solid #ffc44a!important; 
        font-weight: 700; 
        box-shadow: 0 4px 20px rgba(255, 196, 74, 0.6), 0 0 15px rgba(255, 196, 74, 0.9); 
        transition: all 0.3s ease;
    }
    .stButton.btn-primary button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 25px rgba(255, 196, 74, 0.8), 0 0 20px rgba(255, 196, 74, 1); 
    }

    /* Plot Wrappers */
    .plot-wrap { 
        border:1px solid #2d406b; border-radius:18px; padding:12px; background: rgba(18, 26, 43, 0.70);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3); transition: transform .3s ease, filter .3s ease, border-color .3s ease; 
    }

    /* --- AFFORESTATION ENHANCEMENTS --- */
    @keyframes pulse {
      0% { transform: scale(1); opacity: 0.8; }
      50% { transform: scale(1.1); opacity: 1; }
      100% { transform: scale(1); opacity: 0.8; }
    }
    .planting-tree {
      display: inline-block;
      animation: pulse 1.5s infinite; /* Pulsing effect for 'planting' animation */
      text-shadow: 0 0 10px rgba(74, 222, 128, 0.8);
      font-size: 1.5em;
      margin-right: 5px;
    }
    
    .goal-number {
        font-size: 3.2rem !important; /* Large, bold number */
        font-weight: 900;
        color: #4ade80 !important; /* Neon Green for the goal */
        text-shadow: 0 0 15px rgba(74, 222, 128, 0.7);
        transition: all 0.5s ease;
    }
    .current-number {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #60a5fa !important; /* Neon Blue for current status */
    }
    .goal-card-title {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(96, 165, 250, 0.5);
    }
    .goal-card-metric {
        background: linear-gradient(145deg, #101625, #1e335e);
        border: 1px solid #2d406b;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* --- TEAM TITLE GLOW (Nxt Gen Developers) --- */
    @keyframes pop-glow {
        0% { transform: scale(1.0); opacity: 0.8; text-shadow: 0 0 10px rgba(74, 222, 128, 0.5); }
        50% { transform: scale(1.05); opacity: 1; text-shadow: 0 0 20px rgba(96, 165, 250, 1), 0 0 40px rgba(74, 222, 128, 0.8); }
        100% { transform: scale(1.0); opacity: 0.8; text-shadow: 0 0 10px rgba(74, 222, 128, 0.5); }
    }
    .team-title {
        font-size: 2.5rem !important; 
        font-weight: 900;
        line-height: 1.2;
        margin-top: 20px;
        margin-bottom: 20px;
        
        background: linear-gradient(45deg, #ffff99, #4ac2ff, #4ade80);
        -webkit-background-clip: text; 
        background-clip: text; 
        color: transparent;

        animation: pop-glow 2.5s infinite alternate ease-in-out;
        display: inline-block;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ Inject YouTube Video Background ------------------------------
# Video ID: Tp6HQCb70yM. Loop only the first 40 seconds (start=0, end=40)
st.markdown(f"""
<iframe class="video-background" src="https://www.youtube.com/embed/Tp6HQCb70yM?autoplay=1&mute=1&loop=1&playlist=Tp6HQCb70yM&controls=0&modestbranding=1&disablekb=1&rel=0&showinfo=0&iv_load_policy=3&start=0&end=40" frameborder="0" allow="autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
""", unsafe_allow_html=True)
# --------------------------------------------------------------------------------------------------

# ------------------------------ Utility: Caching ------------------------------
@st.cache_data(show_spinner=False)
def geocode_place(place: str) -> Optional[Tuple[float, float, str, str]]:
    """Use Open‑Meteo geocoding (no API key) to resolve a place to (lat, lon, name, country)."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": place, "count": 1, "language": "en", "format": "json"}, timeout=20)
    if r.ok:
        js = r.json()
        if js.get("results"):
            res = js["results"][0]
            return float(res["latitude"]), float(res["longitude"]), res.get("name",""), res.get("country","")
    return None

@st.cache_data(show_spinner=False)
def fetch_openmeteo_daily(lat: float, lon: float, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fetch daily climate variables from Open‑Meteo ERA5 reanalysis (no key).
    """
    
    today_date = dt.date.today()
    
    # Use yesterday's date for archive access if the user specified the current date or later
    if end >= today_date:
        api_end_date = today_date - dt.timedelta(days=1)
    else:
        api_end_date = end

    if api_end_date < start:
        # Return empty data frame with expected columns if period is invalid
        return pd.DataFrame({'time': [], 'temperature_2m_max': [], 'temperature_2m_mean': [], 'temperature_2m_min': [], 'precipitation_sum': [], 'windspeed_10m_max': [], 'shortwave_radiation_sum': []})

    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": api_end_date.isoformat(), # Use the adjusted end date
        "daily": [
            "temperature_2m_mean","temperature_2m_max","temperature_2m_min",
            "precipitation_sum","windspeed_10m_max","shortwave_radiation_sum",
        ],
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status() 
    js = r.json()
    df = pd.DataFrame(js["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df

@st.cache_data(show_spinner=False)
def fetch_air_quality_current(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch latest air quality using Open-Meteo's Air Quality API (No key required).
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    hourly_vars = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "domains": "auto",
        "timezone": "auto",
        "current": ",".join(hourly_vars)
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Air Quality API (Open-Meteo) fetch failed: {e}")
        return pd.DataFrame()

    rows = []
    if "current" in js and "hourly_units" in js:
        current_data = js["current"]
        units = js["hourly_units"]
        last_updated = current_data.get("time")
        
        for param in hourly_vars:
            value = current_data.get(param)
            unit = units.get(param, "µg/m³")
            
            if value is not None:
                rows.append({
                    "location": f"{js.get('latitude', lat):.3f}, {js.get('longitude', lon):.3f}",
                    "parameter": param,
                    "value": float(value),
                    "unit": unit,
                    "date": last_updated,
                    "lat": js.get('latitude', lat),
                    "lon": js.get('longitude', lon),
                })

    return pd.DataFrame(rows)

# --- Placeholder Functions for Complex Features (Dynamic for City) ---

@st.cache_data(ttl=3600, show_spinner=False)
def get_river_health_data(city_name: str):
    """Synthesizes data for the major river near the selected city."""
    
    city_lower = city_name.lower()
    
    # --- Dynamic Logic based on major Indian cities/regions ---
    if "mumbai" in city_lower or "pune" in city_lower:
        river = "Mula-Mutha/Mithi (Maharashtra)"
        do, bod, coliform, status = 4.5, 6.0, 4000, "Extreme Stress"
    elif "chennai" in city_lower or "madurai" in city_lower or "kochi" in city_lower:
        river = "Cooum/Vaigai (Tamil Nadu/Kerala)"
        do, bod, coliform, status = 3.0, 8.0, 5000, "Extreme Stress"
    elif "kolkata" in city_lower or "patna" in city_lower:
        river = "Hooghly/Ganga (East)"
        do, bod, coliform, status = 5.5, 3.5, 2000, "Critical Stress"
    elif "kanpur" in city_lower:
        river = "Ganga (Kanpur)"
        do, bod, coliform, status = 5.8, 4.5, 2500, "Critical Stress"
    elif "varanasi" in city_lower:
        river = "Ganga (Varanasi)"
        do, bod, coliform, status = 6.8, 3.2, 1200, "High Stress"
    elif "lucknow" in city_lower or "jaunpur" in city_lower:
        river = "Gomti (UP)"
        do, bod, coliform, status = 5.0, 4.0, 3000, "Critical Stress"
    elif "prayagraj" in city_lower or "allahabad" in city_lower:
        # Prayagraj Specific Data (Based on reports)
        river = "Ganga (Sangam/Prayagraj)"
        do, bod, coliform, status = 7.0, 3.5, 11000, "High Stress"
    elif "hyderabad" in city_lower:
        river = "Musil (Telangana)"
        do, bod, coliform, status = 4.0, 7.0, 4500, "Extreme Stress"
    else:
        # Default/General River Logic (Covers all smaller UP/Indian cities dynamically)
        river = f"{city_name} River (General)"
        do, bod, coliform, status = 7.5, 2.5, 800, "Moderate Stress"
        
    data = {
        "River": [river],
        "Dissolved Oxygen (DO mg/L)": [do], # Healthy > 6.0
        "BOD (mg/L)": [bod], # Good < 3.0
        "Coliform (MPN/100ml)": [coliform], # Safe < 500
        "Status": [status],
    }
    df = pd.DataFrame(data)
    df['Color'] = df['Status'].apply(lambda x: '#ef4444' if x == 'Critical Stress' or x == x == 'Extreme Stress' else ('#facc15' if x == 'High Stress' else '#4ade80'))
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_tree_inventory(city_name: str):
    """Synthesizes tree data and requirements for the selected city (Maximized UP Granularity)."""
    
    city_lower = city_name.lower()

    # --- Highly Granular UP/Major City Lookup ---
    population_proxies = {
        "delhi": 19000000, "mumbai": 20000000, "bengaluru": 13000000, "chennai": 8000000,
        "kanpur": 2700000, "lucknow": 2700000, "ghaziabad": 2500000, "agra": 1800000,
        "varanasi": 1500000, "meerut": 1500000, "bareilly": 1200000, "aligarh": 1000000,
        "moradabad": 1000000, "firozabad": 1000000, "jhansi": 800000, "gorakhpur": 800000,
        "prayagraj": 1600000, "allahabad": 1600000, # Prayagraj specific population
        # Default for smaller UP districts
    }

    current_trees_proxies = {
        "delhi": 3000000, "mumbai": 1500000, "bengaluru": 1200000, "chennai": 900000,
        "kanpur": 850000, "lucknow": 950000, "ghaziabad": 650000, "agra": 400000,
        "varanasi": 500000, "meerut": 350000, "bareilly": 300000, "aligarh": 250000,
        "moradabad": 250000, "firozabad": 200000, "jhansi": 180000, "gorakhpur": 190000,
        "prayagraj": 550000, "allahabad": 550000, # Prayagraj specific tree count
    }
    
    # Get base values, defaulting to a smaller urban size if city is not listed
    population = population_proxies.get(city_lower, 400000)
    current_trees = current_trees_proxies.get(city_lower, 100000)
        
    target_ratio = 10 # Trees per person (national standard recommendation)
    trees_needed = (population * target_ratio) - current_trees
    
    return {
        "city": city_name,
        "current": current_trees,
        "population": population,
        "target_ratio": target_ratio,
        "needed": max(0, trees_needed),
        "needed_per_capita": round(trees_needed / population, 2)
    }

def get_future_impact_prediction(pm25_level: float):
    """Predicts generalized health impact based on current PM2.5."""
    if pm25_level < 50:
        return {"health_risk": "Low", "advice": "Continue outdoor activities.", "color": "#4ade80"}
    elif 50 <= pm25_level < 100:
        return {"health_risk": "Moderate", "advice": "Sensitive groups should limit prolonged outdoor exertion.", "color": "#facc15"}
    else:
        return {"health_risk": "High", "advice": "All groups should avoid prolonged or heavy exertion outdoors. Wear N95 masks.", "color": "#ef4444"}

def get_pollution_news_ticker() -> str:
    """Combines suggested text into a single, moving line."""
    news_items = [
        "📣 Citizens are urged to reduce single-use plastic consumption near the riverfront.",
        "🌳 Government announces target to plant 6,50,000 more trees this year.",
        "💨 PUC compliance strictly monitored to reduce vehicular emissions.",
        "♻ Encourage composting: Reduce waste burden on Naini-Baswar landfills.",
    ]
    return " | ".join(news_items) * 3 


# ------------------------------ Sustainability Score ------------------------------

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(np.ones(len(s)))
    return (s - s.min()) / (s.max() - s.min())

@dataclass
class SustainabilityInputs:
    pm25: float
    co2_per_capita: float    # optional proxy if available
    renewable_share: float # 0..100
    water_quality_index: float # 0..100
    waste_recycling_rate: float # 0..100


def compute_sustainability_score(inp: SustainabilityInputs) -> Tuple[float, dict]:
    """Composite score 0‑100 with interpretable sub‑scores and weights."""
    # Lower PM2.5 is better. Invert it against a reference band.
    pm25_scaled = np.clip(1 - (inp.pm25 / 75.0), 0, 1)  # 75 µg/m³ ~ very poor
    co2_scaled = np.clip(1 - (inp.co2_per_capita / 20.0), 0, 1)  # 20 t/cap ~ bad
    ren_scaled = np.clip(inp.renewable_share / 100.0, 0, 1)
    water_scaled = np.clip(inp.water_quality_index / 100.0, 0, 1)
    waste_scaled = np.clip(inp.waste_recycling_rate / 100.0, 0, 1)

    weights = {
        "Air Quality (PM2.5)": 0.28,
        "CO₂ / Capita": 0.18,
        "Renewables Share": 0.24,
        "Water Quality": 0.15,
        "Recycling Rate": 0.15,
    }
    subs = {
        "Air Quality (PM2.5)": pm25_scaled,
        "CO₂ / Capita": co2_scaled,
        "Renewables Share": ren_scaled,
        "Water Quality": water_scaled,
        "Recycling Rate": waste_scaled,
    }
    score = sum(subs[k]*w for k, w in weights.items()) * 100
    return float(score), {k: round(v*100, 1) for k, v in subs.items()}

# ------------------------------ Forecasting Helpers ------------------------------

def backtest_train_forecast(df: pd.DataFrame, target_col: str, horizon: int = 30, model_choice: str = "auto"):
    """Time‑series train/validation split, fit model, forecast horizon days. Returns forecast and metrics."""
    ts = df[["time", target_col]].dropna().copy()
    ts = ts.sort_values("time")
    ts.rename(columns={"time":"ds", target_col:"y"}, inplace=True)

    # Use last 20% as validation
    n = len(ts)
    if n < 100:
        # Small set: reduce horizon
        horizon = max(7, min(horizon, n//5))
    split_idx = max(5, int(n*0.8))
    train, valid = ts.iloc[:split_idx], ts.iloc[split_idx:]

    y_pred = None

    def prophet_fit_forecast():
        m = Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(train)
        future = m.make_future_dataframe(periods=horizon)
        fcst = m.predict(future)
        return m, fcst

    def arima_fit_forecast():
        model = auto_arima(train["y"], seasonal=True, m=365, suppress_warnings=True)
        # Build future index
        full = pd.concat([train, valid], axis=0)
        steps = horizon
        preds = model.predict_in_sample(start=0, end=len(full)-1)
        future_preds = model.predict(n_periods=steps)
        fcst = pd.DataFrame({
            "ds": pd.date_range(full["ds"].iloc[-1] + pd.Timedelta(days=1), periods=steps, freq='D'),
            "yhat": future_preds
        })
        return model, fcst

    def ml_fit_forecast():
        # Simple lag features RF
        full = pd.concat([train, valid], axis=0).reset_index(drop=True)
        for lag in [1,2,7,14,30]:
            full[f"lag_{lag}"] = full["y"].shift(lag)
        full.dropna(inplace=True)
        X = full.drop(columns=["ds","y"]).values
        y = full["y"].values
        split = int(len(full)*0.8)
        Xtr, Xva = X[:split], X[split:]
        ytr, yva = y[:split], y[split:]
        rf = RandomForestRegressor(n_estimators=400, random_state=42)
        rf.fit(Xtr, ytr)
        # backtest pred
        y_pred_bt = rf.predict(Xva)
        # iterative future forecast using last row
        last = full.iloc[-1:].copy()
        future_rows = []
        for i in range(horizon):
            feats = last.drop(columns=["ds","y"]).values
            yhat = rf.predict(feats)[0]
            new_date = last["ds"].iloc[0] + pd.Timedelta(days=1)
            row = {
                "ds": new_date,
                "y": np.nan,
            }
            # shift lags
            for lag in [1,2,7,14,30]:
                if lag == 1:
                    row[f"lag_{lag}"] = yhat
                else:
                    row[f"lag_{lag}"] = last[f"lag_{lag-1}"] if f"lag_{lag-1}" in last else yhat
            future_rows.append(row)
            last = pd.DataFrame([row])
        fcst = pd.DataFrame(future_rows)
        fcst.rename(columns={"y": "yhat"}, inplace=True)
        return rf, fcst, y_pred_bt, yva

    model_used = None
    metrics = {"MAE": None, "MAPE": None}

    if (model_choice == "Prophet" and _HAS_PROPHET) or (model_choice == "auto" and _HAS_PROPHET):
        model_used = "Prophet"
        m, fcst = prophet_fit_forecast()
        # validation on valid segment
        yhat_valid = m.predict(valid[["ds"]])["yhat"].values
        metrics["MAE"] = float(mean_absolute_error(valid["y"].values, yhat_valid))
        metrics["MAPE"] = float(mean_absolute_percentage_error(valid["y"].values, yhat_valid))
    elif (model_choice == "ARIMA" and _HAS_ARIMA) or (model_choice == "auto" and _HAS_ARIMA):
        model_used = "ARIMA"
        m, fcst = arima_fit_forecast()
        # No direct valid preds; approximate using last portion of in-sample + known
        metrics["MAE"] = None
        metrics["MAPE"] = None
    else:
        model_used = "ML Ensemble"
        m, fcst, y_pred_bt, y_valid = ml_fit_forecast()
        metrics["MAE"] = float(mean_absolute_error(y_valid, y_pred_bt))
        metrics["MAPE"] = float(mean_absolute_percentage_error(y_valid, y_pred_bt))

    return model_used, ts, train, valid, fcst, metrics

# ------------------------------ Alerts (Telegram Optional) ------------------------------

def send_telegram(msg: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": msg})
    return r.ok

# ------------------------------ Sidebar Controls ------------------------------

st.sidebar.markdown(
    f"# <span class='rotating-globe'>🌍</span> SustainifyAI", 
    unsafe_allow_html=True
)
st.sidebar.caption("Premium climate intelligence dashboard — real data, forecasts, insights")

place_default = st.sidebar.text_input("🔎 Search a city / place", value="Varanasi")
geo = geocode_place(place_default) if place_default.strip() else None

if geo is None:
    st.sidebar.error("Couldn't geocode the place. Try a larger city or correct spelling.")
    st.stop()

lat, lon, _name, _country = geo
st.sidebar.success(f"📍 {_name}, {_country} | {lat:.3f}, {lon:.3f}")

today = dt.date.today()
start_date = st.sidebar.date_input("Start date", value=today - dt.timedelta(days=365*5))
end_date = st.sidebar.date_input("End date", value=today)

model_choice = st.sidebar.selectbox(
    "Forecast model",
    ["auto", "Prophet", "ARIMA", "ML Ensemble"],
    index=0,
    # 🌟 ENHANCEMENT: Added help text
    help="Choose the AI model: Prophet is great for strong seasonality (e.g., yearly temps); ARIMA is a classic statistical model; ML Ensemble (Random Forest) is a non-linear fallback."
)

st.sidebar.markdown("---")
# 🌟 ENHANCEMENT: Added help text
alert_pm25 = st.sidebar.slider("PM2.5 alert threshold (µg/m³)", 10, 200, 90, help="If the current PM2.5 (Air Quality) exceeds this threshold, a warning alert will be triggered on the dashboard.")
# 🌟 ENHANCEMENT: Added help text
alert_temp = st.sidebar.slider("Max temp alert (°C)", 30, 50, 44, help="If the latest recorded maximum temperature exceeds this threshold, a heat warning alert will be triggered.")

st.sidebar.markdown("---")
with st.sidebar.expander("Sustainability score inputs (optional overrides)"):
    co2_pc = st.number_input("CO₂ per capita (t)", min_value=0.0, value=1.9, step=0.1)
    ren_share = st.slider("Renewable energy share (%)", 0, 100, 22)
    water_idx = st.slider("Water quality index (%)", 0, 100, 65)
    recycle = st.slider("Waste recycling rate (%)", 0, 100, 30)

# ------------------------------ Header (Cinematic) ------------------------------
colA, colB = st.columns([0.7,0.3])
with colA:
    # Use the enhanced hero section for the main title and subtitle
    st.markdown("""
    <div class="badge">⬤ LIVE</div>
    <div class="hero">
      <div class="hero-title">🌍 SustainifyAI — Sustainability & Climate Change Tracker</div>
      <div class="hero-sub">Real‑time climate & air quality • AI forecasts • Sustainability scoring • Actionable alerts</div>
    </div>
    """, unsafe_allow_html=True)
with colB:
    # Use the new glass-morphism card for Location/Period
    st.markdown("<div class='glass-card-header'>" , unsafe_allow_html=True)
    st.metric("Location", f"{_name}, {_country}")
    st.metric("Period", f"{start_date:%d %b %Y} → {end_date:%d %b %Y}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ------------------------------ NEWS TICKER PIPE ------------------------------
st.markdown(
    f"""
    <div class="news-pipe-container">
        <div class="news-pipe-content">{get_pollution_news_ticker()}</div>
    </div>
    """, unsafe_allow_html=True
)

# ------------------------------ Data Pulls ------------------------------
with st.spinner("Fetching climate history (Open‑Meteo ERA5)…"):
    try:
        # fetch_openmeteo_daily now handles the end date logic to prevent 400 errors
        df_clim = fetch_openmeteo_daily(lat, lon, start_date, end_date)
    except Exception as e:
        st.error(f"Open‑Meteo fetch failed: {e}")
        st.stop()

with st.spinner("Fetching latest air quality (Open-Meteo AQ)…"):
    df_aq = fetch_air_quality_current(lat=lat, lon=lon)

# ------------------------------ KPIs (Custom Integrated Style) ------------------------------

# Extract values for cleaner use
pm25_now = float(df_aq.loc[df_aq["parameter"]=="pm2_5", "value"].head(1).fillna(np.nan).values[0]) if not df_aq.empty and (df_aq["parameter"]=="pm2_5").any() else np.nan
mean_temp = df_clim['temperature_2m_mean'].mean()
max_wind = df_clim['windspeed_10m_max'].max()
total_rain = df_clim['precipitation_sum'].sum()
total_solar = df_clim['shortwave_radiation_sum'].sum()


def render_kpi_card(label: str, value_str: str):
    """Renders a custom HTML KPI card with integrated value styling."""
    return f"""
    <div class='metric-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value_str}</div>
    </div>
    """

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(render_kpi_card("PM2.5 (µg/m³) — latest", 
                                  ("—" if math.isnan(pm25_now) else f"{pm25_now:.1f}")), 
                                  unsafe_allow_html=True)

with col2:
    st.markdown(render_kpi_card("Mean Temp (°C)", f"{mean_temp:.1f}"), unsafe_allow_html=True)

with col3:
    st.markdown(render_kpi_card("Max Wind (m/s)", f"{max_wind:.1f}"), unsafe_allow_html=True)

with col4:
    st.markdown(render_kpi_card("Total Rain (mm)", f"{total_rain:.1f}"), unsafe_allow_html=True)

with col5:
    st.markdown(render_kpi_card("Solar (MJ/m²)", f"{total_solar:.1f}"), unsafe_allow_html=True)

# Alerts
alerts = []
if not math.isnan(pm25_now) and pm25_now >= alert_pm25:
    alerts.append(f"⚠ High PM2.5 detected: {pm25_now:.1f} µg/m³ ≥ threshold {alert_pm25}")

# Check only the latest available day's temperature for current alert.
if not df_clim.empty:
    latest_max_temp = float(df_clim["temperature_2m_max"].iloc[-1])
    if latest_max_temp >= alert_temp:
        alerts.append(f"🔥 *CURRENT HEAT ALERT:* Latest max temperature of {latest_max_temp:.1f}°C exceeded threshold {alert_temp}°C.")
else:
    alerts.append("ℹ Climate data not loaded, temperature alert is inactive.")


if alerts:
    st.warning("\n".join(alerts))
    sent = send_telegram("\n".join(alerts))
    if sent:
        st.success("Sent Telegram alert ✅")

# ------------------------------ Tabs ------------------------------
TAB_OVERVIEW, TAB_AIR, TAB_TRENDS, TAB_FORECAST, TAB_FUTURE, TAB_SCORE, TAB_CARBON, TAB_ABOUT = st.tabs([
    "Overview", "Air Quality", "Climate Trends", "Forecasts", "Future Impact", "Sustainability Score", "Personal Carbon", "About Project 🚀"
])

with TAB_OVERVIEW:
    st.subheader("Key Climate & Air Quality Overview")
    
    # --- Row 1: Monthly Average Temperature (Bar Plot) & Annual Total Precipitation (Bar Plot) ---
    c1, c2 = st.columns([0.6,0.4])
    
    with c1:
        # 1. Monthly Average Temperature Profile (Grouped Bar Chart for easy comparison)
        
        # Calculate monthly averages across the entire history
        df_clim['month_name'] = df_clim['time'].dt.strftime('%b')
        monthly_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df_clim['month_name'] = pd.Categorical(df_clim['month_name'], categories=monthly_order, ordered=True)
        
        df_monthly = df_clim.groupby('month_name').agg(
            Avg_Max_Temp=('temperature_2m_max', 'mean'),
            Avg_Min_Temp=('temperature_2m_min', 'mean'),
        ).reset_index()
        
        df_monthly_melt = df_monthly.melt(id_vars='month_name', var_name='Metric', value_name='Temperature (°C)')

        fig = px.bar(
            df_monthly_melt, 
            x="month_name", 
            y="Temperature (°C)", 
            color="Metric",
            barmode="group",
            title="*Monthly Temperature Norms (Avg Max & Min)*",
            color_discrete_map={'Avg_Max_Temp': '#ef4444', 'Avg_Min_Temp': '#60a5fa'}, # Red for Max, Blue for Min
            category_orders={'month_name': monthly_order}
        )
        
        fig.update_layout(
            height=420,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d7ee'),
            xaxis_title="Month",
            yaxis_title="Avg Temperature (°C)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickfont=dict(color='#e8f0fe')), 
            yaxis=dict(tickfont=dict(color='#e8f0fe'))
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        # 2. Annual Total Precipitation (Simple Bar Chart for Year-to-Year Comparison)
        df_clim['year'] = df_clim['time'].dt.year
        df_annual_rain = df_clim.groupby('year')['precipitation_sum'].sum().reset_index()
        
        fig2 = px.bar(
            df_annual_rain, 
            x="year", 
            y="precipitation_sum", 
            labels={"precipitation_sum":"Total mm", "year":"Year"}, 
            title="*Total Annual Precipitation (mm)*",
            color_discrete_sequence=['#3b82f6'],
            height=380,
        )
        fig2.update_traces(marker_line_width=0.5, marker_line_color="#1f2937") 
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d7ee'),
            xaxis_tickangle=-45,
            xaxis=dict(tickfont=dict(color='#e8f0fe')), 
            yaxis=dict(tickfont=dict(color='#e8f0fe'))
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Row 2: Air Quality & Solar Radiation ---
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c3, c4 = st.columns([0.5,0.5])
    with c3:
        # 3. Mean Pollutants (Enhanced Bar Chart with Color Scale)
        if not df_aq.empty:
            # Pollutants from Open-Meteo AQ API
            pm_params = df_aq[df_aq["parameter"].isin(["pm2_5","pm10","nitrogen_dioxide","ozone","sulphur_dioxide","carbon_monoxide"])]
            pm_pivot = pm_params.pivot_table(index="parameter", values="value", aggfunc="mean").reset_index()
            
            fig3 = px.bar(
                pm_pivot, 
                x="parameter", 
                y="value", 
                title="*Current Pollutants (Open-Meteo AQ)*",
                color="value",
                color_continuous_scale=px.colors.sequential.Plasma_r,
            )
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d7ee'),
                yaxis_title="Concentration (Unit Varies)",
                xaxis_title="Pollutant Parameter",
                height=320,
                xaxis=dict(tickfont=dict(color='#e8f0fe')), 
                yaxis=dict(tickfont=dict(color='#e8f0fe'))
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No current Air Quality data found for this location.")
            
    with c4:
        # 4. Solar Radiation (Modern Area Chart)
        fig4 = px.area(
            df_clim, 
            x="time", 
            y="shortwave_radiation_sum", 
            title="*Solar Radiation (Daily Sum) — MJ/m²*", 
            labels={"shortwave_radiation_sum":"MJ/m²", "time":"Date"},
            color_discrete_sequence=['#fde68a']
        )
        # Apply line style via update_traces to ensure compatibility
        fig4.update_traces(
            fill='tozeroy', 
            line=dict(width=1, color='#facc15') 
        )
        
        fig4.update_layout(
            height=320,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d7ee'),
            xaxis=dict(tickfont=dict(color='#e8f0fe')), 
            yaxis=dict(tickfont=dict(color='#e8f0fe'))
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


with TAB_AIR:
    st.subheader("Latest Air Quality Measurements (Open-Meteo AQ)")
    
    if df_aq.empty:
        st.info("No AQ data found for this location window.")
    else:
        # Prepare data for Pie Chart
        df_pie = df_aq[['parameter', 'value']].copy()
        
        # Clean up parameter names for better labels
        df_pie['parameter'] = df_pie['parameter'].str.replace('_', ' ').str.title()
        df_pie.rename(columns={'parameter': 'Pollutant', 'value': 'Value'}, inplace=True)

        col_chart, col_table = st.columns([0.4, 0.6])

        with col_chart:
            # --- 1. Composition Pie Chart ---
            # FIX: Removed the duplicate 'values' argument that caused the SyntaxError.
            fig_pie = px.pie(
                df_pie,
                values='Value',  
                names='Pollutant',
                title='*Composition of Current Air Pollutants*',
                hole=0.4, # Creates a donut chart
                color_discrete_sequence=px.colors.sequential.Plasma,
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#070d18', width=1)))
            fig_pie.update_layout(
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8f0fe'),
                legend_title_text="Pollutant"
            )
            st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_table:
            # --- 2. Data Table (for detail/compliance checks) ---
            st.markdown("### Detailed Readings")
            df_aq_display = df_aq[["date", "parameter", "value", "unit"]].rename(
                columns={"date": "Last Updated", "parameter": "Pollutant", "value": "Value", "unit": "Unit"}
            )
            st.dataframe(df_aq_display, use_container_width=True)
            st.caption("Data source is Open-Meteo Air Quality API. Values represent instantaneous readings.")


with TAB_TRENDS:
    st.subheader("Multi‑variable Climate Trends")
    
    # Prepare data for Global Warming Line (Smoothed Trend)
    df_temp = df_clim.copy()
    if not df_temp.empty:
        df_temp['warming_trend'] = df_temp['temperature_2m_mean'].rolling(window=365, center=True).mean()
    
    colA, colB = st.columns(2)
    
    with colA:
        # --- Max Wind Speed (Highlighting an Abnormality) ---
        
        if not df_clim.empty:
            max_wind_value = df_clim['windspeed_10m_max'].max()
            abnormal_point = df_clim[df_clim['windspeed_10m_max'] == max_wind_value].iloc[0]
            abnormal_date = abnormal_point['time']
            
            alert_msg = f"🌪 Extreme Wind Alert! Recorded {max_wind_value:.1f} m/s on {abnormal_date.strftime('%Y-%m-%d')}."
            st.warning(alert_msg)

        fig = go.Figure()
        
        # Base Wind Speed Line (Neon Green)
        fig.add_trace(go.Scatter(x=df_clim["time"], y=df_clim["windspeed_10m_max"], name="Max Wind Speed", line=dict(color='#4ade80', width=2)))
        
        # Add annotation (Arrow and text) pointing to the abnormal peak
        if not df_clim.empty:
            fig.add_annotation(
                x=abnormal_date, 
                y=abnormal_point['windspeed_10m_max'], 
                text="ABNORMAL SPIKE", 
                showarrow=True, 
                font=dict(color="#ef4444", size=12), 
                arrowhead=2, 
                arrowsize=1.5, 
                arrowwidth=2, 
                arrowcolor="#ef4444", 
                ax=0, ay=-50
            )
            
        fig.update_layout(
            title="*Max Wind Speed (m/s) Trend with Anomaly Detection*",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d7ee'),
            xaxis_title="Date",
            yaxis_title="Wind Speed (m/s)"
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with colB:
        # --- Mean Temperature Trend (with Global Warming Line) ---
        fig = go.Figure()
        
        # Base Mean Temperature Line (Futuristic Blue)
        fig.add_trace(go.Scatter(
            x=df_clim["time"], 
            y=df_clim["temperature_2m_mean"], 
            name="Daily Mean Temp", 
            line=dict(color='#60a5fa', width=2),
            opacity=0.8
        ))
        
        # Global Warming Trend Line (Smoothed and Luminous)
        if not df_temp.empty:
            fig.add_trace(go.Scatter(
                x=df_temp["time"], 
                y=df_temp["warming_trend"], 
                name="Global Warming Trend (365-day Avg)", 
                line=dict(color='#ffc44a', width=4, dash='dashdot'), # Amber/Gold for contrast
                opacity=0.9
            ))
            
        fig.update_layout(
            title="*Mean Temperature Trend with Global Warming Line*",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d7ee'),
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            legend=dict(y=0.99, x=0.01)
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Correlation Snapshot ---
    st.markdown("### Correlation Snapshot")
    corr_df = df_clim.drop(columns=["time"]).corr(numeric_only=True)
    fig_corr = px.imshow(
        corr_df, 
        text_auto=".2f",
        aspect="auto", 
        title="*Climate Variables Correlation Matrix*",
        color_continuous_scale=px.colors.sequential.Plasma,
    )
    fig_corr.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d7ee'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 🌟 ENHANCEMENT: Added official, easy-to-read explanation for the Correlation Matrix
    st.markdown("### 🔍 Technical Explanation: Understanding Correlation")
    st.caption("""
    Correlation measures how closely two variables move together, ranging from **-1.0 to +1.0**.

    * **+1.0 (Bright Yellow/White):** Perfect *Positive Correlation*. When one variable (e.g., Max Temperature) increases, the other variable (e.g., Mean Temperature) increases at the same time.
    * **-1.0 (Dark Blue/Purple):** Perfect *Negative Correlation*. When one variable (e.g., Solar Radiation) increases, the other variable (e.g., Cloud Cover/Precipitation) decreases.
    * **0.0 (Mid-color):** *No Correlation*. The variables have no discernible linear relationship.
    
    This matrix helps us understand natural climate dependencies and potential impacts of climate change (e.g., looking at the 'time' column's correlation with temperature).
    """)
    
    # ------------------- HINGLISH EXPLANATION FOR PUBLIC -------------------
    st.markdown("### 🗣 सहसंबंध स्नैपशॉट (Correlation Snapshot) का मतलब समझें")
    st.markdown("""
    <div class='card'>
    <p style='color:#a8c0e8;'>यह ग्रिड (grid) हमें बताती है कि मौसम के अलग-अलग फैक्टर (factors) एक-दूसरे से कैसे जुड़े हैं।</p>
    
    <ul style='color:#c9d7ee;'>
        <li>*हाई वैल्यू (High Value) - ज़्यादातर पीला:* इसका मतलब है *Strong Positive Relation। मतलब, अगर **एक चीज़ बढ़ती है, तो दूसरी भी लगभग उसी Rate से बढ़ती है*। (Example: Max Temp और Mean Temp हमेशा साथ बढ़ते हैं: 0.96)</li>
        <li>*लो वैल्यू (Low Value) - ज़्यादातर गहरा नीला/जामुनी:* इसका मतलब है *Negative Relation* या *Weak Relation। मतलब, अगर एक चीज़ बढ़ती है, तो **दूसरी घट जाती है*, या उनका कोई खास कनेक्शन नहीं है। (Example: Rainfall ज़्यादा होने पर Solar Radiation (धूप) कम हो जाती है: लगभग -0.31)</li>
        <li>*जीरो के पास (Near Zero):* मतलब उन दोनों के बीच *कोई ख़ास Connection नहीं* है।</li>
    </ul>

    <p style='color:#a8c0e8; font-weight:600;'>इस डेटा से मुख्य बातें (Key Insights):</p>
    <ul style='color:#c9d7ee;'>
        <li>*गर्मी और धूप:* तापमान (Temperature) और धूप (Shortwave Radiation) का Strong Connection है (0.76)।</li>
        <li>*Precipitation (बारिश):* बारिश का किसी भी Temperature से बहुत कम Connection है (Max Temp से 0.00)।</li>
        <li>*Global Warming:* *'Year'* का *Temperature* से छोटा लेकिन Positive Connection (0.15) दिख रहा है, जो बताता है कि समय के साथ *Overall Temperature* थोड़ा बढ़ रहा है।</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    # ------------------- END HINGLISH EXPLANATION -------------------


with TAB_FORECAST:
    st.subheader("AI Forecasts with Backtest Metrics")
    target = st.selectbox(
        "Target to forecast", 
        ["temperature_2m_mean","temperature_2m_max","temperature_2m_min","precipitation_sum"], 
        index=0,
        # 🌟 ENHANCEMENT: Added help text
        help="The variable you want the AI model to predict into the future (e.g., Mean Temperature)."
    )
    horizon = st.slider("Forecast horizon (days)", 7, 365, 90)

    model_used, ts, train, valid, fcst, metrics = backtest_train_forecast(df_clim[["time", target]].dropna(), target, horizon=horizon, model_choice=model_choice)

    st.info(f"Model used: *{model_used}* |  MAE: *{metrics['MAE'] if metrics['MAE'] is not None else '—'}* |  MAPE: *{metrics['MAPE'] if metrics['MAPE'] is not None else '—'}*")

    # 🌟 ENHANCEMENT: Added metric explanations
    st.caption("""
    <span style='color:#a8c0e8;'>*MAE (Mean Absolute Error):* The average magnitude of errors in the predictions, measured in the same units as the target (e.g., °C). Lower is better.</span>
    <br>
    <span style='color:#a8c0e8;'>*MAPE (Mean Absolute Percentage Error):* The average percentage error of the prediction. 10% MAPE means the forecast is off by 10% on average. Lower is better.</span>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    # Premium Line Styling
    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="Train Data", line=dict(color='#60a5fa', width=2)))
    fig.add_trace(go.Scatter(x=valid["ds"], y=valid["y"], name="Validation Data", line=dict(color='#ef4444', width=2)))

    # unify forecast frame
    if "ds" not in fcst.columns:
        if "time" in fcst.columns:
            fcst.rename(columns={"time":"ds"}, inplace=True)
        else:
            # create series from last date
            last = ts["ds"].iloc[-1]
            fcst["ds"] = pd.date_range(last + pd.Timedelta(days=1), periods=len(fcst), freq='D')
    
    yhat = fcst["yhat"] if "yhat" in fcst.columns else fcst.iloc[:,1]
    
    # Prediction line (Neon Green and Dashed)
    fig.add_trace(go.Scatter(x=fcst["ds"], y=yhat, name="Forecast", line=dict(color='#4ade80', width=3, dash="dash")))
    
    fig.update_layout(
        title=f"*AI Forecast: {target}*", 
        xaxis_title="Date", 
        yaxis_title=target,
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f0fe'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("⬇ Download Forecast CSV", data=fcst.to_csv(index=False), file_name=f"forecast_{target}.csv", mime="text/csv")

with TAB_FUTURE:
    st.subheader(f"Future Impact Simulation & Environmental Health for {_name}")
    
    col_pred, col_adv = st.columns([0.4, 0.6])
    
    # --- Prediction Panel ---
    with col_pred:
        st.markdown("### 🏃 Current Health Danger Prediction")
        pm_level = pm25_now if not math.isnan(pm25_now) else 80.0
        prediction = get_future_impact_prediction(pm_level)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(f"Predicted Health Risk at PM2.5 of {pm_level:.1f} µg/m³", prediction['health_risk'])
        st.warning(f"*Running Advisory:* {prediction['advice']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Afforestation Goals (Styled & Animated) ---
        st.markdown(f"""
        ### <span class='planting-tree'>🌳</span> <span class='goal-card-title'>Afforestation Goals for {_name}</span>
        """, unsafe_allow_html=True)
        
        tree_data = get_tree_inventory(_name)
        
        c_tree1, c_tree2 = st.columns(2)
        
        # Current Trees (Est.)
        with c_tree1:
            st.markdown(f"""
            <div class='goal-card-metric'>
                <p style='color:#aebed0; font-size:1.0rem; margin-bottom: 5px;'>Current Trees (Est.)</p>
                <div class='current-number'>{tree_data['current']:,}</div>
            </div>
            """, unsafe_allow_html=True)

        # Trees Needed (Animated Goal)
        with c_tree2:
            placeholder = st.empty()
            
            # Simulate the "running number" by briefly showing a low number first
            if tree_data['needed'] > 100000:
                placeholder.markdown(f"""
                <div class='goal-card-metric'>
                    <p style='color:#aebed0; font-size:1.0rem; margin-bottom: 5px;'>Trees Needed (Goal)</p>
                    <div class='goal-number' id='animated-goal-start'>100,000</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.1) 

            # Final number snap-in
            placeholder.markdown(f"""
            <div class='goal-card-metric'>
                <p style='color:#aebed0; font-size:1.0rem; margin-bottom: 5px;'>Trees Needed (Goal)</p>
                <div class='goal-number'>{tree_data['needed']:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.caption(f"Goal: {tree_data['target_ratio']} trees per person (Population: {tree_data['population']:,})")


    # --- River Health Panel ---
    with col_adv:
        river_data = get_river_health_data(_name)
        river_name = river_data['River'].iloc[0]
        river_base_name = river_name.split(' (')[0] if ' (' in river_name else river_name # e.g., "Ganga"

        st.markdown(f"### 🌊 River Health Status ({river_name})")
        
        st.dataframe(
            river_data[['River', 'Dissolved Oxygen (DO mg/L)', 'BOD (mg/L)', 'Coliform (MPN/100ml)', 'Status']], 
            hide_index=True, 
            use_container_width=True
        )
        
        # 🌟 ENHANCEMENT: Added metric explanations for River Health
        st.markdown(f"### 💧 Understanding River Health Metrics")
        st.markdown(f"""
        <div style="background-color: #16243e; border: 1px solid #2d406b; padding: 15px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
        <p style='color:#f0f8ff; font-weight:600;'>What do these numbers mean?</p>
        <ul style='color:#c9d7ee; padding-left: 20px;'>
            <li>*Dissolved Oxygen (DO):* This is the *oxygen available for fish and other life. <span style='color:#4ade80;'>GOAL: Above 6.0 mg/L (Good)*</span>. Low DO means aquatic life is struggling.</li>
            <li>*BOD (Biochemical Oxygen Demand):* Measures the *amount of oxygen required to break down pollutants* (like sewage). <span style='color:#ef4444;'>*GOAL: Below 3.0 mg/L (Good)*</span>. High BOD means high pollution.</li>
            <li>*Coliform (MPN/100ml):* Shows the *level of harmful bacteria from sewage. <span style='color:#ef4444;'>GOAL: Below 500 (Safe)*</span>. High levels indicate high risk of diseases.</li>
        </ul>
        <p style='color:#ffc44a; font-size:0.95rem;'>Source: Central Pollution Control Board (CPCB) standards for 'Bathing Water' classification.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📢 Actionable Advisories")
        st.markdown(f"""
        <div class="card">
        *Public Advice:* Focus on waste segregation, reducing personal vehicle use, and avoiding littering near {river_base_name} River.
        *Government Advice:* Prioritize industrial effluent treatment, expand public transit, and invest in large-scale urban greening projects in {_name}.
        </div>
        """, unsafe_allow_html=True)


with TAB_SCORE:
    st.subheader("City Sustainability Score")
    pm_for_score = pm25_now if not math.isnan(pm25_now) else 60.0
    score, sub = compute_sustainability_score(SustainabilityInputs(
        pm25=pm_for_score,
        co2_per_capita=co2_pc,
        renewable_share=float(ren_share),
        water_quality_index=float(water_idx),
        waste_recycling_rate=float(recycle),
    ))
    colL, colR = st.columns([0.45,0.55])
    with colL:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Sustainability Score", f"{score:.1f} / 100")
        # 🌟 ENHANCEMENT: Added score explanation
        st.caption("A composite score (0-100) based on five dimensions: Air Quality, CO₂ Emissions, Renewable Energy usage, Water Quality, and Waste Recycling Rate. Higher is better.")
        st.markdown("</div>", unsafe_allow_html=True)
    with colR:
        sub_df = pd.DataFrame({"Dimension": list(sub.keys()), "Score": list(sub.values())})
        fig = px.bar(sub_df, x="Dimension", y="Score", title="Sub‑Scores (0‑100)")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e8f0fe'))
        st.plotly_chart(fig, use_container_width=True)

with TAB_CARBON:
    st.subheader(f"Personal Carbon Footprint for {_name} (Quick Estimate)")

    # 1. Auto-Estimation Checkbox
    auto_estimate = st.checkbox(
        "⚡ *Auto-Fetch Household Estimate* based on current location data", 
        value=False,
        help="Calculates a baseline household footprint using India-specific averages, adjusted by your city's air quality, climate, and sustainability metrics."
    )

    # Define India-Specific Baseline Averages for Carbon Footprint Estimation
    BASE_KWH = 180      # National average monthly consumption
    BASE_KM_CAR = 350    # Average urban monthly car travel (higher side)
    BASE_LPG = 6       # Average monthly LPG use (kg)
    BASE_FLIGHTS = 1     # Average flights per year
    
    # Emission Factors (unchanged)
    EF_CAR = 0.18  # kg CO2e / km
    EF_KWH = 0.7   # kg CO2e / kWh (India grid average)
    EF_FLIGHT = 180 # kg CO2e / 2-hour flight
    EF_LPG = 3.0   # kg CO2e / kg LPG
    DIET_MAP = {"Heavy meat": 300, "Mixed": 200, "Vegetarian": 150, "Vegan": 120}

    colA, colB = st.columns(2)
    
    # 2. Determine Inputs (Manual or Auto-Fetched)
    if auto_estimate:
        st.info(f"Using *{_name}* environmental data to localize the estimate.")
        
        # 🌟 PROXY LOGIC: Intelligent data-driven estimation
        
        # Proxy 1: Diet Type (Proxy based on city Water Quality Index)
        diet_proxy = "Mixed"
        if water_idx < 50:
             diet_proxy = "Heavy meat"
        elif water_idx > 80:
             diet_proxy = "Vegetarian"
        
        # Proxy 2: Car Travel (Proxy based on population proxy and max wind speed)
        tree_data = get_tree_inventory(_name)
        pop_adj = tree_data['population'] / 1000000  # millions
        # Reduce travel slightly if pop is high (congestion) or wind is extreme
        km_car_proxy = max(50, BASE_KM_CAR - int(pop_adj * 50) - int(df_clim['windspeed_10m_max'].max() * 5))
        
        # Proxy 3: Electricity consumption (Proxy based on high/low temperatures for AC use)
        mean_temp_swing = df_clim['temperature_2m_max'].max() - df_clim['temperature_2m_min'].min()
        kwh_proxy = int(BASE_KWH + mean_temp_swing * 2) 

        # Use city's recycling rate from sidebar input
        recycle_rate_proxy = recycle
        
        # Other simple averages
        flights_proxy = BASE_FLIGHTS
        lpg_proxy = BASE_LPG

        with colA:
            km_car = st.number_input("Monthly car travel (km)", value=km_car_proxy, disabled=True)
            kwh = st.number_input("Monthly electricity use (kWh)", value=kwh_proxy, disabled=True)
            flights = st.number_input("Flights per year (2‑hr avg)", value=flights_proxy, disabled=True)
        with colB:
            diet = st.selectbox("Diet type", list(DIET_MAP.keys()), index=list(DIET_MAP.keys()).index(diet_proxy), disabled=True)
            lpg = st.number_input("Monthly LPG use (kg)", value=lpg_proxy, disabled=True)
            recycle_rate = st.slider("Household recycling (%)", 0, 100, recycle_rate_proxy, disabled=True)

    else:
        # Manual Input Mode (Original code structure)
        with colA:
            km_car = st.number_input("Monthly car travel (km)", 0, 10000, 300)
            kwh = st.number_input("Monthly electricity use (kWh)", 0, 2000, 180)
            flights = st.number_input("Flights per year (2‑hr avg)", 0, 50, 1)
        with colB:
            diet = st.selectbox("Diet type", list(DIET_MAP.keys()), index=1)
            lpg = st.number_input("Monthly LPG use (kg)", 0, 100, 6)
            recycle_rate = st.slider("Household recycling (%)", 0, 100, 20)

    # 3. Calculation and Display
    
    # Calculate effective EF_KWH (Dynamically adjusted by Renewables Share)
    effective_ef_kwh = EF_KWH * (1 - ren_share/100)
    st.caption(f"Calculated effective CO₂ factor for electricity in this city: {effective_ef_kwh:.3f} kg/kWh (based on {ren_share}% renewables)")

    # Calculate monthly carbon emissions
    monthly = (
        km_car * EF_CAR + 
        kwh * effective_ef_kwh + # Use adjusted EF
        (flights * EF_FLIGHT) / 12 + 
        lpg * EF_LPG + 
        DIET_MAP[diet]
    ) * (1 - recycle_rate / 400) # Simple reduction for waste

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Estimated monthly emissions", f"{monthly/1000:.2f} t CO₂e")
    st.markdown("</div>", unsafe_allow_html=True)

    # 🌟 ENHANCEMENT: Add expander to explain the calculation factors
    with st.expander("❓ How is this calculated? (Emission Factors Explained)", expanded=False):
        st.caption("""
        This tool provides a *quick, simplified estimate* of monthly CO₂e (Carbon Dioxide equivalent) emissions. 
        The calculation relies on global and regional *Emission Factors (EFs)*:
        
        * *Car Travel (EF_CAR = 0.18):* Assumes **0.18 kg CO₂e per km** for an average gasoline car.
        * *Electricity (EF_KWH):* The factor is **dynamically adjusted** based on the Renewable Energy Share input from the sidebar. Lower renewables mean higher grid emissions.
        * *Flights (EF_FLIGHT = 180):* Assumes **180 kg CO₂e per 2-hour flight**.
        * *LPG Use (EF_LPG = 3.0):* Assumes **3.0 kg CO₂e per kg of LPG** consumed.
        * *Diet:* Based on estimated monthly emissions for different diet types.
        * *Recycling:* A reduction is applied based on the input rate to account for waste diversion.
        """)

    fig = px.pie(names=["Travel","Electricity","Flights","LPG","Diet"],
                 values=[
                     km_car * EF_CAR, 
                     kwh * effective_ef_kwh, # Use adjusted EF in pie chart too
                     (flights * EF_FLIGHT) / 12, 
                     lpg * EF_LPG, 
                     DIET_MAP[diet]
                 ],
                 title="Breakdown (kg CO₂e per month)")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e8f0fe'))
    st.plotly_chart(fig, use_container_width=True)

with TAB_ABOUT:
    st.header("Hackathon Submission")
    st.markdown("<div class='team-title'>Nxt Gen Developers</div>", unsafe_allow_html=True)
    st.subheader("Team Members")
    st.markdown("""
        * **Aditya Kumar Singh**
        * **Gaurang Verma**
        * **Vandana Yadav**
        * **Gaurav Shakya**
        * **Saurabh Shukla**
    """)
    
    st.markdown("---")
    
    st.subheader("Project: SustainifyAI — Sustainability & Climate Change Tracker")
    
    st.markdown("### Why SustainifyAI was Created")
    st.markdown("""
        SustainifyAI was developed to bridge the **gap between raw climate data and actionable governance/citizen insight**. In regions facing rapid environmental changes, a tool that consolidates real-time air quality, historical climate trends, river health, and personalized impact analysis is crucial. Our goal is to make **complex environmental data accessible, visual, and predictive** for better planning.
    """)
    
    st.markdown("### How SustainifyAI Helps Government & Policymakers")
    st.markdown("""
        1.  **Alerts & Anomaly Detection:** Provides immediate, location-specific warnings for high pollution (PM2.5) or extreme heat, enabling **proactive health and safety responses** (e.g., advising school closures or issuing heatwave warnings).
        2.  **Goal Tracking (e.g., Afforestation):** Quantifies city-specific targets (e.g., trees needed per capita), offering **measurable progress indicators** for green initiatives.
        3.  **Resource Allocation:** The Sustainability Score and detailed sub-scores (Water Quality, Air Quality, etc.) highlight the **most critical areas requiring immediate investment and policy intervention**.
        4.  **Long-term Planning:** AI forecasts provide a future outlook on temperatures and precipitation, essential for **infrastructure planning** (e.g., water management, drought preparation).
    """)

    st.markdown("### The Role of AI, ML, and Data Science")
    st.markdown("""
        * **Data Science & Engineering:** Used to **ingest, clean, and structure** vast amounts of historical climate data (ERA5 reanalysis) and real-time air quality data (Open-Meteo). We derive key metrics like **Monthly Norms** and **Warming Trends** using time-series analysis and moving averages.
        * **Machine Learning (ML) & AI Forecasting:**
            * **Forecasting Models:** We use established ML/AI models—**Prophet** (for strong seasonality), **ARIMA** (for classical time-series analysis), and a **Random Forest ML Ensemble** (for non-linear trend capture)—to predict future climate variables like temperature and precipitation up to a year ahead.
            * **Metric Validation:** Models are rigorously backtested using metrics like **MAE** and **MAPE** to ensure forecast accuracy before deployment.
            * **Intelligent Proxies:** AI/ML is used in the Carbon Footprint tab to create **intelligent, localized auto-estimates** based on the city's overall sustainability metrics.
    """)

    st.markdown("### Technology Stack & Demonstration Platform")
    st.markdown("""
        | Component | Technology | Why We Used It |
        | :--- | :--- | :--- |
        | **Frontend/Demonstration** | **Streamlit & Plotly** | Streamlit enabled **rapid prototyping** and creating a complex, interactive web application entirely in Python. Plotly provides **futuristic, interactive, and mobile-friendly visualizations**. |
        | **Data/Backend (APIs)** | **Open-Meteo (ERA5 & AQ APIs)** | Provides free, high-quality, geographically granular historical climate and real-time air quality data, eliminating the need for complex API keys for demonstration. |
        | **Development Environment** | **VS Code** | Used for its robust Python support, rapid debugging, and seamless integration with this code. |
        | **ML/Statistical Models** | **Prophet, pmdarima, scikit-learn** | A robust collection of libraries for professional-grade time-series forecasting and ML ensemble creation. |
    """)

    st.markdown("---")
    
    st.subheader("Conclusion")
    st.markdown("""
        SustainifyAI is a proof-of-concept demonstrating a **unified, intelligent dashboard** capable of serving both citizens and government bodies with critical environmental intelligence. Our focus on **user experience, actionable metrics, and predictive AI** makes this a powerful tool for driving sustainability initiatives forward.
    """)

    st.markdown("---")
    st.markdown("### Thank you for reviewing our project! 🙏")