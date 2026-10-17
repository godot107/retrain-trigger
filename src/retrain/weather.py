"""Hourly temperature per grid operator from the Open-Meteo historical archive (ERA5-based reanalysis).

Free for non-commercial use, no API key, data CC BY 4.0 (Open-Meteo.com). One representative city per
operator: a crude proxy for operators that span several states (MISO, PJM, SWPP), stated as a limit.

Observed temperature is used as if it were tomorrow's forecast, i.e. a *perfect* weather forecast. A
real day-ahead forecaster would use a forecast with its own error; this makes the weather model an
upper bound on what weather inputs can do.
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[2] / "data" / "weather"
USER_AGENT = "retrain-trigger/0.1 (contact@willieman.com)"
CITIES = {  # operator: (city, lat, lon)
    "ERCO": ("Dallas", 32.78, -96.80), "PJM": ("Philadelphia", 39.95, -75.17),
    "CISO": ("Los Angeles", 34.05, -118.24), "MISO": ("Minneapolis", 44.98, -93.27),
    "SWPP": ("Oklahoma City", 35.47, -97.52), "NYIS": ("New York", 40.71, -74.01),
    "ISNE": ("Boston", 42.36, -71.06), "SOCO": ("Atlanta", 33.75, -84.39),
    "TVA": ("Nashville", 36.16, -86.78), "BPAT": ("Portland", 45.52, -122.68),
    "DUK": ("Charlotte", 35.23, -80.84), "FPL": ("Miami", 25.76, -80.19),
}


def fetch(ba, start="2018-12-25", end="2025-12-31"):
    path = DIR / f"{ba}.json"
    if not path.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        _, lat, lon = CITIES[ba]
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={start}&end_date={end}&hourly=temperature_2m&timezone=UTC")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        if "hourly" not in data:
            raise ValueError(f"unexpected Open-Meteo response for {ba}: {str(data)[:200]}")
        path.write_text(json.dumps(data))
    data = json.loads(path.read_text())
    s = pd.Series(data["hourly"]["temperature_2m"], index=pd.to_datetime(data["hourly"]["time"], utc=True),
                  name="temp", dtype=float)
    return s


def fetch_forecast(ba, start="2021-06-01", end="2025-12-31"):
    """Day-ahead temperature forecasts as they were issued (Open-Meteo Previous Runs API,
    `temperature_2m_previous_day1` = the forecast made one day earlier). Available from mid-2021."""
    path = DIR / f"{ba}_forecast.json"
    if not path.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        _, lat, lon = CITIES[ba]
        url = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&start_date={start}&end_date={end}&hourly=temperature_2m_previous_day1&timezone=UTC")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
        if "hourly" not in data:
            raise ValueError(f"unexpected Open-Meteo response for {ba}: {str(data)[:200]}")
        path.write_text(json.dumps(data))
    data = json.loads(path.read_text())
    return pd.Series(data["hourly"]["temperature_2m_previous_day1"],
                     index=pd.to_datetime(data["hourly"]["time"], utc=True), name="temp_fc", dtype=float)
