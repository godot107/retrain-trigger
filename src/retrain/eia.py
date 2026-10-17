"""Real data: EIA-930 hourly electricity demand, 2019-2025, turned into weekly forecasting batches.

Source file: the grid-mining project's tidy parquet (EIA-930 BALANCE files, public domain, no API
key). Cleaning ported from grid-mining's `decompose.hourly` (same rules, same reasons):
- EIA stamps each value at the END of its hour: shift back one hour so a day is 00:00-24:00;
- zero/negative demand, and values outside 0.25x-3x the trailing 14-day median, are reporting errors
  (PJM once reported ~2.1 billion MW): they become gaps;
- gaps up to 6 h are interpolated; longer gaps stay missing (those rows are dropped).

The task: a **day-ahead** forecast. At the end of day D-1, predict each hour of day D from the
calendar and loads already known (same hour yesterday, two days ago, last week; yesterday's mean
and peak; the same hour's mean over the last 7 days). No weather: the model is deliberately simple,
which is fine for comparing *retraining triggers*, the point here.

All loads are divided by the operator's 2019 mean, so errors read as a fraction of typical load and
are comparable across operators.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

GRID_MINING = Path(os.environ.get("GRID_MINING_DIR", Path(__file__).resolve().parents[3] / "grid-mining"))
PARQUET = GRID_MINING / "data" / "processed" / "balance.parquet"
TZ = {"ERCO": "America/Chicago", "PJM": "America/New_York", "CISO": "America/Los_Angeles",
      "MISO": "America/Chicago", "SWPP": "America/Chicago", "NYIS": "America/New_York",
      "ISNE": "America/New_York", "SOCO": "America/Chicago", "TVA": "America/Chicago",
      "BPAT": "America/Los_Angeles", "DUK": "America/New_York", "FPL": "America/New_York"}
BAS = tuple(TZ)
FEATURES = ["hour", "dow", "doy", "holiday", "lag24", "lag48", "lag168", "prev_mean", "prev_max", "same_hour_7d"]
WEATHER_FEATURES = ["temp", "temp_lag24", "temp_day_mean"]   # optional (weather=True): see weather.py
DRIFT_FEATURES = ["lag24", "lag48", "lag168", "prev_mean", "prev_max", "same_hour_7d"]   # calendar always "drifts"
EVENTS = {"COVID lockdowns": "2020-03-16", "Winter Storm Uri": "2021-02-13",
          "Winter Storm Elliott": "2022-12-23"}


def load_raw():
    if not PARQUET.exists():
        raise FileNotFoundError(f"{PARQUET} missing: build it in grid-mining (eia.build()) or set GRID_MINING_DIR")
    return pd.read_parquet(PARQUET, columns=["ba", "utc", "demand"])


def hourly(df, ba, max_gap=6):
    s = df.loc[df.ba == ba].set_index("utc")["demand"].astype(float).sort_index()
    s.index = s.index - pd.Timedelta(hours=1)
    s = s.where(s > 0)
    med = s.rolling("14D", min_periods=24).median()
    s = s.where(med.isna() | ((s > 0.25 * med) & (s < 3 * med)))
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    gap = s.isna()
    run = gap.groupby((~gap).cumsum()).transform("sum")
    s = s.interpolate(limit_area="inside").where(~gap | (run <= max_gap))
    return s.tz_convert(TZ[ba])


def add_weather(f, ba):
    """Temperature at the target hour, the same hour yesterday, and the target day's mean
    (observed values standing in for a perfect day-ahead forecast)."""
    from retrain import weather
    t = weather.fetch(ba).tz_convert(f.index.tz)
    t = t.reindex(f.index.union(t.index)).interpolate(limit=3).reindex(f.index)
    f = f.assign(temp=t.values, temp_lag24=t.shift(24).values)
    f["temp_day_mean"] = f.groupby(f.index.normalize()).temp.transform("mean")
    # Serving inputs: the day-ahead *forecast* that was available, where archived (mid-2021 on);
    # observed before that. Yesterday's temperature (temp_lag24) is already known: stays observed.
    fc = weather.fetch_forecast(ba).tz_convert(f.index.tz)
    fc = fc.reindex(fc.index.union(f.index)).interpolate(limit=3).reindex(f.index)
    f["temp_srv"] = fc.fillna(f["temp"]).values
    f["temp_day_mean_srv"] = f.groupby(f.index.normalize()).temp_srv.transform("mean")
    return f.dropna(subset=["temp", "temp_lag24", "temp_day_mean"])


def features(s):
    """One row per hour with the day-ahead features; every lag is at least 24 h old, and the daily
    summaries use the previous calendar day only, so nothing from day D leaks into its own forecast."""
    df = pd.DataFrame({"load": s})
    idx = df.index
    day = idx.normalize()
    daily = df.load.groupby(day).agg(["mean", "max"])
    prev = daily.shift(1).reindex(day)
    hol = USFederalHolidayCalendar().holidays(idx.min().tz_localize(None), idx.max().tz_localize(None))
    out = pd.DataFrame({
        "hour": idx.hour, "dow": idx.dayofweek, "doy": idx.dayofyear,
        "holiday": day.tz_localize(None).isin(hol).astype(int),
        "lag24": df.load.shift(24).values, "lag48": df.load.shift(48).values,
        "lag168": df.load.shift(168).values,
        "prev_mean": prev["mean"].values, "prev_max": prev["max"].values,
        "same_hour_7d": sum(df.load.shift(24 * k) for k in range(1, 8)).values / 7,
        "load": df.load.values}, index=idx)
    return out.dropna()


def stream(ba, raw=None, start="2019-01-07", train_weeks=52, val_weeks=4, weather=False):
    """weather: False | True (observed temperature everywhere: a perfect forecast) | "forecast"
    (train on observed, predict and monitor with the archived day-ahead forecast where it exists)."""
    raw = load_raw() if raw is None else raw
    f = features(hourly(raw, ba))
    feats, drift = list(FEATURES), list(DRIFT_FEATURES)
    if weather:
        f = add_weather(f, ba)
        feats += WEATHER_FEATURES
        drift += ["temp"]
    f = f[f.index >= pd.Timestamp(start, tz=f.index.tz)]
    scale = f.loc[f.index.year == 2019, "load"].mean()
    for c in ("load", *DRIFT_FEATURES):
        f[c] = f[c] / scale
    week = f.index.tz_localize(None).to_period("W-SUN")
    serve = [{"temp": "temp_srv", "temp_day_mean": "temp_day_mean_srv"}.get(c, c) for c in feats] \
        if weather == "forecast" else None
    X, y, weeks, Xs = [], [], [], []
    for w, g in f.groupby(week):
        if len(g) < 24 * 5:      # skip weeks mostly lost to data gaps
            continue
        X.append(g[feats].to_numpy(float))
        if serve:
            Xs.append(g[serve].to_numpy(float))
        y.append(g["load"].to_numpy(float))
        weeks.append(w.start_time)
    return SimpleNamespace(scenario=ba, X=X, y=y, weeks=weeks, onset=None, harmful=False,
                           train_batches=train_weeks, val_batches=val_weeks, season=52,
                           drift_cols=[feats.index(c) for c in drift], scale=float(scale), weather=weather,
                           X_serve=Xs or None)
