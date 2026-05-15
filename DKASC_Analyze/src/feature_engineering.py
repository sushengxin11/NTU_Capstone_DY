from __future__ import annotations

import numpy as np
import pandas as pd

from .config import IRRADIANCE_COLUMNS, TIMESTAMP_COL


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out[TIMESTAMP_COL]
    out["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    out["month"] = ts.dt.month
    out["day_of_year"] = ts.dt.dayofyear
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["sin_day_of_year"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["cos_day_of_year"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    return out


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Wind_Direction" in out.columns:
        radians = np.deg2rad(out["Wind_Direction"])
        out["wind_direction_sin"] = np.sin(radians)
        out["wind_direction_cos"] = np.cos(radians)
    if "Weather_Daily_Rainfall" in out.columns:
        out["rain_indicator"] = (out["Weather_Daily_Rainfall"].fillna(0) > 0).astype(int)
    if {"Global_Horizontal_Radiation", "Radiation_Global_Tilted"}.issubset(out.columns):
        denom = out["Global_Horizontal_Radiation"].replace(0, np.nan)
        out["tilted_to_global_ratio"] = out["Radiation_Global_Tilted"] / denom
        out["tilted_to_global_ratio"] = out["tilted_to_global_ratio"].replace([np.inf, -np.inf], np.nan)
    return out


def add_lag_rolling_features(df: pd.DataFrame, include_lag_features: bool = False) -> pd.DataFrame:
    if not include_lag_features:
        return df
    out = df.copy()
    lag_steps = {"lag_5min": 1, "lag_15min": 3, "lag_30min": 6, "lag_60min": 12}
    for col in [c for c in IRRADIANCE_COLUMNS + ["Weather_Temperature_Celsius", "Weather_Relative_Humidity"] if c in out.columns]:
        for suffix, steps in lag_steps.items():
            out[f"{col}_{suffix}"] = out[col].shift(steps)
    for col in [c for c in IRRADIANCE_COLUMNS if c in out.columns]:
        out[f"{col}_roll_mean_15min"] = out[col].rolling(window=3, min_periods=1).mean()
        out[f"{col}_roll_std_30min"] = out[col].rolling(window=6, min_periods=2).std()
    return out


def build_features(df: pd.DataFrame, include_lag_features: bool = False) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_weather_features(out)
    out = add_lag_rolling_features(out, include_lag_features=include_lag_features)
    return out


def existing_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]
