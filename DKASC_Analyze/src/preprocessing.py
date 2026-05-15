from __future__ import annotations

import numpy as np
import pandas as pd

from .config import IRRADIANCE_COLUMNS, TARGET_COL, TIMESTAMP_COL, WEATHER_COLUMNS


def align_label_with_weather(
    label_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    if target_col not in label_df.columns:
        raise ValueError(f"PV label file does not contain target column '{target_col}'")

    label_cols = [TIMESTAMP_COL, target_col]
    optional_label_cols = [
        "Active_Energy_Delivered_Received",
        "Current_Phase_Average",
        "Power_Factor_Signed",
        "Average_Voltage_Line_to_Neutral",
        "Frequency",
        "THD_Voltage_Average",
    ]
    label_cols += [col for col in optional_label_cols if col in label_df.columns]

    available_weather = [col for col in WEATHER_COLUMNS if col in weather_df.columns]
    if not available_weather:
        raise ValueError("Weather file does not contain any recognized weather feature columns.")

    merged = pd.merge(
        label_df[label_cols],
        weather_df[[TIMESTAMP_COL] + available_weather],
        on=TIMESTAMP_COL,
        how="inner",
        validate="one_to_one",
    )
    return merged.sort_values(TIMESTAMP_COL)


def clean_merged_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    capacity_kw: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()
    checks: list[dict[str, object]] = []

    numeric_cols = [col for col in cleaned.columns if col != TIMESTAMP_COL]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if target_col in cleaned.columns:
        neg_count = int((cleaned[target_col] < 0).sum())
        checks.append({"check": "negative_target_values", "count": neg_count})
        if capacity_kw:
            high_limit = capacity_kw * 1.2
            high_count = int((cleaned[target_col] > high_limit).sum())
            checks.append({"check": f"target_above_{high_limit:.2f}_kw", "count": high_count})
            cleaned.loc[cleaned[target_col] > high_limit, target_col] = np.nan
        cleaned.loc[cleaned[target_col] < 0, target_col] = np.nan

    for col in IRRADIANCE_COLUMNS:
        if col in cleaned.columns:
            neg_count = int((cleaned[col] < 0).sum())
            checks.append({"check": f"{col}_negative", "count": neg_count})
            cleaned.loc[cleaned[col] < 0, col] = np.nan

    if "Weather_Relative_Humidity" in cleaned.columns:
        bad = ~cleaned["Weather_Relative_Humidity"].between(0, 100) & cleaned["Weather_Relative_Humidity"].notna()
        checks.append({"check": "humidity_outside_0_100", "count": int(bad.sum())})
        cleaned.loc[bad, "Weather_Relative_Humidity"] = np.nan

    if "Wind_Direction" in cleaned.columns:
        bad = ~cleaned["Wind_Direction"].between(0, 360) & cleaned["Wind_Direction"].notna()
        checks.append({"check": "wind_direction_outside_0_360", "count": int(bad.sum())})
        cleaned.loc[bad, "Wind_Direction"] = np.nan

    if "Wind_Speed" in cleaned.columns:
        bad = (cleaned["Wind_Speed"] < 0) & cleaned["Wind_Speed"].notna()
        checks.append({"check": "wind_speed_negative", "count": int(bad.sum())})
        cleaned.loc[bad, "Wind_Speed"] = np.nan

    if "Weather_Daily_Rainfall" in cleaned.columns:
        bad = (cleaned["Weather_Daily_Rainfall"] < 0) & cleaned["Weather_Daily_Rainfall"].notna()
        checks.append({"check": "daily_rainfall_negative", "count": int(bad.sum())})
        cleaned.loc[bad, "Weather_Daily_Rainfall"] = np.nan

    cleaned = cleaned.dropna(subset=[target_col])
    return cleaned, pd.DataFrame(checks)


def make_daylight_mask(df: pd.DataFrame, irradiance_threshold: float = 20.0) -> pd.Series:
    for col in ["Radiation_Global_Tilted", "Global_Horizontal_Radiation"]:
        if col in df.columns:
            return df[col].fillna(0) > irradiance_threshold
    return df[TIMESTAMP_COL].dt.hour.between(6, 18)


def interpolate_weather_only(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    out = df.copy()
    weather_like_cols = [
        col
        for col in out.columns
        if col not in {TIMESTAMP_COL, target_col}
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    if weather_like_cols:
        out[weather_like_cols] = out[weather_like_cols].interpolate(limit=3, limit_direction="both")
    return out
