from __future__ import annotations

import pandas as pd

from .config import (
    HUMIDITY_COLUMNS,
    IRRADIANCE_COLUMNS,
    RAINFALL_COLUMNS,
    TEMPERATURE_COLUMNS,
    TIME_COLUMNS,
    WIND_COLUMNS,
)
from .feature_engineering import existing_columns
from .modeling import make_random_forest_model, regression_metrics, chronological_split, deterministic_downsample


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "baseline_time_only": TIME_COLUMNS,
        "baseline_irradiance_only": IRRADIANCE_COLUMNS,
        "time_plus_irradiance": TIME_COLUMNS + IRRADIANCE_COLUMNS,
        "time_irradiance_temperature": TIME_COLUMNS + IRRADIANCE_COLUMNS + TEMPERATURE_COLUMNS,
        "time_irradiance_humidity": TIME_COLUMNS + IRRADIANCE_COLUMNS + HUMIDITY_COLUMNS,
        "time_irradiance_wind": TIME_COLUMNS + IRRADIANCE_COLUMNS + WIND_COLUMNS,
        "time_irradiance_rainfall": TIME_COLUMNS + IRRADIANCE_COLUMNS + RAINFALL_COLUMNS,
        "full_weather": TIME_COLUMNS
        + IRRADIANCE_COLUMNS
        + TEMPERATURE_COLUMNS
        + HUMIDITY_COLUMNS
        + WIND_COLUMNS
        + RAINFALL_COLUMNS,
    }
    return {name: existing_columns(df, list(dict.fromkeys(cols))) for name, cols in groups.items()}


def run_ablation(
    df: pd.DataFrame,
    target_col: str,
    random_state: int,
    max_rf_train_rows: int,
    mape_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, features in feature_groups(df).items():
        if not features:
            rows.append({"feature_group": group_name, "model": "RandomForest", "status": "skipped_no_features"})
            continue
        model_df = df[features + [target_col]].dropna(subset=[target_col]).copy()
        if len(model_df) < 100:
            rows.append({"feature_group": group_name, "model": "RandomForest", "status": "skipped_too_few_rows"})
            continue
        train_df, val_df, test_df = chronological_split(model_df)
        fit_df = deterministic_downsample(train_df, max_rf_train_rows)
        model = make_random_forest_model(random_state)
        model.fit(fit_df[features], fit_df[target_col])
        for split_name, split_df in [("validation", val_df), ("test", test_df)]:
            pred = model.predict(split_df[features])
            metrics = regression_metrics(split_df[target_col], pred, mape_threshold)
            rows.append(
                {
                    "feature_group": group_name,
                    "model": "RandomForest",
                    "split": split_name,
                    "status": "ok",
                    "feature_count": len(features),
                    "features": ", ".join(features),
                    **metrics,
                    "n": len(split_df),
                }
            )
    return pd.DataFrame(rows)
