from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TIMESTAMP_COL


def read_csv_with_timestamp(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    df = pd.read_csv(path, usecols=usecols)
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"{path.name} does not contain required column '{TIMESTAMP_COL}'")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    invalid = df[TIMESTAMP_COL].isna().sum()
    if invalid:
        print(f"WARNING: {path.name} has {invalid} rows with unparseable timestamps; dropping them.")
        df = df.dropna(subset=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    return df


def summarize_dataframe(df: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    summary: dict[str, object] = {
        "dataset": dataset_name,
        "rows": len(df),
        "columns": len(df.columns),
        "start_time": df[TIMESTAMP_COL].min() if TIMESTAMP_COL in df else pd.NaT,
        "end_time": df[TIMESTAMP_COL].max() if TIMESTAMP_COL in df else pd.NaT,
    }
    if TIMESTAMP_COL in df.columns and len(df) > 1:
        diffs = df[TIMESTAMP_COL].diff().dropna()
        summary["median_interval"] = str(diffs.median())
        summary["most_common_interval"] = str(diffs.value_counts().index[0]) if not diffs.empty else ""
        summary["duplicate_timestamps_after_dedup"] = int(df[TIMESTAMP_COL].duplicated().sum())
    return summary


def missing_value_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "dataset": dataset_name,
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
        }
    )
    return out.sort_values(["dataset", "missing_rate", "column"], ascending=[True, False, True])
