from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import IRRADIANCE_COLUMNS, TARGET_COL, TIMESTAMP_COL


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def sample_for_plot(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42).sort_values(TIMESTAMP_COL)


def plot_pv_timeseries(df: pd.DataFrame, target_col: str, path: Path, title: str) -> None:
    plot_df = df[[TIMESTAMP_COL, target_col]].set_index(TIMESTAMP_COL).resample("D").mean().reset_index()
    plt.figure(figsize=(13, 4))
    sns.lineplot(data=plot_df, x=TIMESTAMP_COL, y=target_col, linewidth=0.8)
    plt.title(title)
    plt.ylabel("Daily mean PV power (kW)")
    savefig(path)


def plot_weather_timeseries(df: pd.DataFrame, path: Path) -> None:
    cols = [c for c in ["Global_Horizontal_Radiation", "Radiation_Global_Tilted", "Weather_Temperature_Celsius", "Weather_Relative_Humidity", "Wind_Speed"] if c in df.columns]
    if not cols:
        return
    plot_df = df[[TIMESTAMP_COL] + cols].set_index(TIMESTAMP_COL).resample("D").mean().reset_index()
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 2.4 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.lineplot(data=plot_df, x=TIMESTAMP_COL, y=col, ax=ax, linewidth=0.8)
        ax.set_ylabel(col)
    savefig(path)


def plot_scatter(df: pd.DataFrame, x_col: str, target_col: str, path: Path, max_rows: int) -> None:
    if x_col not in df.columns:
        return
    plot_df = sample_for_plot(df[[TIMESTAMP_COL, x_col, target_col]].dropna(), max_rows)
    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(data=plot_df, x=x_col, y=target_col, s=8, alpha=0.25, edgecolor=None)
    plt.title(f"{x_col} vs {target_col}")
    savefig(path)


def plot_correlation_heatmap(df: pd.DataFrame, features: list[str], target_col: str, path: Path) -> None:
    cols = [c for c in features + [target_col] if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    plt.figure(figsize=(max(8, len(cols) * 0.45), max(6, len(cols) * 0.35)))
    sns.heatmap(corr, cmap="vlag", center=0, square=False)
    plt.title("Feature correlation heatmap")
    savefig(path)


def plot_importance(table: pd.DataFrame, feature_col: str, value_col: str, path: Path, title: str, top_n: int = 15) -> None:
    if table.empty or value_col not in table.columns:
        return
    plot_df = table.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, len(plot_df) * 0.35)))
    sns.barplot(data=plot_df, x=value_col, y=feature_col, color="#377eb8")
    plt.title(title)
    savefig(path)


def plot_ablation(ablation_df: pd.DataFrame, path: Path) -> None:
    ok = ablation_df[(ablation_df.get("status") == "ok") & (ablation_df.get("split") == "test")].copy()
    if ok.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.barplot(data=ok, x="feature_group", y="RMSE", color="#4daf4a")
    plt.xticks(rotation=35, ha="right")
    plt.title("Ablation results on test set")
    savefig(path)


def plot_predictions(predictions: pd.DataFrame, original_df: pd.DataFrame, path: Path) -> None:
    if predictions.empty:
        return
    plot_df = predictions.copy()
    plot_df[TIMESTAMP_COL] = original_df.loc[predictions.index, TIMESTAMP_COL].values
    plot_df = plot_df.set_index(TIMESTAMP_COL).resample("H").mean().reset_index()
    if len(plot_df) > 24 * 30:
        plot_df = plot_df.tail(24 * 30)
    plt.figure(figsize=(13, 4.5))
    sns.lineplot(data=plot_df, x=TIMESTAMP_COL, y="actual", label="Actual", linewidth=1)
    if "predicted_RandomForest" in plot_df:
        sns.lineplot(data=plot_df, x=TIMESTAMP_COL, y="predicted_RandomForest", label="RandomForest", linewidth=1)
    if "predicted_Ridge" in plot_df:
        sns.lineplot(data=plot_df, x=TIMESTAMP_COL, y="predicted_Ridge", label="Ridge", linewidth=1)
    plt.title("Predicted vs actual PV output")
    plt.ylabel("Hourly mean PV power (kW)")
    savefig(path)


def plot_daily_profile(df: pd.DataFrame, target_col: str, path: Path) -> None:
    plot_cols = [target_col] + [c for c in ["Global_Horizontal_Radiation", "Radiation_Global_Tilted"] if c in df.columns]
    plot_df = df[[TIMESTAMP_COL] + plot_cols].copy()
    plot_df["time_of_day"] = plot_df[TIMESTAMP_COL].dt.hour + plot_df[TIMESTAMP_COL].dt.minute / 60
    profile = plot_df.groupby("time_of_day")[plot_cols].mean().reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=profile, x="time_of_day", y=target_col, ax=ax1, color="#1b9e77", label="PV power")
    ax1.set_ylabel("PV power (kW)")
    ax2 = ax1.twinx()
    for col in [c for c in IRRADIANCE_COLUMNS if c in profile.columns][:2]:
        sns.lineplot(data=profile, x="time_of_day", y=col, ax=ax2, label=col, linewidth=1)
    ax2.set_ylabel("Irradiance")
    ax1.set_title("Average daily profile")
    savefig(path)
