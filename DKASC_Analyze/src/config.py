from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TIMESTAMP_COL = "timestamp"
TARGET_COL = "Active_Power"
ENERGY_COL = "Active_Energy_Delivered_Received"

WEATHER_COLUMNS = [
    "Wind_Speed",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Wind_Direction",
    "Weather_Daily_Rainfall",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]

IRRADIANCE_COLUMNS = [
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
]

TEMPERATURE_COLUMNS = ["Weather_Temperature_Celsius"]
HUMIDITY_COLUMNS = ["Weather_Relative_Humidity"]
WIND_COLUMNS = ["Wind_Speed", "wind_direction_sin", "wind_direction_cos"]
RAINFALL_COLUMNS = ["Weather_Daily_Rainfall", "rain_indicator"]

TIME_COLUMNS = [
    "hour",
    "month",
    "day_of_year",
    "sin_hour",
    "cos_hour",
    "sin_day_of_year",
    "cos_day_of_year",
]


@dataclass(frozen=True)
class LabelConfig:
    key: str
    display_name: str
    path: Path
    target_col: str = TARGET_COL
    capacity_kw: float | None = None
    enabled: bool = True


@dataclass(frozen=True)
class AnalysisConfig:
    project_dir: Path
    weather_path: Path
    primary_label: LabelConfig
    total_label: LabelConfig | None
    output_dir: Path
    processed_dir: Path
    figures_dir: Path
    results_dir: Path
    daylight_irradiance_threshold: float = 20.0
    target_mape_threshold: float = 0.05
    random_state: int = 42
    max_rf_train_rows: int = 120_000
    max_plot_rows: int = 35_000


def ensure_output_dirs(config: AnalysisConfig) -> None:
    for path in [
        config.output_dir,
        config.processed_dir,
        config.figures_dir,
        config.results_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
