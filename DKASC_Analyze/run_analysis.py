from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ablation import run_ablation
from src.analysis import correlation_table, mutual_information_table
from src.config import (
    HUMIDITY_COLUMNS,
    IRRADIANCE_COLUMNS,
    RAINFALL_COLUMNS,
    TEMPERATURE_COLUMNS,
    TIME_COLUMNS,
    WIND_COLUMNS,
    AnalysisConfig,
    LabelConfig,
    ensure_output_dirs,
)
from src.data_loader import missing_value_summary, read_csv_with_timestamp, summarize_dataframe
from src.feature_engineering import build_features, existing_columns
from src.modeling import train_and_evaluate_models
from src.preprocessing import align_label_with_weather, clean_merged_data, interpolate_weather_only, make_daylight_mask
from src.utils import write_csv
from src.visualization import (
    plot_ablation,
    plot_correlation_heatmap,
    plot_daily_profile,
    plot_importance,
    plot_predictions,
    plot_pv_timeseries,
    plot_scatter,
    plot_weather_timeseries,
    set_plot_style,
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    default_raw_dir = project_dir.parent / "Capstone"
    parser = argparse.ArgumentParser(description="DKASC Alice Springs PV weather feature analysis")
    parser.add_argument("--weather-csv", type=Path, default=default_raw_dir / "101-Site_DKA-WeatherStation.csv")
    parser.add_argument("--pv-1a-csv", type=Path, default=default_raw_dir / "91-Site_DKA-M9_B-Phase.csv")
    parser.add_argument("--total-csv", type=Path, default=default_raw_dir / "96-Site_DKA-MasterMeter1.csv")
    parser.add_argument("--include-total-model", action="store_true", help="Run optional total-site model after manual sign validation.")
    parser.add_argument("--include-lag-features", action="store_true", help="Add lag and rolling weather features.")
    parser.add_argument("--daylight-irradiance-threshold", type=float, default=20.0)
    parser.add_argument("--max-rf-train-rows", type=int, default=120_000)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    project_dir = Path(__file__).resolve().parent
    return AnalysisConfig(
        project_dir=project_dir,
        weather_path=args.weather_csv,
        primary_label=LabelConfig(
            key="1a_trina",
            display_name="1A Trina, 10.5kW, mono-Si, Dual, 2009",
            path=args.pv_1a_csv,
            capacity_kw=10.5,
        ),
        total_label=LabelConfig(
            key="total_sites",
            display_name="0 263.0kW, Total of all sites",
            path=args.total_csv,
            capacity_kw=263.0,
            enabled=args.include_total_model,
        ),
        output_dir=project_dir,
        processed_dir=project_dir / "data" / "processed",
        figures_dir=project_dir / "figures",
        results_dir=project_dir / "results",
        daylight_irradiance_threshold=args.daylight_irradiance_threshold,
        max_rf_train_rows=args.max_rf_train_rows,
    )


def all_candidate_features(df: pd.DataFrame) -> list[str]:
    base = (
        TIME_COLUMNS
        + IRRADIANCE_COLUMNS
        + TEMPERATURE_COLUMNS
        + HUMIDITY_COLUMNS
        + WIND_COLUMNS
        + RAINFALL_COLUMNS
        + ["Wind_Direction", "tilted_to_global_ratio"]
    )
    lag_like = [c for c in df.columns if "_lag_" in c or "_roll_" in c]
    return existing_columns(df, list(dict.fromkeys(base + lag_like)))


def run_label_analysis(
    label_config: LabelConfig,
    weather_df: pd.DataFrame,
    config: AnalysisConfig,
    include_lag_features: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Analyzing label: {label_config.display_name}")
    label_df = read_csv_with_timestamp(label_config.path)
    merged = align_label_with_weather(label_df, weather_df, target_col=label_config.target_col)
    cleaned, quality_checks = clean_merged_data(
        merged, target_col=label_config.target_col, capacity_kw=label_config.capacity_kw
    )
    cleaned = interpolate_weather_only(cleaned, target_col=label_config.target_col)
    featured = build_features(cleaned, include_lag_features=include_lag_features)
    featured["is_daylight"] = make_daylight_mask(
        featured, irradiance_threshold=config.daylight_irradiance_threshold
    )
    daylight = featured[featured["is_daylight"]].copy()

    processed_path = config.processed_dir / f"cleaned_{label_config.key}.csv"
    write_csv(featured, processed_path)
    write_csv(quality_checks, config.results_dir / f"quality_checks_{label_config.key}.csv")

    features = all_candidate_features(featured)
    daylight_features = daylight.dropna(subset=[label_config.target_col])
    corr_day = correlation_table(daylight_features, features, label_config.target_col)
    corr_all = correlation_table(featured, features, label_config.target_col)
    corr_day["sample_scope"] = "daylight"
    corr_all["sample_scope"] = "all_day"
    corr = pd.concat([corr_day, corr_all], ignore_index=True)
    write_csv(corr, config.results_dir / f"feature_correlation_{label_config.key}.csv")

    mi = mutual_information_table(
        daylight_features, features, label_config.target_col, random_state=config.random_state
    )
    write_csv(mi, config.results_dir / f"mutual_information_{label_config.key}.csv")

    metrics, models, rf_importance, perm_importance, predictions = train_and_evaluate_models(
        daylight_features,
        features,
        label_config.target_col,
        random_state=config.random_state,
        max_rf_train_rows=config.max_rf_train_rows,
        mape_threshold=config.target_mape_threshold,
    )
    metrics["label"] = label_config.key
    write_csv(rf_importance, config.results_dir / f"feature_importance_{label_config.key}.csv")
    write_csv(perm_importance, config.results_dir / f"permutation_importance_{label_config.key}.csv")

    ablation = run_ablation(
        daylight_features,
        label_config.target_col,
        random_state=config.random_state,
        max_rf_train_rows=config.max_rf_train_rows,
        mape_threshold=config.target_mape_threshold,
    )
    write_csv(ablation, config.results_dir / f"ablation_results_{label_config.key}.csv")

    set_plot_style()
    plot_pv_timeseries(
        featured,
        label_config.target_col,
        config.figures_dir / f"pv_output_timeseries_{label_config.key}.png",
        f"PV output time series - {label_config.display_name}",
    )
    plot_weather_timeseries(featured, config.figures_dir / "weather_variable_timeseries.png")
    scatter_map = {
        "Global_Horizontal_Radiation": "irradiance",
        "Weather_Temperature_Celsius": "temperature",
        "Weather_Relative_Humidity": "humidity",
        "Wind_Speed": "wind_speed",
    }
    for col, suffix in scatter_map.items():
        plot_scatter(
            daylight_features,
            col,
            label_config.target_col,
            config.figures_dir / f"{suffix}_vs_pv_{label_config.key}.png",
            config.max_plot_rows,
        )
    plot_correlation_heatmap(
        daylight_features,
        features,
        label_config.target_col,
        config.figures_dir / f"correlation_heatmap_{label_config.key}.png",
    )
    plot_importance(
        rf_importance,
        "feature",
        "importance",
        config.figures_dir / f"feature_importance_{label_config.key}.png",
        "Random Forest feature importance",
    )
    plot_importance(
        perm_importance,
        "feature",
        "importance_mean",
        config.figures_dir / f"permutation_importance_{label_config.key}.png",
        "Permutation importance",
    )
    plot_ablation(ablation, config.figures_dir / f"ablation_results_{label_config.key}.png")
    plot_predictions(
        predictions,
        daylight_features,
        config.figures_dir / f"predicted_vs_actual_{label_config.key}.png",
    )
    plot_daily_profile(
        featured,
        label_config.target_col,
        config.figures_dir / f"daily_profile_{label_config.key}.png",
    )

    return metrics, featured


def total_sign_diagnostic(total_path: Path, weather_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    total_df = read_csv_with_timestamp(total_path)
    merged = align_label_with_weather(total_df, weather_df)
    featured = build_features(merged)
    featured["is_daylight"] = make_daylight_mask(featured, config.daylight_irradiance_threshold)
    target = "Active_Power"
    rows = []
    for scope, subset in [("all_day", featured), ("daylight", featured[featured["is_daylight"]])]:
        values = subset[target].dropna()
        rows.append(
            {
                "scope": scope,
                "rows": len(values),
                "mean_active_power": values.mean(),
                "median_active_power": values.median(),
                "min_active_power": values.min(),
                "max_active_power": values.max(),
                "negative_rate": (values < 0).mean(),
                "positive_rate": (values > 0).mean(),
                "zero_rate": (values == 0).mean(),
            }
        )
    diagnostic = pd.DataFrame(rows)
    write_csv(diagnostic, config.results_dir / "total_sites_sign_diagnostic.csv")
    return diagnostic


def write_readme(config: AnalysisConfig, metrics: pd.DataFrame, total_diagnostic: pd.DataFrame | None) -> None:
    readme_path = config.project_dir / "README.md"
    best_rows = metrics[(metrics["split"] == "test") & (metrics["model"] == "RandomForest")]
    metric_text = "尚未生成模型指标。"
    if not best_rows.empty:
        row = best_rows.iloc[0]
        metric_text = (
            f"Random Forest 在 daylight 测试集上的结果：MAE={row['MAE']:.4f}, "
            f"RMSE={row['RMSE']:.4f}, MAPE={row['MAPE']:.2f}%, R2={row['R2']:.4f}。"
        )
    total_text = "Total of all sites 当前仅输出符号诊断，默认不纳入建模。"
    if total_diagnostic is not None and not total_diagnostic.empty:
        total_text += " 诊断结果见 `results/total_sites_sign_diagnostic.csv`。"

    content = f"""# DKASC Alice Springs 光伏气象特征分析

## 项目背景

本项目服务于 Capstone 项目中的“基于天气大数据的新能源 / 光伏发电量预测”模块。分析对象为 DKASC Alice Springs 光伏与气象数据，核心问题是：哪些气象变量对光伏功率预测影响最大？

## 数据集说明

- 主 PV 标签：`91-Site_DKA-M9_B-Phase.csv`，对应 `1A Trina, 10.5kW, mono-Si, Dual, 2009`，主预测字段为 `Active_Power`。
- 可选对照标签：`96-Site_DKA-MasterMeter1.csv`，对应 `0 263.0kW, Total of all sites`。由于 `Active_Power` 存在正负号方向问题，当前默认只做符号诊断。
- 天气特征：`101-Site_DKA-WeatherStation.csv`，包含温度、相对湿度、风速、风向、降雨、水平辐照度和倾斜面辐照度等字段。

当前分析是基于 DKASC Alice Springs 单站点 / 聚合站点数据进行的初步研究，结论主要适用于该站点和所选时间范围，不能直接泛化到所有地区的 PV 系统。

## 文件结构

```text
dkasc_pv_feature_analysis/
  README.md
  requirements.txt
  run_analysis.py
  data/processed/
  src/
  figures/
  results/
  notebooks/
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 运行方式

默认从相邻的 `Capstone/*.csv` 读取原始数据，不移动、不覆盖原始文件：

```bash
python run_analysis.py
```

如需显式指定路径：

```bash
python run_analysis.py --weather-csv ../Capstone/101-Site_DKA-WeatherStation.csv --pv-1a-csv ../Capstone/91-Site_DKA-M9_B-Phase.csv
```

Total of all sites 默认只做符号诊断。确认 `Active_Power` 方向后，可使用：

```bash
python run_analysis.py --include-total-model
```

## 数据预处理方法

1. 解析并统一 `timestamp`。
2. 按时间戳对齐 PV label 与 WeatherStation 数据。
3. 保留 5-minute 粒度，图表中部分时间序列使用 daily/hourly 聚合以提升可读性。
4. 对负辐照度、异常湿度、异常风向、负风速、负降雨等进行质量检查。
5. 对主系统 `Active_Power < 0` 视为异常并置为缺失；超过 10.5kW 的 1.2 倍也标记为异常。
6. 天气字段使用短间隔插值，目标变量不插值。
7. 建模使用 daylight subset，完整日周期图保留全天样本。

## 特征工程方法

- 时间特征：`hour`, `month`, `day_of_year`, `sin_hour`, `cos_hour`, `sin_day_of_year`, `cos_day_of_year`。
- 辐照度特征：`Global_Horizontal_Radiation`, `Diffuse_Horizontal_Radiation`, `Radiation_Global_Tilted`, `Radiation_Diffuse_Tilted`。
- 温度特征：`Weather_Temperature_Celsius`。
- 湿度特征：`Weather_Relative_Humidity`。
- 风场特征：`Wind_Speed`, `Wind_Direction`, `wind_direction_sin`, `wind_direction_cos`。
- 降雨特征：`Weather_Daily_Rainfall`, `rain_indicator`。
- 可选滞后/滚动特征：使用 `--include-lag-features` 开启，且只使用当前及过去数据，避免未来信息泄漏。

## 特征评估方法

- Pearson correlation。
- Spearman correlation。
- Mutual information。
- Random Forest feature importance。
- Permutation importance。
- Feature group ablation。

## 消融实验设计

- `baseline_time_only`
- `baseline_irradiance_only`
- `time_plus_irradiance`
- `time_irradiance_temperature`
- `time_irradiance_humidity`
- `time_irradiance_wind`
- `time_irradiance_rainfall`
- `full_weather`

如果某一类字段不存在，程序会跳过或减少该组特征，并在结果表中体现。

## 模型与评估

使用 chronological split，按时间顺序划分训练、验证和测试集，避免时间泄漏。第一版模型包括 Ridge Regression 与 Random Forest Regressor。评估指标包括 MAE、RMSE、MAPE 和 R2。MAPE 仅在 daylight 且真实值大于阈值的样本上有解释意义。

当前主模型摘要：{metric_text}

## 主要可视化

图表保存在 `figures/`：

- PV output time series。
- Weather variable time series。
- Irradiance / temperature / humidity / wind speed vs PV output scatter。
- Correlation heatmap。
- Feature importance。
- Permutation importance。
- Ablation result。
- Predicted vs actual PV output。
- Daily profile。

## 主要结果表

结果保存在 `results/`：

- `data_summary.csv`
- `missing_value_summary.csv`
- `feature_correlation_1a_trina.csv`
- `mutual_information_1a_trina.csv`
- `feature_importance_1a_trina.csv`
- `permutation_importance_1a_trina.csv`
- `ablation_results_1a_trina.csv`
- `model_metrics.csv`
- `total_sites_sign_diagnostic.csv`

## 当前主要发现

第一版结果显示，辐照度变量是 1A Trina 光伏功率预测中最重要的气象信息。Random Forest impurity importance 中，`Radiation_Global_Tilted` 的重要性最高，其次是 `Global_Horizontal_Radiation`；permutation importance 也给出了相同方向的结论。统计相关性中，daylight 样本下 `Radiation_Global_Tilted` 与 `Active_Power` 的 Pearson 相关系数约为 0.796，`Global_Horizontal_Radiation` 约为 0.786。

非辐照度变量中，`Weather_Relative_Humidity` 与输出呈负相关，`Wind_Speed` 与输出呈正相关，说明湿度、云雨条件和组件冷却效应可能对功率有间接影响。不过这些变量的重要性明显低于辐照度。消融实验中，加入 wind features 的组合在当前 test split 上优于只使用 time + irradiance 的组合，说明风场变量对该站点可能有补充解释力。

模型结果需要谨慎解释。当前采用 chronological split，最后 15% 时间段作为测试集；Random Forest 在 validation split 上表现尚可，但在 test split 上 R2 为负，说明后期数据可能存在分布漂移、系统状态变化、维护/传感器差异或季节覆盖不均等问题。因此，本项目更适合作为“特征影响分析和消融研究”的第一版，而不是最终高精度预测模型。

## Total of all sites 状态

{total_text}

## 局限性

- 该分析主要针对 Alice Springs 单地点和当前数据时间范围。
- 相关性不等于因果关系，尤其全天样本会受到昼夜周期强烈影响。
- Total of all sites 的功率符号方向需要进一步确认。
- 第一版模型以解释和消融为主，不追求最优预测精度。
- 未强制使用 XGBoost、LightGBM 或 SHAP，以降低环境依赖复杂度。

## 下一步

- 确认 Total of all sites 的 `Active_Power` 正负号含义。
- 对比 5-minute 与 hourly 建模结果。
- 加入更严格的按年份或季节分组验证。
- 尝试 LightGBM / XGBoost 和 SHAP。
- 将关键图表和结果整理进 Capstone Journal。

## GitHub 上传前 checklist

- 确认原始数据不被上传，或使用 `.gitignore` 排除大体积 CSV。
- 确认 README 中结果与 `figures/`、`results/` 一致。
- 确认代码可通过命令行参数复现。
- 检查 `data/processed/` 是否包含可公开的数据。
- 检查是否需要添加数据来源和引用说明。
- 上传前先人工检查图表和结果表。

## 建议 commit message

```text
Add DKASC PV weather feature analysis workflow
```
"""
    readme_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    ensure_output_dirs(config)

    print("Loading weather data...")
    weather_df = read_csv_with_timestamp(config.weather_path)

    summaries = [summarize_dataframe(weather_df, "weather_station")]
    missing_tables = [missing_value_summary(weather_df, "weather_station")]

    label_df_for_summary = read_csv_with_timestamp(config.primary_label.path)
    summaries.append(summarize_dataframe(label_df_for_summary, "1a_trina_raw"))
    missing_tables.append(missing_value_summary(label_df_for_summary, "1a_trina_raw"))
    del label_df_for_summary

    if config.total_label and config.total_label.path.exists():
        total_raw = read_csv_with_timestamp(config.total_label.path)
        summaries.append(summarize_dataframe(total_raw, "total_sites_raw"))
        missing_tables.append(missing_value_summary(total_raw, "total_sites_raw"))
        del total_raw

    write_csv(pd.DataFrame(summaries), config.results_dir / "data_summary.csv")
    write_csv(pd.concat(missing_tables, ignore_index=True), config.results_dir / "missing_value_summary.csv")

    metrics, _ = run_label_analysis(config.primary_label, weather_df, config, args.include_lag_features)
    all_metrics = [metrics]

    total_diagnostic = None
    if config.total_label and config.total_label.path.exists():
        total_diagnostic = total_sign_diagnostic(config.total_label.path, weather_df, config)
        if config.total_label.enabled:
            total_metrics, _ = run_label_analysis(config.total_label, weather_df, config, args.include_lag_features)
            all_metrics.append(total_metrics)

    model_metrics = pd.concat(all_metrics, ignore_index=True)
    write_csv(model_metrics, config.results_dir / "model_metrics.csv")
    write_readme(config, model_metrics, total_diagnostic)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
