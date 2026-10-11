"""Consolidated feature-engineering entry point for the station05 workflow.

This script folds the three notebook-era steps into a single executable flow:
1. optional source inventory / download manifest
2. merge or standardize the station-level feature table
3. cleaning, feature selection, stability checks, and report export

The implementation is intentionally robust to the current repository layout.
If a raw source download layout is not available, the script can still run from
the existing merged station table under ``PVOD_Forecast/models/station05.csv``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"

DEFAULT_SOURCE_CSV = REPO_ROOT / "data" / "PVODdatasets_v1.0" / "station05.csv"
DEFAULT_MERGED_CSV = ARTIFACTS_DIR / "station05_merged_2019-04.csv"
DEFAULT_CLEANED_CSV = ARTIFACTS_DIR / "station05_cleaned_2019-04.csv"
DEFAULT_ENGINEERED_CSV = ARTIFACTS_DIR / "station05_engineered_2019-04.csv"
DEFAULT_MASTER_TABLE = ARTIFACTS_DIR / "station05_master_feature_table_2019-04.csv"
DEFAULT_FEATURE_METADATA = ARTIFACTS_DIR / "station05_feature_metadata_2019-04.csv"
DEFAULT_SUMMARY = ARTIFACTS_DIR / "station05_analysis_summary_2019-04.txt"
DEFAULT_REPORT = ARTIFACTS_DIR / "station05_feature_analysis_report_2019-04.md"
DEFAULT_BRIDGE_REPORT = REPORTS_DIR / "feature_engineering_pipeline_bridge.md"
DEFAULT_BRIDGE_MANIFEST = REPORTS_DIR / "feature_engineering_pipeline_bridge.json"
DEFAULT_PEARSON_MI_FIG = ARTIFACTS_DIR / "station05_pearson_vs_mi_scatter.png"


@dataclass(frozen=True)
class Paths:
    source_csv: Path
    merged_csv: Path
    cleaned_csv: Path
    engineered_csv: Path
    master_table: Path
    feature_metadata: Path
    summary_txt: Path
    report_md: Path
    bridge_report: Path
    bridge_manifest: Path
    pearson_mi_fig: Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidated feature-engineering pipeline for station05")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV, help="Existing merged station table or input CSV")
    parser.add_argument("--month", default="2019-04", help="Month to keep in YYYY-MM format")
    parser.add_argument("--skip-download", action="store_true", help="Skip the optional source inventory/download manifest stage")
    parser.add_argument("--build-only", action="store_true", help="Stop after merging and metadata export")
    parser.add_argument("--no-shap", action="store_true", help="Skip optional SHAP analysis even if shap is installed")
    return parser.parse_args()


def resolve_paths() -> Paths:
    return Paths(
        source_csv=DEFAULT_SOURCE_CSV,
        merged_csv=DEFAULT_MERGED_CSV,
        cleaned_csv=DEFAULT_CLEANED_CSV,
        engineered_csv=DEFAULT_ENGINEERED_CSV,
        master_table=DEFAULT_MASTER_TABLE,
        feature_metadata=DEFAULT_FEATURE_METADATA,
        summary_txt=DEFAULT_SUMMARY,
        report_md=DEFAULT_REPORT,
        bridge_report=DEFAULT_BRIDGE_REPORT,
        bridge_manifest=DEFAULT_BRIDGE_MANIFEST,
        pearson_mi_fig=DEFAULT_PEARSON_MI_FIG,
    )


def load_station_table(source_csv: Path) -> pd.DataFrame:
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {source_csv}")

    df = pd.read_csv(source_csv)
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], utc=False)
        df = df.sort_values("date_time").set_index("date_time")
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")].copy()

    return df


def month_slice(df: pd.DataFrame, month: str) -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-01 00:00:00")
    end = (start + pd.offsets.MonthEnd(1)) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return df.loc[start:end].copy()


def optional_source_inventory(paths: Paths) -> dict[str, object]:
    source_files: list[Path] = []
    for candidate in [paths.source_csv]:
        if candidate.exists():
            source_files.append(candidate)

    manifest = {
        "source_csv": safe_rel(paths.source_csv),
        "source_csv_sha256": sha256(paths.source_csv) if paths.source_csv.exists() else None,
        "available_sources": [safe_rel(p) for p in source_files],
        "available_source_count": len(source_files),
        "note": "Current repository ships the merged station table; raw download inputs are optional here.",
    }
    return manifest


def export_merged_table(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    ensure_parent(paths.merged_csv)
    merged = df.copy()
    merged.index.name = "date_time"
    merged.to_csv(paths.merged_csv)
    return merged


def basic_domain_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if "power" in cleaned.columns:
        cleaned.loc[cleaned["power"] < 0, "power"] = 0
        q1 = cleaned["power"].quantile(0.25)
        q3 = cleaned["power"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 2.0 * iqr
        upper = q3 + 2.0 * iqr
        cleaned["power"] = cleaned["power"].clip(lower=max(lower, 0), upper=upper)

    for col in cleaned.columns:
        if "humidity" in col.lower():
            cleaned[col] = cleaned[col].clip(0, 100)
        if "windspeed" in col.lower() or col.lower().endswith("_speed"):
            cleaned[col] = cleaned[col].clip(lower=0)
        if "winddirection" in col.lower() or "wind_direction" in col.lower():
            cleaned[col] = cleaned[col] % 360
        if any(token in col.lower() for token in ["irrad", "radiation", "gti", "ghi", "dhi", "dni", "ssrd", "fdir"]):
            cleaned[col] = cleaned[col].clip(lower=0)

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    index = pd.DatetimeIndex(engineered.index)
    hour_float = index.hour + index.minute / 60.0
    doy = index.dayofyear

    engineered["sin_hour"] = np.sin(2 * np.pi * hour_float / 24.0)
    engineered["cos_hour"] = np.cos(2 * np.pi * hour_float / 24.0)
    engineered["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    engineered["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    irradiance_candidates = [c for c in engineered.columns if any(token in c.lower() for token in ["totalirrad", "globalirrad", "ssrd", "ghi", "gti"])]
    if irradiance_candidates:
        irradiance = engineered[irradiance_candidates[0]].clip(lower=0)
        daily_p95 = irradiance.groupby(index.date).transform(lambda s: s.quantile(0.95) if s.notna().any() else np.nan)
        engineered["kcs_proxy"] = (irradiance / (daily_p95 + 1e-6)).clip(lower=0, upper=2.0)

    if "power" in engineered.columns:
        engineered["power_lag1"] = engineered["power"].shift(1)
        engineered["power_lag2"] = engineered["power"].shift(2)

    source_features = [c for c in engineered.columns if c != "power"]
    priority = []
    for candidate in ["lmd_totalirrad", "nwp_globalirrad", "nwp_directirrad", "lmd_diffuseirrad", "nwp_humidity", "nwp_temperature", "nwp_windspeed", "nwp_pressure"]:
        if candidate in source_features:
            priority.append(candidate)
    priority.extend([c for c in ["sin_hour", "cos_hour", "sin_doy", "cos_doy", "kcs_proxy", "power_lag1", "power_lag2"] if c in source_features])
    engineered = engineered.dropna().copy()
    engineered.attrs["priority_features"] = list(dict.fromkeys(priority))
    return engineered


def feature_family_counts(df: pd.DataFrame) -> dict[str, int]:
    families = {
        "1D": [c for c in df.columns if c.startswith("1D_") or c.startswith("lmd_")],
        "2D": [c for c in df.columns if c.startswith("2D_") or c.startswith("nwp_")],
        "3D": [c for c in df.columns if c.startswith("3D_")],
    }
    return {name: len(cols) for name, cols in families.items()}


def build_metadata(df: pd.DataFrame, source_csv: Path) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == "power":
            family = "target"
        elif col.startswith("1D_") or col.startswith("lmd_"):
            family = "1D"
        elif col.startswith("2D_") or col.startswith("nwp_"):
            family = "2D"
        elif col.startswith("3D_"):
            family = "3D"
        elif col in {"sin_hour", "cos_hour", "sin_doy", "cos_doy", "kcs_proxy", "power_lag1", "power_lag2"}:
            family = "engineered"
        else:
            family = "other"
        rows.append({
            "feature": col,
            "family": family,
            "dtype": str(df[col].dtype),
            "non_null_ratio": float(df[col].notna().mean()),
            "source": safe_rel(source_csv),
        })
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame, features: Iterable[str], target_col: str = "power") -> pd.DataFrame:
    rows = []
    for col in features:
        if col not in df.columns or col == target_col:
            continue
        pair = df[[col, target_col]].dropna()
        if len(pair) < 3:
            continue
        rows.append({
            "feature": col,
            "pearson": pair[col].corr(pair[target_col], method="pearson"),
            "spearman": pair[col].corr(pair[target_col], method="spearman"),
            "n": len(pair),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_pearson"] = out["pearson"].abs()
        out = out.sort_values("abs_pearson", ascending=False)
    return out


def mutual_information_table(df: pd.DataFrame, features: Iterable[str], target_col: str = "power") -> pd.DataFrame:
    cols = [c for c in features if c in df.columns and c != target_col]
    if not cols:
        return pd.DataFrame(columns=["feature", "mutual_information"])
    model_df = df[cols + [target_col]].dropna(subset=[target_col])
    if model_df.empty:
        return pd.DataFrame(columns=["feature", "mutual_information"])
    x = model_df[cols]
    y = model_df[target_col]
    x = x.ffill().bfill()
    scores = mutual_info_regression(x, y, random_state=42)
    return pd.DataFrame({"feature": cols, "mutual_information": scores}).sort_values("mutual_information", ascending=False)


def weighted_score_table(corr: pd.DataFrame, mi: pd.DataFrame, shap_like: pd.DataFrame | None = None) -> pd.DataFrame:
    tables = []
    if not corr.empty:
        tmp = corr[["feature", "abs_pearson"]].rename(columns={"abs_pearson": "pearson_score"})
        tables.append(tmp)
    if not mi.empty:
        tmp = mi[["feature", "mutual_information"]].rename(columns={"mutual_information": "mi_score"})
        tables.append(tmp)
    if shap_like is not None and not shap_like.empty:
        tables.append(shap_like)

    combined = None
    for table in tables:
        combined = table if combined is None else combined.merge(table, on="feature", how="outer")
    if combined is None:
        return pd.DataFrame(columns=["feature", "score"])

    for col in [c for c in combined.columns if c.endswith("_score")]:
        combined[col] = combined[col].fillna(0)
    score_cols = [c for c in combined.columns if c.endswith("_score")]
    if score_cols:
        combined["score"] = combined[score_cols].sum(axis=1)
    else:
        combined["score"] = 0.0
    return combined.sort_values("score", ascending=False).reset_index(drop=True)


def time_series_cv_summary(df: pd.DataFrame, feature_cols: list[str], target_col: str = "power") -> pd.DataFrame:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    y = df[target_col]
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            n_iter_no_change=10,
            validation_fraction=0.1,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        truth = y.iloc[test_idx]
        rows.append({
            "fold": fold,
            "R2": r2_score(truth, pred),
            "MAE": mean_absolute_error(truth, pred),
            "RMSE": mean_squared_error(truth, pred) ** 0.5,
        })
    return pd.DataFrame(rows)


def bootstrap_stability(df: pd.DataFrame, candidate_features: list[str], target_col: str = "power") -> pd.DataFrame:
    if not candidate_features:
        return pd.DataFrame(columns=["feature", "hit_count", "hit_rate"])
    X = df[candidate_features].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    y = df[target_col]
    rng = np.random.default_rng(42)
    hit_count = pd.Series(0, index=X.columns, dtype=int)
    n_bootstrap = min(20, max(8, len(X) // 500))
    top_k = min(20, len(candidate_features))
    for seed in range(n_bootstrap):
        sample_idx = rng.choice(len(X), size=max(int(len(X) * 0.65), top_k + 5), replace=False)
        rf = RandomForestRegressor(
            n_estimators=80,
            max_depth=10,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42 + seed,
            n_jobs=-1,
        )
        rf.fit(X.iloc[sample_idx], y.iloc[sample_idx])
        ranked = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        hit_count.loc[ranked.head(top_k).index] += 1
    out = pd.DataFrame({
        "feature": hit_count.index,
        "hit_count": hit_count.values,
        "hit_rate": hit_count.values / n_bootstrap,
    }).sort_values("hit_rate", ascending=False).reset_index(drop=True)
    return out


def maybe_shap_summary(df: pd.DataFrame, candidate_features: list[str], target_col: str = "power") -> pd.DataFrame | None:
    try:
        import shap  # type: ignore
    except Exception:
        return None

    if not candidate_features:
        return None

    cols = candidate_features[: min(25, len(candidate_features))]
    X = df[cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    y = df[target_col]
    model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(X, y)
    sample = X.sample(min(len(X), 600), random_state=42)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[0]
    mean_abs = np.abs(values).mean(axis=0)
    return pd.DataFrame({"feature": cols, "shap_score": mean_abs}).sort_values("shap_score", ascending=False)


def make_report(
    paths: Paths,
    source_manifest: dict[str, object],
    merged: pd.DataFrame,
    cleaned: pd.DataFrame,
    engineered: pd.DataFrame,
    metadata: pd.DataFrame,
    corr: pd.DataFrame,
    mi: pd.DataFrame,
    scoring: pd.DataFrame,
    cv_summary: pd.DataFrame,
    stability: pd.DataFrame,
    shap_summary: pd.DataFrame | None,
) -> str:
    top_features = scoring.head(10)[["feature", "score"]] if not scoring.empty else pd.DataFrame(columns=["feature", "score"])
    top_lines = "\n".join(f"- {row.feature}: {row.score:.4f}" for row in top_features.itertuples(index=False)) if not top_features.empty else "- 无"
    corr_top = "\n".join(f"- {row.feature}: {row.pearson:.4f}" for row in corr.head(8).itertuples(index=False)) if not corr.empty else "- 无"
    mi_top = "\n".join(f"- {row.feature}: {row.mutual_information:.4f}" for row in mi.head(8).itertuples(index=False)) if not mi.empty else "- 无"
    stable_top = "\n".join(f"- {row.feature}: {row.hit_rate:.2f}" for row in stability.head(8).itertuples(index=False)) if not stability.empty else "- 无"
    shap_top = "\n".join(f"- {row.feature}: {row.shap_score:.4f}" for row in shap_summary.head(8).itertuples(index=False)) if shap_summary is not None and not shap_summary.empty else "- 未启用 / 未安装 shap"
    cv_text = cv_summary.round(4).to_string(index=False) if not cv_summary.empty else "(无)"

    return f"""# station05 特征工程闭环报告

这份报告由 `run_feature_engineering.py` 生成，用来把 notebook 时代的三步流程收敛为一个可执行入口。

## 闭环角色

- `feature_engineering/` 负责把原始表整理成统一、可消费的特征表，并产出可追踪的特征分析结果。
- `models/` 负责做输入模态可行性判断，例如 3D vs 2D 是否值得作为新输入分支。
- `experiments/` 负责在固定切分和指标下验证模型收益，属于下游验证层。
- `reports/` 负责收口结论、边界和下一步建议。

## 本次输入概况

- 原始输入: `{source_manifest['source_csv']}`
- 原始输入 SHA256: `{source_manifest['source_csv_sha256']}`
- 计算后样本数: {len(merged)}
- 清洗后样本数: {len(cleaned)}
- 工程化后样本数: {len(engineered)}
- 特征家族计数: {json.dumps(feature_family_counts(engineered), ensure_ascii=False)}

## 主要输出

- 合并表: `{paths.merged_csv.relative_to(REPO_ROOT)}`
- 清洗表: `{paths.cleaned_csv.relative_to(REPO_ROOT)}`
- 工程化表: `{paths.engineered_csv.relative_to(REPO_ROOT)}`
- 主特征表: `{paths.master_table.relative_to(REPO_ROOT)}`
- 特征元数据: `{paths.feature_metadata.relative_to(REPO_ROOT)}`
- 文本摘要: `{paths.summary_txt.relative_to(REPO_ROOT)}`
- 报告: `{paths.report_md.relative_to(REPO_ROOT)}`

## 关键结论

### 综合特征排序

{top_lines}

### Pearson 相关性 Top

{corr_top}

### Mutual Information Top

{mi_top}

### Bootstrap 稳定性 Top

{stable_top}

### SHAP 结果 Top

{shap_top}

## 时序验证摘要

{cv_text}

## 逻辑解释

1. 先用这个入口把数据整理成统一的 station05 特征表。
2. 再把 `models/3D_vs_2D_Analysis.md` 的结论作为输入模态选择的软门控，而不是硬编码的停止条件。
3. 最后由 `experiments/` 用统一切分和指标验证：
   - v1 以表格特征为主，属于 1D/2D 风格的 tabular 基线。
   - v2/v3/v5 以 16 步历史序列输入为主，张量形状是 3D：`[samples, time_steps, features]`。

## 解释边界

- 这个入口的目标是把 notebook 变成可执行、可复用、可报告的脚本，不是保留 notebook 的所有交互式展示。
- SHAP 是可选的；如果环境里没有安装，脚本会自动跳过，不影响主流程。
- 模态分析是 experiments 的前置建议，不是强制阻断条件。
"""


def main() -> None:
    args = parse_args()
    paths = resolve_paths()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        source_manifest = optional_source_inventory(paths)
    else:
        source_manifest = {
            "source_csv": safe_rel(paths.source_csv),
            "source_csv_sha256": sha256(paths.source_csv) if paths.source_csv.exists() else None,
            "available_sources": [safe_rel(paths.source_csv)] if paths.source_csv.exists() else [],
            "available_source_count": 1 if paths.source_csv.exists() else 0,
            "note": "download/inventory stage skipped by flag",
        }

    source_df = load_station_table(args.source_csv)
    source_df = month_slice(source_df, args.month)
    merged = export_merged_table(source_df, paths)

    cleaned = basic_domain_cleanup(merged)
    cleaned.index.name = "date_time"
    cleaned.to_csv(paths.cleaned_csv)

    engineered = engineer_features(cleaned)
    engineered.index.name = "date_time"
    engineered.to_csv(paths.engineered_csv)

    metadata = build_metadata(engineered, args.source_csv)
    metadata.to_csv(paths.feature_metadata, index=False)

    master_table = engineered.copy()
    master_table.to_csv(paths.master_table)

    feature_cols = [c for c in engineered.columns if c != "power"]
    corr = correlation_table(engineered, feature_cols)
    mi = mutual_information_table(engineered, feature_cols)
    shap_summary = None if args.no_shap else maybe_shap_summary(engineered, feature_cols)
    scoring = weighted_score_table(corr, mi, shap_summary)
    cv_summary = time_series_cv_summary(engineered, feature_cols)
    stability = bootstrap_stability(engineered, list(scoring.head(40)["feature"]) if not scoring.empty else feature_cols)

    # A lightweight visualization to keep parity with the notebook outputs.
    if not corr.empty and not mi.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        merged_scores = corr[["feature", "abs_pearson"]].merge(mi, on="feature", how="inner")
        ax.scatter(merged_scores["abs_pearson"], merged_scores["mutual_information"], s=30, alpha=0.8)
        for row in merged_scores.nlargest(12, "mutual_information").itertuples(index=False):
            ax.annotate(row.feature, (row.abs_pearson, row.mutual_information), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Pearson |r|")
        ax.set_ylabel("Mutual Information")
        ax.set_title("Pearson vs Mutual Information")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(paths.pearson_mi_fig, dpi=160)
        plt.close(fig)

    report_text = make_report(paths, source_manifest, merged, cleaned, engineered, metadata, corr, mi, scoring, cv_summary, stability, shap_summary)
    paths.report_md.write_text(report_text + "\n", encoding="utf-8")

    summary_lines = [
        f"源表: {args.source_csv}",
        f"月度切片: {args.month}",
        f"合并表: {paths.merged_csv}",
        f"清洗表: {paths.cleaned_csv}",
        f"工程化表: {paths.engineered_csv}",
        f"主特征表: {paths.master_table}",
        f"特征元数据: {paths.feature_metadata}",
        f"报告: {paths.report_md}",
        f"Pearson-MI 图: {paths.pearson_mi_fig}",
        f"特征家族计数: {json.dumps(feature_family_counts(engineered), ensure_ascii=False)}",
        f"CV 平均 RMSE: {cv_summary['RMSE'].mean():.4f}",
    ]
    paths.summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    bridge_manifest = {
        "source_manifest": source_manifest,
        "source_csv": safe_rel(args.source_csv),
        "source_csv_sha256": sha256(args.source_csv) if args.source_csv.exists() else None,
        "month": args.month,
        "outputs": {
            "merged_csv": safe_rel(paths.merged_csv),
            "cleaned_csv": safe_rel(paths.cleaned_csv),
            "engineered_csv": safe_rel(paths.engineered_csv),
            "master_table": safe_rel(paths.master_table),
            "feature_metadata": safe_rel(paths.feature_metadata),
            "summary_txt": safe_rel(paths.summary_txt),
            "report_md": safe_rel(paths.report_md),
            "pearson_mi_fig": safe_rel(paths.pearson_mi_fig),
        },
        "feature_families": feature_family_counts(engineered),
        "shap_enabled": not args.no_shap and shap_summary is not None,
        "soft_gate_message": "Feature engineering is an upstream recommendation gate, not a hard stop for experiments.",
        "experiments_input_modes": {
            "v1": "tabular history + current time features",
            "v2_v3_v5": "3D sequence tensor [samples, time_steps, features]",
        },
    }
    paths.bridge_manifest.write_text(json.dumps(bridge_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bridge_text = f"""# feature_engineering -> models -> experiments 闭环说明

这个文档由 `run_feature_engineering.py` 生成，解释 feature_engineering 在整个仓库中的位置。

## 角色划分

- `feature_engineering/`：把原始站点表与天气表整理成统一特征表，并输出可复用的统计和报告。
- `models/`：做模态可行性判断，例如 3D vs 2D 是否值得作为默认输入空间。
- `experiments/`：在统一切分和统一指标下验证模型收益，决定哪些模态真的值得保留。
- `reports/`：收束结论与下一步研究建议。

## 逻辑顺序

```mermaid
flowchart TD
    A[feature_engineering: merge / clean / engineer] --> B[models: modality feasibility]
    B --> C[experiments: train / validate / ablate]
    C --> D[reports: conclusions]
    B --> D
```

## 模态判断是软门控

- `models/3D_vs_2D_Analysis.md` 给出的结论，不是硬停止条件，而是 experiments 的默认输入策略建议。
- 如果 3D 数据只带来更粗的时间分辨率和更少的样本，那么它不应自动升级为 experiments 的默认主分支。
- experiments 仍然可以显式做 1D、2D、3D 或序列融合实验，但应在报告中说明这是有针对性的假设验证，而不是默认结论。

## experiments 里到底用了什么维度

- `run_station05.py` 主要是表格型输入，属于 1D / tabular 风格基线。
- `run_sequence_v2.py`、`run_weather_ablation_v3.py`、`run_weighted_loss_v5.py` 和 `run_routing_v5.py` 都使用 16 步历史序列，模型输入张量本质上是 3D：`[样本数, 时间步, 特征数]`。
- 因此，experiments 不是单纯的“一维/二维/三维数据选择”，而是“先定特征空间，再用序列模型验证”的组合流程。

## 当前建议

1. 先运行 `run_feature_engineering.py` 生成统一的特征表和报告。
2. 再让 `models/` 的模态分析作为 experiments 的默认输入建议。
3. 最后用 `experiments/` 的滚动验证和消融实验确认是否真的有增益。

## 产物

- 特征工程报告: `{paths.report_md.relative_to(REPO_ROOT)}`
- 桥接报告: `{paths.bridge_report.relative_to(REPO_ROOT)}`
- 桥接 manifest: `{paths.bridge_manifest.relative_to(REPO_ROOT)}`
"""
    paths.bridge_report.write_text(bridge_text + "\n", encoding="utf-8")

    print(f"Wrote {paths.merged_csv}")
    print(f"Wrote {paths.cleaned_csv}")
    print(f"Wrote {paths.engineered_csv}")
    print(f"Wrote {paths.report_md}")
    print(f"Wrote {paths.bridge_report}")


if __name__ == "__main__":
    main()