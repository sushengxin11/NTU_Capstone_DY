"""Bridge the 3D vs 2D feature-space study into the experiment pipeline.

This script is intentionally lightweight: it standardizes the comparison
report, emits a machine-readable manifest, and makes the upstream data-space
decision available to downstream experiment stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
REPORTS_DIR = ROOT / "reports"
DEFAULT_SOURCE_MD = REPORTS_DIR / "3D_vs_2D_Analysis.md"
DEFAULT_STATION_CSV = REPO_ROOT / "data" / "PVODdatasets_v1.0" / "station05.csv"
DEFAULT_REPORT = REPORTS_DIR / "3D_vs_2D_experiment_bridge.md"
DEFAULT_MANIFEST = REPORTS_DIR / "3D_vs_2D_experiment_bridge.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_cell(value: str) -> str:
    return (
        value.replace("**", "")
        .replace("❌", "")
        .replace("✓", "")
        .replace("`", "")
        .strip()
    )


def parse_metric_table(text: str) -> dict[str, dict[str, str]]:
    metrics: dict[str, dict[str, str]] = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("| 指标 |"):
            in_table = True
            continue
        if in_table and not line.startswith("|"):
            break
        if not in_table:
            continue
        cells = [clean_cell(part) for part in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        if cells[0] in {"MAE", "RMSE", "R²"}:
            metrics[cells[0]] = {
                "2d": cells[1],
                "3d": cells[2],
                "delta": cells[3],
            }
    return metrics


def load_station_profile(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, parse_dates=["date_time"]).sort_values("date_time")
    numeric = frame.select_dtypes(include="number")
    correlations = (
        numeric.corr(numeric_only=True)["power"]
        .dropna()
        .drop(labels=["power"], errors="ignore")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    feature_groups = {
        "nwp": [c for c in frame.columns if c.startswith("nwp_")],
        "lmd": [c for c in frame.columns if c.startswith("lmd_")],
    }
    return {
        "rows": int(len(frame)),
        "start_utc": str(frame["date_time"].min()),
        "end_utc": str(frame["date_time"].max()),
        "feature_count": int(len(frame.columns) - 1),
        "feature_groups": {k: len(v) for k, v in feature_groups.items()},
        "power_min": float(frame["power"].min()),
        "power_max": float(frame["power"].max()),
        "power_mean": float(frame["power"].mean()),
        "power_top_correlations": [
            {"feature": name, "pearson": float(value)}
            for name, value in correlations.head(5).items()
        ],
    }


def build_report(metric_table: dict[str, dict[str, str]], profile: dict[str, object], source_md: Path) -> str:
    rows = []
    for metric in ["MAE", "RMSE", "R²"]:
        if metric in metric_table:
            rows.append(
                f"| {metric} | {metric_table[metric]['2d']} | {metric_table[metric]['3d']} | {metric_table[metric]['delta']} |"
            )

    top_corr = profile["power_top_correlations"]
    corr_lines = "\n".join(
        f"- {item['feature']}: {item['pearson']:.4f}" for item in top_corr  # type: ignore[index]
    )

    return f"""# 3D vs 2D 特征空间桥接报告

这个报告把 [models/3D_vs_2D_Analysis.md]({source_md.relative_to(ROOT)}) 里的探索性结论，标准化为一个可以被 experiments 消费的上游决策步骤。

## 结论先行

- 这里的 3D vs 2D 分析不是和 experiments 重复做同一件事，而是在做“输入模态选择”的上游判断。
- 如果 3D 数据的时间分辨率更粗、样本更少、且对短时光伏预测没有稳定增益，那么它不应成为默认主分支。
- 更合理的做法是：先用 models 目录完成模态可行性判断，再在 experiments 目录验证模型收益，最后统一输出 report。

## 证据摘要

### 原报告中的性能对比

| 指标 | 2D 模型 | 3D 模型 | 变化 |
|---|---:|---:|---:|
{chr(10).join(rows)}

### 当前站点输入概况

- 时间范围: {profile['start_utc']} -> {profile['end_utc']}
- 行数: {profile['rows']}
- 特征数: {profile['feature_count']}
- 特征分组: NWP={profile['feature_groups']['nwp']}，LMD={profile['feature_groups']['lmd']}
- power 统计: min={profile['power_min']:.4f}，max={profile['power_max']:.4f}，mean={profile['power_mean']:.4f}

### 与 power 相关性最高的特征

{corr_lines}

## 如何并入 experiments

```mermaid
flowchart TD
    A[models: 特征模态对标] --> B[experiments: 模型/窗口/消融实验]
    B --> C[experiments: 验证与稳健性检查]
    C --> D[reports: 汇总结论与建议]
    A --> D
```

推荐的闭环是：

1. 在 `models/` 里先判断“该不该引入更高维或更高频模态”。
2. 在 `experiments/` 里验证“引入后是否真的带来可重复提升”。
3. 在 `reports/` 里只保留结论、边界和下一步建议。

## 这一步的作用

- 它把原来偏 notebook 的探索，变成了一个可执行脚本。
- 它让“模态选择”成为 experiments 的前置门，而不是事后解释。
- 它为后续追加 3D 高分辨率数据、2D+3D 融合、或多站点外推留出标准接口。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a bridge report for the 3D vs 2D feature-space study.")
    parser.add_argument("--source-md", type=Path, default=DEFAULT_SOURCE_MD, help="Path to the source analysis markdown")
    parser.add_argument("--station-csv", type=Path, default=DEFAULT_STATION_CSV, help="Path to the station CSV used for profile summary")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Output report path")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Output machine-readable manifest path")
    args = parser.parse_args()

    if not args.source_md.exists():
        raise FileNotFoundError(f"Missing source analysis markdown: {args.source_md}")
    if not args.station_csv.exists():
        raise FileNotFoundError(f"Missing station CSV: {args.station_csv}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_table = parse_metric_table(args.source_md.read_text(encoding="utf-8"))
    profile = load_station_profile(args.station_csv)
    report = build_report(metric_table, profile, args.source_md)
    args.report.write_text(report + "\n", encoding="utf-8")

    manifest = {
        "source_md": str(args.source_md.relative_to(ROOT)),
        "station_csv": str(args.station_csv.relative_to(ROOT)),
        "source_md_sha256": sha256(args.source_md),
        "station_csv_sha256": sha256(args.station_csv),
        "metric_table": metric_table,
        "profile": profile,
        "report": str(args.report.relative_to(ROOT)),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()