# 光伏预测实验

本目录集中光伏相关的数据、分析、可运行实验与保存结果。报告统一在 [reports](reports/README.md)，数据审计记录在 `audits/`。

## 模块与结果

| 模块 | 内容 | 入口 |
|---|---|---|
| PVOD_Forecast | station05 v1–v5：15/60分钟预测、LSTM/GRU、天气消融、场景诊断及加权/路由 | [最新运行说明](PVOD_Forecast/experiments/README_improvements_v5.md) |
| DKASC_Analyze | 光伏与本地天气探索、基线、特征及符号诊断 | [模块说明](DKASC_Analyze/README.md) |
| ERA5_Analyze | Alice Springs 2024年1月天气与DKASC小时配对 | [模块说明](ERA5_Analyze/README.md) |
| data | PVOD站点数据及ERA5原始输入 | [数据适用性](reports/PVOD数据适用性与剩余数据缺口_2026-10-05.md) |

主预测结果与保存模型位于 `PVOD_Forecast/results/`，各轮独立目录。旧Notebook与特征分析产物仍在各模块中，报告已抽出到 `reports/`。

## 主实验环境与复现

历史主实验使用 Python3.9、NumPy1.26.4、scikit-learn1.6.1、torch2.6.0。依赖见 `PVOD_Forecast/experiments/requirements_sequence.txt`。从仓库根目录：

```sh
python3.9 -m venv .venv-pv
.venv-pv/bin/python -m pip install -r PV_Prediction/PVOD_Forecast/experiments/requirements_sequence.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-pv/bin/python PV_Prediction/PVOD_Forecast/experiments/validate_improvements_v5.py
```

重跑最新实验：

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-pv/bin/python PV_Prediction/PVOD_Forecast/experiments/run_improvements_v5.py
```

station05原始表在 `data/PVODdatasets_v1.0/`，ERA5在 `data/ERA5_data/`。输入保持UTC/MW口径，排除2019-05-31，小时天气只向后保持。源码从自身位置定位本目录，执行不依赖当前工作目录。

## 边界

这是历史探索性研究；气象再分析可用延迟未模拟，不能承诺实时部署或外部泛化。各轮时段和方法有差别，不应跨任务直接比较指标。完整正负结果见集中报告。迁移时保留历史哈希和值，路径变化由 `audits/repository_reorganization_2026-10-07.json` 记录。
