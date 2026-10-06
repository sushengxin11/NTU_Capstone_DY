# NTU Capstone：光伏预测与充电热预测

当前交付快照更新于 **2026-10-06**。仓库保存光伏数据对齐、天气消融、时序预测、突变加权与分类路由的代码、输入数据、保存模型和可核查结果。充电热预测已形成数据获取与实验计划，尚未完成温度模型实验。

## 已完成的实验

| 实验 | 内容 | 结果入口 |
|---|---|---|
| DKASC 分析 | 光伏与天气分析、预测基线和天气消融 | [说明](DKASC_Analyze/README.md) |
| ERA5 初步分析 | Alice Springs 2024年1月天气与DKASC对齐 | [说明](ERA5_Analyze/README.md) |
| station05 v1 | PVOD与ERA5按坐标/UTC对齐，15/60分钟预测及树模型天气消融 | [报告](docs/station05首轮光伏预测实验结果_2026-10-05.md) |
| station05 v2 | LSTM/GRU、三个滚动区间、三个种子 | [报告](docs/station05时序网络与滚动验证实验结果_2026-10-05.md) |
| station05 v3 | 同一GRU架构、初始化及样本的天气消融 | [报告](docs/station05同一GRU天气消融实验结果_2026-10-05.md) |
| station05 v4 | 天气一致性与快速升降等场景误差诊断 | [报告](docs/station05天气一致性与场景误差诊断_2026-10-05.md) |
| station05 v5 | 突变加权HGB/GRU、校准分类器、硬/软路由及保存模型推理 | [报告](docs/station05突变加权与分类路由实验结果_2026-10-05.md) |

最新实验的验证选定路由在后两个历史评价区间，全天RMSE降低约0.47%/0.57%，突变RMSE降低约0.52%/0.92%。直接加权没有通过预设的全天/白天平稳RMSE的2%验证约束，GRU加权未形成稳定优势；最新区间夜间误差略增加。完整正、负结果均保留，原地表HGB继续作为稳健基线。

## 目录与数据来源

- `PVODdatasets_v1.0/`：station00–station09、站点元数据及现有McClear文件。PVOD原作者资料见 [yaotc/PVODataset](https://github.com/yaotc/PVODataset)；这些是外部数据输入，不是本项目采集的设备数据。字段、时间和单位核查见[数据审计](docs/PVOD数据适用性与剩余数据缺口_2026-10-05.md)。
- `ERA5_data/`：7份2019年NetCDF，地表即时变量、辐射及500/850/1000 hPa变量。原始产品来自 [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)，读取与覆盖记录见[核查文件](docs/ERA5文件读取核查_2026-10-05.json)。数据来源的使用条款继续适用。
- `PVOD_Forecast/experiments/`：可运行脚本、依赖和各轮复现说明。旧Notebook及初步分析在相邻目录。
- `PVOD_Forecast/results/`：v1–v5分目录保存配置、文件哈希、模型/scaler、逐点预测、分组与每日指标、训练日志及图表。
- `docs/`：项目范围、数据核查、研究依据和中文实验报告；[项目计划](docs/项目现状与后续交付计划_2026-10-05.md)、[充电热预测计划](docs/充电热预测数据获取与实验计划_2026-10-05.md)。

PVOD功率及本地天气与ERA5按同一站点位置、UTC时刻关联；ERA5是补充输入，不替代电站功率真值。station05坐标38.2355°N、114.1236°E来自数据集的 `metadata.csv`。小时天气只向后保持至最近15分钟记录，不使用未来小时插值；UTC 2019-05-31按协议排除。

## 复现最新实验

本轮实际使用Python3.9、NumPy1.26.4、scikit-learn1.6.1和torch2.6.0。创建持久环境后运行：

```bash
python3 -m venv .venv-pv
source .venv-pv/bin/activate
python -m pip install -r PVOD_Forecast/experiments/requirements_sequence.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python PVOD_Forecast/experiments/run_improvements_v5.py
```

该入口顺序执行加权对照、条件触发路由、模型重载与指标验证、图表和推理示例生成。再次运行会更新v5产物，不覆盖v1–v4；运行前需保留原数据及早期实验参考产物。完整命令、协议和结果索引见[最新实验运行说明](PVOD_Forecast/experiments/README_improvements_v5.md)。

只复核现有结果：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python PVOD_Forecast/experiments/validate_improvements_v5.py
```

保存模型推理无需输入未来实测功率或目标实测辐照，示例见同一运行说明。早期报告保留了当时的本地路径和临时环境信息；从新机器运行应使用上述持久环境和仓库相对路径。

## 结论边界与下一步

目前是已查看历史时段的探索性研究。ERA5再分析的真实发布延迟未模拟，不能据此宣称实时可部署或独立泛化已得到验证。下一步冻结协议，核查station02/03天气覆盖并开展跨站点复现；充电热实验仍需明确对象与真实日志或替代数据路线。
