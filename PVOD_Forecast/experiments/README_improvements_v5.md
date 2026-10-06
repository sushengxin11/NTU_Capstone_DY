# station05 v5：加权训练与条件触发路由

本轮冻结1小时任务、地表输入、原时间切分和种子；只改变事件样本权重，再按验证专家优势决定是否进行分类路由。中文结果见 [实验报告](../../docs/station05突变加权与分类路由实验结果_2026-10-05.md)。

## 环境与运行

在仓库根目录执行；Python3.9，本轮使用torch2.6.0、NumPy1.26.4、scikit-learn1.6.1。依赖见 `requirements_sequence.txt`。正式复现可在本地虚拟环境安装该文件，随后执行：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 PVOD_Forecast/experiments/run_improvements_v5.py
```

当前会话的依赖在 `/tmp/era5_read_deps`，临时目录可能被系统清除。当前环境对应命令：

```bash
PYTHONPATH=/tmp/era5_read_deps MPLCONFIGDIR=/tmp/era5_mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 PVOD_Forecast/experiments/run_improvements_v5.py
```

单入口按顺序训练加权模型、条件触发路由、验证、生成图表和汇总。复现会更新两个v5结果目录，不更新v1–v4。要求原始数据、v1对齐表、v2/v3参考预测及v4训练阈值存在且哈希匹配；数据改变时应建立新实验版本，不应绕过检查。

分阶段入口为 `run_weighted_loss_v5.py`、`run_routing_v5.py`、`validate_improvements_v5.py`、`summarize_improvements_v5.py`。只检查已有产物时运行后两项即可。

## 协议

- α=1/2/4；训练白天终点净变化p90定义事件。所有普通点权重1，事件点权重α；不重采样。
- GRU按原未加权验证RMSE早停；α=1使用原MSE计算，避免浮点算式改变基线。
- 全局选模最小化验证突变RMSE，约束全天和白天平稳RMSE不超过α=1的1.02倍；GRU用三种子平均指标。
- 候选专家只按验证突变误差选，必须优于未加权地表HGB才开启路由。实际本轮roll_1跳过，roll_2/3使用HGB α=4专家。
- 分类拟合/校准为外层训练期按时间80/20拆分；拟合目标必须早于首次校准起点。LR/HGB分别平衡训练类别，用校准期的无加权LR做sigmoid校准。
- 硬阈值0.10–0.90、间隔0.05；软路由使用校准概率。验证集端到端回归选择仍执行2%约束；没有优势则退回基础模型。
- 各分类器最好的硬路由作为对照保存。若该分类器所有硬阈值均违反约束，可保存其突变误差最低阈值用于诊断，但不能成为正式选定方案。最终选择另在全部满足约束的候选中完成。
- Oracle未来标签路由仅作诊断。真实事件标签不得进入分类特征或实际推理。

## 保存模型推理

`predict_routing_v5.py`读取 `origin_utc` 为索引的CSV及保存的起点特征列，输出基础/选定功率；路由区间另输出专家功率和突变概率。字段清单见加权结果的 `config.json` 中 `tree_features`。`target_*`是预先可计算的时间周期特征，不是目标实测天气或功率。功率单位MW。

```bash
PYTHONPATH=/tmp/era5_read_deps MPLCONFIGDIR=/tmp/era5_mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 PVOD_Forecast/experiments/predict_routing_v5.py --fold roll_3 --input PVOD_Forecast/results/station05_weighted_loss_v5/roll_3_demo_origin_features.csv --output /tmp/station05_v5_demo_predictions.csv
```

该接口支持本轮实际选择的HGB专家；roll_1输出回退基线。演示输入已经保存且不含未来实测目标。它不提供实时采集、天气发布延迟处理或未经验证的部署保障。

## 结果索引

`station05_weighted_loss_v5`：config/manifest、36个模型及GRU scaler、metrics/seed_summary/paired_deltas、training_history、training_thresholds、split表、每日误差、验证/评价预测、选择和路由准入、基线复现与验证记录、加权图、推理示例。

`station05_routing_v5`：config/manifest、4个分类器及校准器、分类和校准指标、分类切分审计、全部验证候选、验证/评价路由预测、全指标及每日误差、选择、正式方案对比、误报漏报代价、路由图及验证记录。

结果是已查看历史区间的探索性研究；ERA5 valid time的可用性假设未模拟真实发布延迟。保留负结果，选模不读取评价指标。
