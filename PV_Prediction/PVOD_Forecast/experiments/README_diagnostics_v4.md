# ERA5/LMD 一致性与场景诊断 v4

复用 v1 对齐数据、v2/v3 预测和 v3 切分，不训练或修改模型。依赖沿用 requirements_sequence.txt。

```bash
python PV_Prediction/PVOD_Forecast/experiments/run_diagnostics_v4.py
```

本次使用临时依赖目录：

```bash
PYTHONPATH=/tmp/era5_read_deps MPLCONFIGDIR=/tmp/era5_mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 PV_Prediction/PVOD_Forecast/experiments/run_diagnostics_v4.py
```

结果保存到 `PV_Prediction/PVOD_Forecast/results/station05_diagnostics_v4/`。再次执行只覆盖该诊断版本。脚本保存输入哈希、方法、天气配对、场景阈值、指标、逐点标签、图及检查结果。

天气：同 UTC 原生小时 valid time 比较温度/气压/风速；ERA5 前一小时辐射与 LMD 五点梯形积分近似比较，另报右端四点平均敏感性。LMD 积分区间语义、站点/网格高度、站点风速传感器高度仍未确认，不是仪器标定。

场景：训练白天辐照三分位、训练白天起点至目标净功率变化的 p90。评价的未来真实值只用于事后归类，不能拿此标签路由模型或当作预测输入。day_stable 指未超过净变化阈值，并非区间内完全平稳。阈值因 fold/horizon 不同，因此跨区间同名场景不代表相同绝对边界。

验证：逐行重算场景 MAE/RMSE、分组计数、预测有限性和 v2/v3 完整 GRU 相等。所有评价区间已经查看过，本轮属于探索诊断。
