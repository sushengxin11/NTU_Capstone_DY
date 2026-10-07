# 同一 GRU 网络的天气消融 v3

使用 v2 同一 GRU 残差架构、35 epoch 上限、验证 early stopping、三组时间区间、种子 17/42/2026 和 15/60 分钟任务。每组都在共同样本训练/评价，完整 ERA5 组用于复现 v2 GRU。

三种输入：

- `local`：历史功率 + LMD（8 个活跃序列字段）。
- `local_surface`：再加 ERA5 地表（16 个活跃字段）。
- `local_surface_pressure`：再加气压层（28 个活跃字段）。

为固定网络形状、参数数量及同种子初始化，三组都保留 28 个输入槽，被排除字段在 scaler 拟合前置为零。对应 scaler 的 mean=0、scale=1，网络实际收到零；其权重没有数据梯度。这是固定容量的输入遮蔽消融，不是对不同宽度网络的比较。全组参数为 7,169；GRU hidden=32，输出接 4 个目标时间周期特征和 Dense32/ReLU/Dense1。

同一种子的各组初始权重相同，批次顺序、优化器与验证选 epoch 规则相同；最佳训练 epoch 可以不同。各组单独训练，不是给全特征训练好的模型在推理时临时删变量。仍然是假设零测量延迟的 ERA5 回顾性实验，不是实时预报验证。

## 复现

使用 `requirements_sequence.txt`，先生成首轮对齐表。v3 复用 v2 的训练函数，并将本次进程输出目录指定为 v3，不修改 v2 文件。

```bash
python PV_Prediction/PVOD_Forecast/experiments/run_weather_ablation_v3.py
python PV_Prediction/PVOD_Forecast/experiments/validate_weather_ablation_v3.py
```

本次临时依赖命令前缀：

```bash
PYTHONPATH=/tmp/era5_read_deps MPLCONFIGDIR=/tmp/era5_mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
```

结果目录：`PV_Prediction/PVOD_Forecast/results/station05_weather_ablation_v3/`。含 54 个模型与 scaler、训练曲线、切分、逐点预测、全天/白天/每日指标、同种子配对差异、配置、哈希和图。误差棒是三种子样本标准差，不是置信区间。负的配对 RMSE 差值表示新特征改善。

继承 v2 的标签可用性边界、训练集 scaler 和 5 月 31 日缺口规则。所有时间区间已在之前实验查看过，因此本轮是固定协议的探索性消融，不是新独立 holdout；结果只支持该模型/训练预算和单站时间段内的观察。
