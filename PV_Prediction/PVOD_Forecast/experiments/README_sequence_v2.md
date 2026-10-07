# station05 时序网络与滚动验证 v2

本轮在首轮数据契约上增加 LSTM / GRU、3 个时间前推评估区间与 3 个随机种子。全部结果独立保存于 `PV_Prediction/PVOD_Forecast/results/station05_sequence_v2/`。

## 运行

先按首轮 README 生成 `station05_v1/station05_aligned_15min.csv`。脚本会检查原始输入文件 SHA256 与首轮 manifest 完全一致，复用对齐表，避免重跑首轮模型。

```bash
pip install -r PV_Prediction/PVOD_Forecast/experiments/requirements_sequence.txt
python PV_Prediction/PVOD_Forecast/experiments/run_sequence_v2.py
python PV_Prediction/PVOD_Forecast/experiments/validate_sequence_v2.py
python PV_Prediction/PVOD_Forecast/experiments/summarize_sequence_v2.py
```

本次临时环境：

```bash
PYTHONPATH=/tmp/era5_read_deps MPLCONFIGDIR=/tmp/era5_mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 PV_Prediction/PVOD_Forecast/experiments/run_sequence_v2.py
```

所有模型在 CPU 跑；PyTorch 2.6.0、NumPy 1.26.4，两个计算线程，随机种子 17、42、2026。启用 deterministic algorithms，但跨硬件/库版本仍可能有数值差异。同目录再次运行会覆盖 v2；首轮目录保持原样。

## 模型和输入

- 16 个观测的序列，顺序从最旧到起点，包含功率、LMD、ERA5 地表与气压层；不使用 NWP。
- 单层 LSTM / GRU，hidden=32；最后时间步输出与 4 个未来目标时间周期特征拼接，接 Dense32/ReLU/Dense1。
- 预测目标是对持续性预测的修正：`未来功率 = 起点功率 + 网络预测残差`，最后非负截断。不是直接重跑原 Keras 双层 LSTM。
- Adam：lr=0.001，weight_decay=0.0001，batch=256，梯度裁剪=1，最多 35 epoch；验证 RMSE 连续 7 epoch 无改善则停止并恢复最佳 epoch。
- 输入、时间和残差 scaler 只拟合训练数据。序列 scaler 将训练窗口展开，重叠历史观测按出现次数加权；无验证/评价数据参与拟合。
- 固定按时间批次，批次间不传递 hidden state。测试及验证采用推理模式。
- 按 PyTorch 官方 [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) / [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) 接口，`batch_first=True`，张量形状 `[样本, 16, 特征]`。

## 滚动区间与可用性

| Fold | 训练标签截止前 | 验证起点 | 验证标签截止前 / 评价起点 | 评价标签截止前 |
|---|---|---|---|---|
| roll_1 | 2019-04-15 | 2019-04-15 | 2019-04-25 | 2019-05-05 |
| roll_2 | 2019-05-01 | 2019-05-01 | 2019-05-11 | 2019-05-21 |
| roll_3 | 2019-05-15 | 2019-05-15 | 2019-05-30 | 2019-06-14 |

全部 UTC。训练使用 target < validation_start；验证使用 origin >= validation_start 且 target < evaluation_start；评价使用 origin >= evaluation_start 且 target < evaluation_end。这样训练标签严格早于首个验证起点，验证标签严格早于首个评价起点。边界样本因此与 v1 不完全一致，不能直接拿 v1 汇总指标对比；本轮在共同样本重跑树模型和持续性。

后面的 fold 可以训练此前 fold 已评价过的历史数据，符合按时间扩展训练，而不是三个独立验证集。每次训练仅使用当时已过去的标签。ERA5 仍为再分析 valid-time 的回顾性假设，未模拟其发布延迟；站点测量也假设零延迟。

所有模型使用相同的当前 fold / horizon 评价样本。树模型沿用首轮超参数，分别比较 LMD、LMD+地表、LMD+地表+气压层。序列网络使用 16 步天气历史，树模型只用起点天气和功率历史，因此比较同时涉及算法和表示变化，不能归因于架构单一因素。

UTC 5 月 31 日排除规则和窗口连续性检查保留，任何序列或起点到目标跨过缺口则删除。白天只根据目标实测辐照 >20 W/m² 在评价时分层。

## 结果边界

本轮时间段来自此前已经查看过的数据；roll_3 与 v1 测试段重叠。因此是探索性的滚动复核，不是新的独立 holdout。参数和种子列表在训练前固定，不围绕评价指标调整。

seed_summary 的误差棒为同一区间三次神经网络训练 RMSE 的样本标准差，不是置信区间，也不代表季节泛化不确定性。树模型和持续性只运行一次（其设置确定），不制造多种子重复记录。

输出包含：每个 fold/horizon 的切分、逐点评价预测、模型权重与 scaler、训练历史、最佳 epoch、全天/白天指标、每日期误差、跨种子汇总、配置、源文件/脚本哈希及图。所有结果使用 MW；容量归一化指标仍沿用 35 MW 的已注明假设。
