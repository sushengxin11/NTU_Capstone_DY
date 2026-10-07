# station05 光伏预测实验 v1

使用 station05 的原始 PVOD 和 `PV_Prediction/data/ERA5_data/`，运行 15 分钟与 1 小时超前功率预测。原始输入只读，新产物存于 `../results/station05_v1/`。

## 运行

Python 3.9 已实际运行；安装到独立环境后：

```bash
python3 -m venv .venv-pv
source .venv-pv/bin/activate
pip install -r PV_Prediction/PVOD_Forecast/experiments/requirements.txt
python PV_Prediction/PVOD_Forecast/experiments/run_station05.py
```

本次临时环境的运行命令（依赖在 `/tmp`，不是持久交付环境）：

```bash
PYTHONPATH=/tmp/era5_read_deps MPLCONFIGDIR=/tmp/era5_mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 PV_Prediction/PVOD_Forecast/experiments/run_station05.py
```

同一输出目录再次运行会覆盖该轮产物；新实验应修改 `OUT` 创建不同版本。

## 冻结方法

- 所有时间带 UTC 时区，时间周期特征用目标的 Asia/Shanghai 时间。功率单位 MW。
- 过去 16 个观测（起点及之前 15 个 15 分钟点）构成 4 小时窗口；预测目标为起点后 15 或 60 分钟，不是同期拟合。
- 输入仅取起点或之前的功率和 LMD；测量零延迟是本轮假设。PVOD NWP 缺 issue time，不使用。
- ERA5 空间双线性插值至 38.2355N、114.1236E；时间只取起点之前最近一个小时，有效期不超过 59 分钟，不使用未来小时插值。
- ERA5 气温 K 转 °C、地表气压 Pa 转 hPa，小时辐射 J/m² 除以 3600 得 W/m²。辐射为截至 valid time 的前一个小时平均，不能当作精确瞬时辐照。依据 [ECMWF ERA5 说明](https://confluence.ecmwf.int/pages/viewpage.action?pageId=414588701)。
- ERA5 是事后再分析；其真实发布延迟未模拟。因此天气模型是历史回顾性特征增益实验，不能声称实时可部署。即使只用 valid time <= 起点，也不等于当时已发布。
- 排除 UTC 2019-05-31。原始时间网格保留，排除行设空，窗口和目标之间有断点时删除该样本；不会跨天缺口 shift 或补值。
- 以目标时间划分：训练 `<2019-05-15 00:00 UTC`；验证 `[2019-05-15, 2019-05-30)`；测试 `>=2019-05-30`。历史上下文可来自此前集合，这符合滚动预测设定。
- 所有模型使用同一 horizon 的共同样本。持续性预测用起点功率，前日基线用精确目标时间减 24 小时的功率；该点缺失时回退到起点功率，标为 previous_day_fallback，保证比较样本完全一致。
- HGB 消融：功率历史+时间 → 加 LMD → 加 ERA5 地表 → 加气压层。固定超参数，不按测试表现选择特征。
- Ridge 和两层 MLP 使用全特征；scaler 只拟合训练集。MLP 按时间顺序训练、以验证 RMSE 保存最优 epoch，连续 15 个 epoch 无改善停止，最多 100 个 epoch。没有随机 validation split。
- 神经网络基线为 MLP，不是旧版 LSTM；本轮先验证端到端流程，后续另建 LSTM/GRU 实验版本。
- 全天及白天评价：白天按目标实测总辐照 >20 W/m²，仅用于评价分层，不进入预测输入。MAE、RMSE 为 MW，容量归一化 RMSE 的分母假设为 35 MW，源 metadata=35000，容量单位仍需数据字典确认。
- 输出预测非负截断，不按容量封顶；保留原始功率峰值。单站、约 101 天、单随机种子，不足以证明季节泛化或稳定显著增益。

## 来源和结果

UTC 与 MW 依据作者工具：[时间读取](https://raw.githubusercontent.com/yaotc/PVODataset/main/src/pvodataset.py)、[功率示例](https://raw.githubusercontent.com/yaotc/PVODataset/main/src/Demo_Kpv.py)。本地基础审计见 `PV_Prediction/reports/PVOD数据适用性与剩余数据缺口_2026-10-05.md`。

结果目录含源文件 SHA256、运行版本、配置、时空对齐表、每个 horizon 的 split、逐点验证/测试预测、指标、MLP 训练历史、模型/scaler，以及测试图。最终测试集在本轮结束后已经查看；后续模型选择应继续使用验证集，若围绕已看过的测试表现迭代，应标记为探索并增加独立时间段或站点验证。
