# 充电温度实验：公开数据与最小模型比较

2026-10-07 已完成公开数据准备、有效 RC 标定、小型 MLP 残差训练、分功率评价、保存模型推理和指标复核。详见[实验报告](reports/充电温度与功率工况实验结果_2026-10-07.md)与[准备报告](reports/充电热实验数据准备结果_2026-10-07.md)。

7次后续快充运行、753个测试窗口的60秒 RMSE：温度保持1.416 °C、RC 1.209 °C、RC＋MLP 0.708 °C。混合模型仅在3/7测试批次优于RC，历史平稳组不如温度保持。当前支持公开电池包原型比较，未验证其他车辆或模块温度。

主候选：[KU Leuven BEV energy dynamics V2](https://doi.org/10.48804/8KPDTW)，CC-BY-4.0，25次快充、2个车辆标识；目标为 BMS 电池包最高温度，不是 Deye 充电模块实测。原 ZIP 完整保留，发布者 MD5 校验通过，SHA256 与各快充成员哈希见 `results/preparation/summary.json`。

TU Dortmund 候选：[官方数据](https://zenodo.org/records/14065331)，CC-BY-4.0。只下载选择的 ZIP 成员，记录成员 CRC 与 SHA256；没有下载并校验完整 ZIP。含 CCS 温度记录的运行实际使用 CHAdeMO，故不用于 CCS 热模型。

## 运行

Python3.12 已验证。首次准备环境（仓库根目录）：

```sh
python3.12 -m venv .venv-thermal
.venv-thermal/bin/python -m pip install -r Thermal_Prediction/environment-lock.txt
```

下载或校验主候选，随后准备数据：

```sh
.venv-thermal/bin/python Thermal_Prediction/src/fetch_bev.py
.venv-thermal/bin/python Thermal_Prediction/src/prepare_public_data.py
```

可选：重新获取连接器候选以复核不适用原因：

```sh
.venv-thermal/bin/python Thermal_Prediction/src/fetch_tudortmund.py
.venv-thermal/bin/python Thermal_Prediction/src/fetch_tudortmund.py --raw-monitor
.venv-thermal/bin/python Thermal_Prediction/src/prepare_public_data.py
```

下载操作需要网络；校验和准备仅使用本地文件。准备会重新生成 processed、preparation 和候选配置，不修改原始文件。首次从空目录准备且没有 TU Dortmund 成员时，其审计为空，不表示该数据源被实际检查。

## 数据与规则

`bev_fast_candidate.csv` 保存抽取后的候选记录；`bev_fast_clean.csv` 仅保留有效时间及必要有限数值。没有填补温度标签；窗口不得跨连续段或批次。同车同 UTC 日期保持同一划分。`provisional_split.csv` 为14/4/7批次，`candidate_protocol.json` 保存仅从训练数据确定的功率阈值。

CSV 核心字段：session_id、device_id、timestamp_utc、time_s、temperature_c、ambient_c、dc_voltage_v、dc_current_a、dc_power_w、battery_voltage_v、battery_current_a、soc_percent、segment_id。功率为 W，温度为 °C。

`summary.json` 为机器可读摘要；`bev_fast_session_quality.csv`、`continuous_segments.csv`、`power_coverage.csv` 和 `tudortmund_target_audit.json` 为检查证据。`preparation_review.ipynb` 提供相同准备入口的复核方式，未保存虚构执行输出。

## 运行实验、复核与演示

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-thermal/bin/python Thermal_Prediction/src/run_thermal_v1.py
.venv-thermal/bin/python Thermal_Prediction/src/validate_thermal_v1.py
.venv-thermal/bin/python Thermal_Prediction/src/predict_thermal_v1.py --history Thermal_Prediction/results/thermal_v1/demo_history.csv --output Thermal_Prediction/results/thermal_v1/demo_cli_predictions.csv
.venv-thermal/bin/python Thermal_Prediction/src/plot_thermal_demo.py
```

实验入口更新 `results/thermal_v1/` 和 `models/thermal_v1/`；复核入口不训练模型。输入为61个连续1秒历史点，输出10至60秒的6个温度值。窗口每10秒生成一次，不跨缺口或批次。14/4/7运行划分在本轮冻结；原先文件名保留 provisional，实际使用的配置和输入哈希已存入 `results/thermal_v1/config.json`。

保存内容包括3种预测、总体/分功率/逐批次/分阶段指标、训练日志、模型与归一化参数、全部测试曲线、固定Demo、输入哈希和独立复算记录。训练归一化及RC标定仅使用训练集合，网络及轮次仅按验证集合选择。

固定Demo使用按标识排序的第一个测试批次，不按误差挑选。图中整次曲线是每个起点重置温度的滚动短时预测，不是一次初始化后预测完整充电过程。

## 适用边界

对象为BMS电池包最高温度，RC仅是有效热响应基线，不能分别辨识真实热阻/热容。电功率不等于热功率；冷却/预热状态未完整观察。三档功率是实测数据分组，不是三种受控实验协议。未来实测功率不作为输入；实际模型假设起点电流和环境保持。跨车辆泛化、Deye模块实测及认证安全仍未验证。混合网络没有强制物理约束，RC方程检查不等于网络在任意输入下都物理合理。
