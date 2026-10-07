# NTU–Deye Capstone：光伏预测与充电温度实验

仓库按两条实验主线整理，更新于 **2026-10-07**。

| 主线 | 代码、数据与结果 | 集中报告 |
|---|---|---|
| 光伏预测实验 | [PV_Prediction](PV_Prediction/README.md) | [光伏报告索引](PV_Prediction/reports/README.md) |
| 充电温度与功率工况实验 | [Thermal_Prediction](Thermal_Prediction/README.md) | [温度实验报告索引](Thermal_Prediction/reports/README.md) |

## 目录

```text
PV_Prediction/                 # 光伏预测实验
  reports/                     # 历次实验、数据适用性、研究与项目计划
  audits/                      # 数据核查及目录迁移记录
  src/                         # 数据审计入口
  data/                        # PVOD 与 ERA5 原始输入
  PVOD_Forecast/                # 主预测：脚本、旧Notebook、模型和各轮结果
  DKASC_Analyze/                # DKASC 补充分析
  ERA5_Analyze/                 # ERA5 初步天气一致性分析
Thermal_Prediction/             # 充电温度与功率工况实验
  reports/                     # 实验计划、数据准备与最终实验报告
  src/                         # 下载、清洗、训练、推理和复核
  configs/                     # 划分与候选协议
  data/                        # 公开数据、来源字典及处理结果
  models/thermal_v1/            # RC参数、神经网络与归一化参数
  results/                     # 准备证据、指标、预测、图表和固定Demo
```

## 当前结果与边界

光伏 station05 已完成超前预测、天气消融、时序模型、天气对齐诊断及突变分类路由。最新路由在后两个历史区间的总体RMSE下降约0.47%/0.57%，收益有限；原地表HGB保留为稳健基线。ERA5发布延迟未模拟，相关实验是历史回顾研究。

温度实验使用公开真实车辆快充数据，目标为 **BMS电池包最高温度**。7次后续运行、753个测试窗口，未来60秒RMSE：温度保持 **1.416°C**、有效RC **1.209°C**、RC＋MLP **0.708°C**。混合模型只在3/7测试批次优于RC，历史平稳组不如温度保持。它不构成Deye充电模块实测、跨车辆验证或认证安全结论。

## 复现入口

光伏环境及各轮说明见 [PV_Prediction/README.md](PV_Prediction/README.md)。从仓库根目录复核最新保存结果：

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python PV_Prediction/PVOD_Forecast/experiments/validate_improvements_v5.py
```

温度环境及下载说明见 [Thermal_Prediction/README.md](Thermal_Prediction/README.md)。从仓库根目录运行：

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-thermal/bin/python Thermal_Prediction/src/run_thermal_v1.py
.venv-thermal/bin/python Thermal_Prediction/src/validate_thermal_v1.py
```

两个实验使用不同的依赖环境。复核不重新训练；训练会更新对应版本结果，应先保留需引用的历史产物。

## 数据与归档

外部原始数据的许可和引用要求继续适用。大型下载ZIP、开发环境和凭据不上传；温度原始ZIP可由下载入口重新取得，发布者校验及SHA256保留。连接器候选的下载成员留在本地，来源清单与不适用审计上传。模型、可复核处理数据、指标、报告和演示材料随实验保存。

旧Notebook和早期报告中的绝对路径、历史状态属于归档证据；当前复现以两个主线README和脚本为准。目录迁移表见 [迁移记录](PV_Prediction/audits/repository_reorganization_2026-10-07.json)。课程Journal原始PDF未随仓库公开。
