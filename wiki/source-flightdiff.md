---
title: "FlightDiff: A Dual-Constraint Guided Two-Phase Diffusion Framework for Accurate Flight Prediction"
type: source-summary
tags:
  - diffusion-model
  - flight-trajectory
  - trajectory-prediction
  - variational-autoencoder
  - air-traffic-management
  - constraint-guided
created: 2026-06-18
last_updated: 2026-06-18
source_count: 1
confidence: medium
status: active
---

# FlightDiff 论文摘要

**FlightDiff**（Dual-Constraint Guided Two-Phase Diffusion）由中国海洋大学 Peilan He、Zewei Zhang、Yanwei Yu 等人发表于 GeoInformatica 2026。论文提出了首个基于扩散模型的飞行轨迹预测框架，通过双约束引导的两阶段去噪过程生成物理上连贯且操作合规的飞行轨迹[^src-flightdiff]。

## 核心贡献

1. **首个飞行轨迹扩散模型**：首次将扩散模型应用于飞行轨迹预测，同时实现高统计精度和严格的法规合规性。

2. **双约束两阶段去噪**：第一阶段同时施加航路点（局部约束）和目的地（全局约束），确保物理可行性；第二阶段仅保留目的地约束进行精细调整，在可行性和灵活性之间取得平衡。

3. **共享 VAE 模块**：用同一个 VAE 编码密集轨迹和稀疏航路点数据到统一潜在空间，保留空间细节，避免传统掩码方法的信息损失。

## 核心方法

### 问题形式化

给定观测轨迹 O_{t-k+1:t}（过去 k 个状态，含经纬度、高度、速度）和飞行计划 W（含航路点和目的地），目标是预测未来 n 个状态 P_{t+1:t+n} 和到达时间 T_dest[^src-flightdiff]。

### 共享 VAE

将轨迹特征图（含经纬度、高度、合速度、剩余到达时间）和航路点图编码到统一潜在空间。VAE 通过最大化 ELBO 训练，航路点图被视为稀疏轨迹，使用同一编码器处理，保持空间特征一致性。

### 轨迹-目的地编码器（TTE）

基于 FlightBERT++ 的自监督编码器，输入观测轨迹和目的地的差分序列，学习融合历史运动状态和全局目标的上下文向量 T_enc，用于条件化每一步反向扩散。

### 双约束两阶段扩散

- **第一阶段（双约束执行）**：UNet1 在条件化于航路点潜在编码 z 和轨迹-目的地上下文 T_enc 的情况下，从纯噪声 x_T 去噪到 x_{αT}，强制遵守航路点走廊和全局目的地。
- **第二阶段（全局约束精炼）**：UNet2 仅条件化于 T_enc，继续从 x_{αT} 去噪到 x_0，释放航路点约束以实现曲率和速度的精细调整。α 控制两阶段交接点，实验取 α=0.7。

## 实验结果

### 飞行轨迹预测（FTP）

在青岛 2024 年 5–7 月航班数据集（训练集约 60 万条，测试集约 7.5 万条）上，使用 3 分钟历史预测 10 分钟未来轨迹。FlightDiff 在所有预测步长（1/3/9/15/30 步）和所有坐标（经度/纬度/高度）上均取得最优 MAE、MAPE、RMSE 和 MDE[^src-flightdiff]。

相比最强基线 FlightBERT++，经度 MAE 平均降低 11.7%，高度 MAE 平均降低 13.6%。优势随预测步长增加而扩大，30 步时 MDE 降低 24%。

### 到达时间预测（FAT）

在四个剩余时间区间（<15、15–20、20–30、30–60 分钟）上，FlightDiff 均取得最优 MAE 和 RMSE。关键 20–30 分钟区间 MAE 从 FlightBERT++ 的 5.24 分钟降至 3.32 分钟。

### 消融实验

- 移除共享 VAE：轨迹重建质量下降，航路点编码精度受损
- 移除目的地约束：30 步 MDE 从 7.24 km 升至 7.60 km
- 移除航路点约束：30 步 MDE 升至 7.42 km
- 单阶段扩散（仅 UNet1 或 UNet2）：性能均不如两阶段方案
- α 参数：α=0.7 在各时间区间取得最均衡结果

## 局限性

- 扩散模型顺序去噪导致推理速度慢于确定性模型（每批 1 分钟 vs 基线 1 秒）
- 当前仅考虑单架飞机，未建模多机交互
- 航路点数据不含高度信息，限制了三维约束的完整性

[^src-flightdiff]: [[source-flightdiff]]