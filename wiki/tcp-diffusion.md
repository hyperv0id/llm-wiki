---
title: "TCP-Diffusion"
type: entity
tags:
  - tropical-cyclone
  - precipitation-forecasting
  - diffusion-model
  - multi-modal
  - deep-learning
created: 2026-07-24
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# TCP-Diffusion

TCP-Diffusion（Tropical Cyclone Precipitation Diffusion）是论文自称（To our knowledge）首个基于深度学习的全球热带气旋（TC）降水预测模型，由 Huang, Mu, Bai & Watson（浙江工业大学 + 布里斯托大学）发表于 ICML 2025[^src-tcp]。模型以 TC 中心周围 10°×10° 区域为目标，基于过去 12h 观测和多模态环境变量，以 3h 间隔预测未来 12h 的降水场[^src-tcp]。

## 核心设计

### Adjacent Residual Prediction (ARP)

不同于直接预测绝对降水值，[[adjacent-residual-prediction|ARP]] 将训练目标改为预测相邻时间步的降水变化量（$\Delta$ Rainfall），最终通过累积得到绝对降水值[^src-tcp]。此机制赋予模型"变化感知"（change awareness）能力——降水强度和空间格局的变化与历史观测趋势保持一致，从而减少累积误差并确保物理一致性[^src-tcp]。

### 多模态编码器架构

为处理异构气象数据，TCP-Diffusion 构建三个专用编码器[^src-tcp]：

- **Historical Data2d Encoder**：3D CNN 编码 2D 数据（MSWEP 降水、ERA5 地表变量、ERA5 气压层变量）
- **Historical Data1d Encoder**：MLP + Transformer 编码标量 TC 变量（强度、移动速度、月份、轨迹位置）
- **Future Data2d Encoder**：ResNet-18 编码 ERA5-IFS 未来预测数据，为扩散去噪过程提供物理引导

### EA-3DUNet 去噪网络

核心去噪网络在 3DUNet 基础上添加空间注意力（SA）和时间注意力（TA）模块，每个编码器模块包含 2 个 CNN block + SA block + TA block + 下采样/上采样 block[^src-tcp]。去噪过程从纯噪声开始，通过 $N=200$ 步逐步去噪，每步接收编码器提取的条件信息。

## 性能

- 1877 个全球 TC（1980-2020，六大洋区），126 个测试 TC（2018-2020）[^src-tcp]
- ETS-24（中等降水 >24 mm/3h）：0.147 vs PreDiff 0.119，提升 ~23%[^src-tcp]
- ETS-60（强降水 >60 mm/3h）：0.00644，唯一超越 Persistence 的 DL 模型[^src-tcp]
- 超越 ECMWF-IFS：ETS-6 0.412 vs 0.302，TPM AE 0.474 vs 0.507[^src-tcp]
- 消融实验：ARP（+0.1~15.0%）、多模态数据（+5.6~13.0%）、NWP 引导（+2.0~9.3%）[^src-tcp]

## 局限

预测时效仅 12h；扩散模型推理较慢（1.253 s/sample）；以 ERA5-IFS 为条件输入（论文消融显示无该输入时仍优于基线）；TC 形成期和消散期预测性能较弱[^src-tcp]。

## 相关页面

- [[diffusion-models|扩散模型]]
- [[extreme-weather-forecasting|极端天气预测]]
- [[precipitation-nowcasting|降水临近预报]]
- [[tropical-cyclone-precipitation-forecasting|TC 降水预测]]
- [[adjacent-residual-prediction|ARP]]

[^src-tcp]: [[source-tcp]]
