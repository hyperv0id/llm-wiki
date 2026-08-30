---
title: "CirT: Geometry-Inspired Spherical Transformer for S2S Climate Forecasting"
type: source-summary
tags:
  - weather-forecasting
  - s2s
  - transformer
  - spherical-geometry
  - fourier-transform
  - iclr-2025
created: 2026-07-14
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# CirT: Geometry-Inspired Spherical Transformer for S2S Climate Forecasting

Liu, Zheng, Cheng, Tsung, Zhao, Rong & Li (HKUST Guangzhou + HKUST + DAMO Academy, Alibaba) 在 ICLR 2025 上提出 **CirT（Circular Transformer）**，一种面向次季节到季节（S2S）气候预测的几何启发 Transformer 架构[^src-cirt]。

## 核心问题

S2S 预测（提前 2–6 周）被称为"可预测性荒漠"——时间跨度太长以致丢失大气初始条件记忆，又太短来不及让海洋等缓变组分主导。现有数据驱动模型（FourCastNetV2、PanguWeather、[[graphcast|GraphCast]]、ClimaX）将球面经纬网格视为平面图像，产生严重的几何失真[^src-cirt]。

## 两大几何归纳偏置设计

**1. 圆形分块（Circular Patching）**：按纬度均匀划分，将每条纬线上的气象变量作为一个 circular patch。各 patch 几何形状一致，相邻 patch 等距，消除了标准固定角度分块在高纬度区域的面积不均问题[^src-cirt]。

**2. 傅里叶域自注意力**：利用 circular patch 的空间周期性（$X_w = X_{w+W}$），在每个 Transformer block 内先对 patch embedding 做 DFT，在频域中执行多头自注意力以捕获全局空间关系，再 IDFT 回到空间域。频域操作天然编码了空间周期性[^src-cirt]。

## 训练策略

与迭代自回归模型不同，CirT 直接预测 Weeks 3-4 和 Weeks 5-6 的平均值，避免累积误差。损失函数为两个预测窗口的 MSE 均值[^src-cirt]。

## 实验结果

在 ERA5（1.5° 分辨率，63 变量，1979–2018）上的实验表明[^src-cirt]：

- **全面超越数据驱动基线**：在 Weeks 3-4 和 5-6 的所有 7 个目标变量上一致优于 FourCastNetV2、GraphCast、PanguWeather 和 ClimaX。z500 Weeks 3-4 RMSE 从最佳基线 602→477（↓20.8%）。
- **超越数值预报系统**：在几乎所有气压层和变量上超过 UKMO、NCEP、CMA 和 ECMWF。
- **高纬度改善显著**：由于几何偏置，CirT 在中高纬度相对改善远大于低纬度（t500 Mid-Lat 改善 19.6% vs Low-Lat 1.2%）。
- **轻量高效**：仅 16M 参数、2.2G FLOPs，远小于 GraphCast（37M/110T）和 PanguWeather（256M/168T）。
- 消融实验验证了圆形分块和傅里叶变换两者缺一不可——仅加 FT 于 grid patch 反而可能降性能。

## 意义与局限

CirT 首次将球面几何归纳偏置系统性地引入 S2S Transformer 设计，证明显式建模地球球面几何优于隐式学习。局限包括：未利用海洋/陆面等缓变分量，2D Transformer 未建模垂直气压层间的三维交互[^src-cirt]。

## 相关页面

- [[cirt]] — CirT 模型实体
- [[subseasonal-to-seasonal-forecasting]] — S2S 预测概念
- [[spherical-geometry-inductive-bias]] — 球面几何归纳偏置
- [[circular-patching]] — 圆形分块技术
- [[fourier-self-attention]] — 傅里叶域自注意力
- [[source-climax]] — ClimaX 基线模型
- [[graphcast]] — GraphCast 基线模型
- [[multi-mesh-representation]] — 多分辨率 mesh 表征
- [[weather-foundation-model]] — 天气基础模型范式

[^src-cirt]: [[source-cirt]]
