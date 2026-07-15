---
title: "Air Quality Forecasting"
type: concept
tags:
  - air-quality
  - spatio-temporal
  - environmental-monitoring
  - physics-informed
created: 2026-07-14
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# Air Quality Forecasting

空气质量预测是智慧城市和环境监测的核心任务之一，目标是根据历史和气象条件预测未来时刻的污染物浓度（PM2.5、PM10、O3、SO2、NOx、CO 等）。[^src-ctenet]

## 建模方法演进

### 物理模型
基于大气动力学 PDE 的数值模拟（如高斯烟羽模型），可解释性强但计算昂贵、依赖高质量输入。[^src-ctenet]

### 数据驱动模型
- **经典 ML**：随机森林、SVM，时间/空间建模能力有限[^src-ctenet]
- **深度学习**：CNN-LSTM、图神经网络（STGCN、DCRNN、GTS）、Transformer（Airformer），均在离散站点上建模[^src-ctenet]
- **物理引导 DL**：AirPhyNet（图边平流-扩散约束）、CTENet（欧拉 ADR 架构嵌入）[^src-ctenet]

## 核心挑战

1. **空间连续性**：站点数据稀疏，但污染物在空间上连续分布。传统图/时序方法忽略此特性。[^src-ctenet]
2. **物理化学驱动因素建模不足**：光化学反应、气粒转化等二次污染生成机制常被忽略。[^src-ctenet]
3. **外生信息利用**：气象条件（风、温、辐射、边界层高度）对污染物传输和转化的调制作用未被充分利用。[^src-ctenet]

## CTENet 的方案

CTENet 是首个在空气质量预测中采用**欧拉连续空间表示**的 PINN 方法：RBF 插值实现稀疏站点到连续场的转换，ADR 方程嵌入架构显式建模平流、扩散和化学反应，气象数据通过专用编码器和预测器融入。[^src-ctenet]

## 相关页面

- [[ctenet]] — CTENet 模型
- [[advection-diffusion-reaction-equation]] — 核心 PDE
- [[physics-informed-neural-network]] — PINN 方法论
- [[multimodal-exogenous-guided-long-term-st-forecasting]] — 外生信息引导长期 ST 预测

[^src-ctenet]: [[source-ctenet]]
