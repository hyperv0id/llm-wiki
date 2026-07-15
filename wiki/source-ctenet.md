---
title: "CTENet: Chemical Transport Eulerian Network for Air Quality Forecasting"
type: source-summary
tags:
  - air-quality
  - physics-informed
  - pinn
  - eulerian
  - advection-diffusion-reaction
  - spatio-temporal
  - neurips-2025
created: 2026-07-14
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# CTENet: Chemical Transport Eulerian Network for Air Quality Forecasting

**Authors:** Xukai Zhang, Shuliang Wang, Guangyin Jin, Ziqiang Yuan, Hanning Yuan, Sijie Ruan (Beijing Institute of Technology, Sapienza University of Rome)
**Venue:** NeurIPS 2025
**Code:** https://github.com/santafirefox0/CTENet

## Summary

CTENet 提出一种将化学传输模型（CTM）与深度学习融合的空气质量预测框架，核心创新在于将 Advection-Diffusion-Reaction（ADR）偏微分方程嵌入神经网络架构中，采用**欧拉连续空间表示**替代传统离散站点图表示。[^src-ctenet]

传统空气质量预测方法（无论是纯物理模型还是图神经网络）均在离散监测站点上建模，忽略了污染物浓度的空间连续性。CTENet 通过 RBF 插值将稀疏站点数据转化为连续欧拉场，并在欧拉 ADR 解码器中显式模拟平流（风驱动传输）、扩散（湍流混合）和化学反应（光化学/二次气溶胶生成）。[^src-ctenet]

## 方法架构

1. **欧拉污染编码器**：Multiquadric RBF 插值将离散站点浓度映射到连续空间网格，解决站点稀疏覆盖问题。[^src-ctenet]
2. **气象编码器**：提取风矢量（东西/南北分量）用于平流建模，1×1 卷积降维气象通道，含 Wind Predictor 和 Meteorology Predictor（可替换的时空序列预测器，如 ConvLSTM/TAU）。[^src-ctenet]
3. **欧拉 ADR 解码器**：多层结构，每层以 FTCS 有限差分离散化 ADR 方程。平流项显式计算风驱动传输；扩散项含可学习扩散系数 kθ；反应项通过 sigmoid 门控将气象特征映射为 (0,1) 调制系数，模拟环境引导的化学反应。[^src-ctenet]

## 实验结果

在中美两个真实数据集（中国 480 站点、美国 365 站点，2018 全年）上，CTENet 对比 HA、VAR、STGCN、DCRNN、GTS、Airformer、AirPhyNet、PM2.5-GNN、TAU 等基线，RMSE 分别降低 45.8%（美国）和 21.0%（中国）。消融实验验证了 ADR 三项各自的贡献（平流项最重要），以及 RBF 插值优于最近邻插值和无插值方案。[^src-ctenet]

## 局限

缺乏不确定性量化（概率预测），限制在风险敏感场景的适用性；计算复杂度 O(B·T·D²·H·W) 高于图方法 O(B·T·D²·N)，但推理 72 小时预测 <0.15 秒。[^src-ctenet]

## 交叉链接

- [[ctenet]] — CTENet 模型实体
- [[physics-informed-neural-network]] — PINN 概念，CTENet 属于架构嵌入型 PINN
- [[advection-diffusion-reaction-equation]] — ADR 方程
- [[air-quality-forecasting]] — 空气质量预测领域
- [[radial-basis-function-interpolation]] — RBF 插值技术
- [[source-pi-mfm]] — 损失约束型 PINN（对比参照）
- [[source-multimodal-pinn]] — 多模态 PINN（对比参照）
- [[multimodal-exogenous-guided-long-term-st-forecasting]] — 多模态外生信息引导长期时空预测分析

[^src-ctenet]: [[source-ctenet]]
