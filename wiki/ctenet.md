---
title: "CTENet"
type: entity
tags:
  - air-quality
  - physics-informed
  - pinn
  - eulerian
  - spatio-temporal
  - neurips-2025
created: 2026-07-14
last_updated: 2026-07-14
source_count: 1
confidence: medium
status: active
---

# CTENet (Chemical Transport Eulerian Network)

CTENet 是北京理工大学与罗马萨皮恩扎大学提出的空气质量预测深度学习模型，发表于 NeurIPS 2025。其核心创新是将化学传输模型（CTM）中的 Advection-Diffusion-Reaction（ADR）偏微分方程**嵌入神经网络架构**（而非损失函数），并在**欧拉连续空间**而非离散站点图上建模污染物演化。[^src-ctenet]

## 与现有方法的区别

| 维度 | 传统方法 | CTENet |
|------|---------|--------|
| 空间表示 | 离散站点图/多变量时序 | 欧拉连续场（RBF 插值） |
| 物理融入方式 | 损失函数约束（soft） | 架构内嵌 FTCS 离散化（hard） |
| 平流建模 | 图边扩散卷积模拟 | 风矢量场显式计算 |
| 化学反应 | 大多忽略 | Sigmoid 门控气象特征调制 |
| 空间连续性 | 无显式连续建模 | RBF 插值 + 全空间网格预测 |

## 架构组件

- **欧拉污染编码器**：Multiquadric RBF 插值将站点浓度转连续场[^src-ctenet]
- **气象编码器**：1×1 卷积 + Wind/Meteorology Predictors[^src-ctenet]
- **欧拉 ADR 解码器**：FTCS 离散化 ADR 方程的多层网络，含可学习扩散系数 kθ 和 sigmoid 门控反应项[^src-ctenet]
- **污染物预测器**：对 ADR 解码器的输出做最终预测并 query 回站点位置[^src-ctenet]

## 性能

中美数据集上 RMSE 分别降低 45.8%（美国）和 21.0%（中国），单张 RTX 4090 训练 3-6 小时，推理 72 小时预测 <0.15 秒。[^src-ctenet]

[^src-ctenet]: [[source-ctenet]]
