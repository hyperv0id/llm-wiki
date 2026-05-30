---
title: "Spatial Imputation Decoder"
type: technique
tags:
  - graph-neural-network
  - data-imputation
  - spatio-temporal
  - message-passing
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# Spatial Imputation Decoder

空间填补解码器是 [[grin]] 的核心创新之一，通过两阶段填补流程和**仅邻居约束**实现基于空间依赖的缺失值重建[^src-2108-00298]。

## 两阶段填补流程

### 第一阶段：线性读出

$$\hat{Y}_t = H_{t-1} V_h + b_h$$

$$\tilde{X}_t^{(1)} = \Phi(\hat{Y}_t) = M_t \odot X_t + \bar{M}_t \odot \hat{Y}_t$$

从 MPGRU 隐藏表示直接线性预测，经 filler 算子 $\Phi$ 仅替换缺失位置[^src-2108-00298]。

### 第二阶段：消息传递精化

$$s_t^i = \gamma\left(h_{t-1}^i, \sum_{j \in \mathcal{N}(i)/\{i\}} \rho\left(\Phi(\hat{x}_t^{j(1)}) \| h_{t-1}^j \| m_t^j\right)\right)$$

关键约束：**仅聚合来自邻居 $j \in \mathcal{N}(i) / \{i\}$ 的消息，排除节点 $i$ 自身**。这迫使模型必须从邻居节点的观测值和表示推断目标节点的缺失值，而非依赖自身的（可能缺失的）输入特征[^src-2108-00298]。

$$\hat{Y}_t^{(2)} = [S_t \| H_{t-1}] V_s + b_s$$

$$\tilde{X}_t^{(2)} = \Phi(\hat{Y}_t^{(2)})$$

精化填补作为下一步 MPGRU 的输入。

## 仅邻居约束的正则化效果

这个约束的设计动机是：将每条边视为软功能依赖，约束相应节点的观测值。通过仅从邻居重建，模型被强制学习局部空间依赖模式，产生正则化效果[^src-2108-00298]。

实验证据：
- 移除空间解码器后 AQI MAE 从 14.73 升至 15.40 (+4.5%)
- METR-LA Block 场景 MAE 从 2.03 升至 2.32 (+14.3%)
- GRIN 在孤立节点（距离 > 40km 的断连节点）上仍优于 BRITS，证明全局图结构提供的间接正则化也有效

## 与去噪自编码器的对比

消融实验对比了"去噪式解码器"（使用当前时间步的隐藏表示而非仅邻居约束）：

| 方法 | AQI MAE | METR-LA Block MAE |
|------|---------|-------------------|
| GRIN 空间解码器 | **14.73** | **2.03** |
| 去噪式解码器 | 17.23 (+17.0%) | 2.96 (+45.8%) |

去噪式在块缺失下严重退化，因为它依赖当前步的隐藏表示（包含缺失值污染），而仅邻居约束绕过了这一问题[^src-2108-00298]。

## 影响与后续

GRIN 的空间解码器设计确立了"仅从邻居重建"的填补范式。后续工作在以下方向演进：
- [[gsli]] (AAAI 2025)：多尺度图结构学习增强空间解码的图质量
- [[imputeformer]] (KDD 2024)：用空间嵌入注意力替代显式 MPNN，隐式建模空间依赖

[^src-2108-00298]: [[source-2108-00298]]
