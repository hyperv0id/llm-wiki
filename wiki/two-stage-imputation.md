---
title: "Two-Stage Imputation"
type: technique
tags:
  - time-series
  - data-imputation
  - diffusion-models
  - refinement
created: 2026-06-08
last_updated: 2026-06-08
source_count: 3
confidence: medium
status: active
---

# Two-Stage Imputation (双阶段插补)

**双阶段插补**是一种插补精炼范式，最早由 [[saits|SAITS]] 的 DMSA (Diagonal Masked Self-Attention) 块引入，被 [[sadi|SADI]] 首次应用于扩散模型插补框架[^src-sadi]。核心思想：第一阶段产生初始插补，第二阶段基于初始插补结合原始信息进行精炼，最终通过可学习加权组合融合两阶段输出[^src-sadi]。

## SADI 的双阶段实现

SADI 在去噪函数中部署两个 [[gated-temporal-attention|GTA]] 块[^src-sadi]：

1. **第一阶段 (GTA₁)**：以 FDE 输出和观测值为输入，输出 $\epsilon_1$（初始噪声预测）和注意力权重 $W_L$
2. **第二阶段 (GTA₂)**：以 GTA₁ 的隐状态 + **重新引入的原始噪声数据**为输入，输出 $\epsilon_2$（精炼的噪声预测）

重新引入原始噪声数据的设计是为了**接地**——防止第二阶段完全依赖第一阶段可能错误的插补而偏离真实分布[^src-sadi]。

## 加权组合

SADI 不直接使用第二阶段输出，而是学习动态加权[^src-sadi]：

$$\tilde{W}_L = \text{sigmoid}(\text{linear}(\text{concat}(W_L, M_0^{co})))$$

$$\epsilon_\theta = (1 - \tilde{W}_L) \odot \epsilon_1 + \tilde{W}_L \odot \epsilon_2$$

关键设计[^src-sadi]：
- $W_L$ 来自 GTA₁ 的注意力权重矩阵（$L\times L$）
- $M_0^{co}$ 缺失掩码告知模型哪些位置是观测值
- 拼接后经 FFN 投影到 $(L, K)$ 维度
- 逐元素加权，允许**每个位置独立决定**两阶段的贡献比例

## 为什么需要加权组合

SADI 作者在实践中观察到第二阶段有时会降低精度[^src-sadi]。加权组合机制允许模型在第二阶段不利于插补时自动降低其权重，避免退化。

消融实验证明[^src-sadi]：移除加权组合（直接用 $\epsilon_2$）导致 AgAID MSE 从 $2.93\times10^{-4}$ 升至 $7.81\times10^{-4}$（~2.7×），Electricity（10 缺失特征）MSE 从 0.107 飙升至 2.10（~19.6×）。

## 与其他方法的双阶段设计

| 方法 | 阶段 1 | 阶段 2 | 融合方式 |
|------|--------|--------|---------|
| [[saits\|SAITS]] | First DMSA | Second DMSA | 加权组合（来自 DMSA 注意力权重） |
| **SADI** | GTA₁ (FDE 后) | GTA₂ (噪声接地) | 动态逐元素加权 |
| [[cofill\|CoFILL]] | 时域 TCN | 频域 DCT | Cross-Attention 融合（非序列化） |
| ImputeFormer | 投影注意力 | 嵌入注意力 | 堆叠（非加权） |

SADI 的独特之处在于：(1) 第二阶段重新引入原始噪声数据作为接地信号；(2) 逐元素的独立动态权重允许细粒度控制[^src-sadi]。

## 关联页面

- [[sadi]] — SADI，将双阶段插补引入扩散模型的首次实践
- [[gated-temporal-attention]] — GTA，双阶段的核心计算单元
- [[feature-dependency-encoder]] — FDE，为第一阶段提供特征依赖信息
- [[mixed-partial-blackout-training]] — MPB，增强双阶段在 partial blackout 下的鲁棒性
- [[saits]] — SAITS，双阶段插补的注意力设计灵感来源

[^src-sadi]: [[source-sadi]]
