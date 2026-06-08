---
title: "Source: FlowTS — Time Series Generation via Rectified Flow"
type: source-summary
tags:
  - rectified-flow
  - flow-matching
  - time-series-generation
  - unconditional-generation
  - conditional-generation
  - arxiv
created: 2026-06-08
last_updated: 2026-06-08
source_count: 0
confidence: medium
status: active
---

# Source: FlowTS — Time Series Generation via Rectified Flow

**Authors**: Yang Hu, Xiao Wang, Zezhen Ding, Lirong Wu, Huatian Zhang, Stan Z. Li, Sheng Wang, Jiheng Zhang, Ziyun Li, Tianlong Chen  
**Venue**: arXiv:2411.07506v3 (Feb 2025)  
**Affiliation**: Westlake University, University of Washington, HKUST, USTC, KTH, UNC Chapel Hill  
**Code**: [github.com/UNITES-Lab/FlowTS](https://github.com/UNITES-Lab/FlowTS)

## 核心问题

扩散模型在时间序列生成中取得了显著成就，但存在关键限制：采样需要数百至数千次 drift function 评估，计算成本过高。FlowTS 提出用 rectified flow 替代迭代式 ODE/SDE 求解器，通过概率空间中的直线输运实现高效生成。

## 核心方法

1. **Rectified Flow for TS Generation**：学习一个 ODE $\frac{dZ_t}{dt} = v(Z_t, t)$，将 $Z_0 \sim \mathcal{N}(0, I)$ 沿直线路径 $(Z_1 - Z_0)$ 输运到目标分布。通过最小二乘回归 $\mathcal{L} = \mathbb{E}\|(Z_1-Z_0) - G(Z_t, t)\|^2$ 训练，$t$ 从 Logit-Normal 分布采样。

2. **Adaptive Sampling**：受探索-利用权衡启发，使用缩放因子 $t^k$（$k \in (0, 1]$），早期鼓励探索（大噪声），后期聚焦利用（密集小步长）。采样迭代数增加时最优 $k$ 减小。

3. **Trend-Seasonality Decomposition**：集成 Trend Synthetic Layers 和 Fourier Synthetic Layers，将预测分解为残差、均值、趋势和季节性分量。

4. **Attention Registers**：借鉴 Vision Transformer 的 register tokens 设计，作为持久记忆单元捕获全局上下文模式，序列 token 聚焦局部时序依赖。

5. **Rotary Position Embedding (RoPE)**：自然衰减特性与时序依赖性对齐，灵活处理变长序列。

6. **Seamless Conditional Adaptation**：无条件训练的模型无需重新训练即可直接适应条件任务——条件生成时用已观测值替换对应位置，迭代精炼。

## 实验结果

- **无条件生成 (24-length)**：Context-FID 分数 Stocks 0.019、ETTh 0.011（此前最优 0.067、0.061）
- **无条件生成 (256-length)**：Context-FID 0.302（Diffusion-TS 0.423）
- **预测 (Solar)**：MSE 213，相对此前最优 375 降低 **43.2%**
- **插补 (MuJoCo)**：70% 缺失率 MSE 7e-5，相对 Diffusion-TS (2.7e-4) 降低 **74.1%**
- **效率**：仅 30 采样步 (N=30) 和 2,500 训练迭代即可超越 Diffusion-TS 的 200 步 + 10,000 迭代
- **消融**：RoPE (50k 频率) + 128 个 attention register tokens 组合最优，去除两者 FID 显著恶化

## 局限性

- arXiv 预印本，未经同行评审
- 短序列 (24) 和长序列 (256) 实验覆盖数据集中等

## 与现有工作的关系

FlowTS 是首个将 rectified flow 用于时间序列生成的模型，既支持无条件生成也支持预测和插补。在无条件生成上与 Diffusion-TS 直接对标，在条件预测上与 CSDI、SSSD 对标。与 TSFlow (CFM+GP 先验) 不同，FlowTS 使用 rectified flow 的直线输运，强调计算效率和自适应采样。
