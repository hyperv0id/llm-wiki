---
title: "TSFlow — Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting"
type: source-summary
tags:
  - flow-matching
  - time-series
  - probabilistic-forecasting
  - gaussian-process
  - optimal-transport
  - diffusion-models
  - iclr-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# TSFlow — Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting

**Authors:** Marcel Kollovieh, Marten Lienen, Leo Schwinn, David Lüdke, Stephan Günnemann (TU Munich)
**Venue:** ICLR 2025
**arXiv:** 2410.03024v2
**Code:** [github.com/marcelkollovieh/TSFlow](https://github.com/marcelkollovieh/TSFlow)

## 核心贡献

TSFlow 是首个将条件流匹配 (Conditional Flow Matching, CFM) 应用于概率时间序列预测的模型[^src-tsflow]。它引入三个关键创新：

1. **高斯过程先验**：替代扩散模型中默认的各向同性高斯先验，使用三种 GP 核函数——平方指数 (SE)、Ornstein-Uhlenbeck (OU) 和周期 (PE) 核——将先验分布与数据的时序结构对齐[^src-tsflow]。
2. **最优传输耦合**：通过 mini-batch 最优传输在训练时配对先验样本与数据样本，缩短概率路径、降低训练方差[^src-tsflow]。
3. **双重条件化策略**：无条件模型通过条件先验采样 (Langevin 动力学) + 引导实现条件预测；条件模型直接使用 GP 回归先验训练[^src-tsflow]。

## 方法概览

TSFlow 在 CFM 框架中定义两个正交时间维度：流匹配时间 $t \in [0,1]$ 参数化从先验到数据的变换路径，而 $\tau$ 是时间序列内部的时间索引[^src-tsflow]。

### 无条件模型 (Sec. 3.1)
- 使用 GP 先验 $q_0 = \mathcal{GP}(0, K)$ + 最优传输耦合训练[^src-tsflow]
- 推理时通过 **条件先验采样** (CPS)：Langevin 动力学从 $q_0(x_0 \mid y^p)$ 采样，其中 $\nabla_{x_0} \log q_0(x_0 \mid y^p) = \nabla_{x_0} \log q_1(y^p \mid x_0) + \nabla_{x_0} \log q_0(x_0)$[^src-tsflow]
- 然后通过 **引导生成** 修改向量场：$\tilde{u}_\theta(t, x_t) = u_\theta(t, x_t) - s\sigma_t \nabla_{x_t} \log p_t(y^p \mid x_t)$[^src-tsflow]

### 条件模型 (Sec. 3.2)
- 联合分布 $q(x_0, y) = q_1(y) q_0(x_0 \mid y^p)$，其中 $q_0(x_0^p \mid y^p) = \delta(x_0^p - y^p)$ 和 $q_0(x_0^f \mid y^p) = \mathcal{N}(\mu_{f|p}, \Sigma_{f|p})$ 通过 GP 回归解析计算[^src-tsflow]
- 训练和推理使用一致的条件先验分布

### 架构
- DiffWave 风格，3 个残差块，S4 层沿时间维度操作
- 隐藏维度 64，~176k 可训练参数
- 64 维正弦时间嵌入，Euler ODE 求解器 (32 步)

## 实验结果

在 8 个真实数据集（Electricity、Exchange、KDDCup、M4-Hourly、Solar、Traffic、UberTLC、Wikipedia）上评估[^src-tsflow]：

- **无条件生成**：GP 先验在 4 NFE 下超越各向同性先验在 16 NFE 下的性能。周期核 (PE) 在 6/8 数据集上 LPS 最优[^src-tsflow]。
- **概率预测**：TSFlow-Cond. 在 6/8 数据集上取得 SOTA CRPS，较第二优方法提升最高 14%。OU 核条件模型全面超越 CSDI、SSSD、TSDiff 和 Biloš et al. (2023)，且 NFE 更少[^src-tsflow]。
- **无条件→条件桥接**：TSFlow-Uncond. 通过 CPS + 引导在 7/8 数据集上达到或超越仅引导的模型[^src-tsflow]。

## 局限性

- 仅在单变量时间序列上验证[^src-tsflow]
- 不同数据集/任务需要选择不同核函数，需在验证集上优化[^src-tsflow]
- 未来方向：扩展到多元时间序列（多元 GP）、使用更复杂的先验分布（如基于数据统计的先验）[^src-tsflow]

[^src-tsflow]: [[source-tsflow]]
