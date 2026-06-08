---
title: "Adaptive Sampling in Flow Matching"
type: technique
tags:
  - flow-matching
  - rectified-flow
  - sampling
  - exploration-exploitation
  - ode
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Adaptive Sampling in Flow Matching

**Adaptive Sampling**（自适应采样）是 FlowTS 提出的 ODE 采样策略，受强化学习中探索-利用权衡的启发，动态调整采样时间步的密度分布，以平衡噪声适应和精度[^src-flowts]。

## 动机

标准流匹配中的采样时间步通常均匀分布在 $[0, 1]$ 上（或从 Logit-Normal 分布采样）。然而，不同时间区域的难度不同[^src-flowts]：

- **早期阶段 ($t \to 0$)**：噪声大，需要更多探索来覆盖可能路径
- **后期阶段 ($t \to 1$)**：接近目标分布，需要精细的小步长来确保精度

均匀采样无法区分这些需求。

## 方法

引入自适应缩放因子 $k \in (0, 1]$，将时间步计算为[^src-flowts]：

$$t_i = \left(\frac{i}{N}\right)^k$$

其中 $N$ 是总采样迭代数，$i = 0, 1, \dots, N-1$。

### 效应

- $k = 1$：**均匀采样**，时间步等距分布
- $k < 1$：**自适应采样**，早期时间步稀疏（鼓励探索大噪声变化），后期时间步密集（精细粒度的利用）

这种非线性映射在保持总迭代数不变的前提下，重新分配计算预算。

## 实验发现

FlowTS 在 MuJoCo 插补和 Solar 预测任务上的实验表明[^src-flowts]：

1. **自适应采样一致优于均匀采样**：对于所有 $N$ 和任务设置，$k < 1$ 的 MSE 低于 $k = 1$
2. **$k$ 与 $N$ 的反比关系**：随着采样迭代数增加，最优 $k$ 减小。大的 $N$ 意味着每个时间步已有足够精度，可以更具侵略性地偏向早期探索
3. **更稳定的推理**：$k$ 减小时 MSE 持续下降，表明更小的 $k$ 带来更稳定、方差更低的推理

例如，在 MuJoCo 70% 缺失率插补任务中，$N=100$ 时最优 $k \approx 0.3$ 的 MSE 为 $6.53 \times 10^{-5}$[^src-flowts]。

## 算法

### 无条件生成

```
for i = 0 to N-1:
    t_{i+1} = ((i+1)/N)^k
    v_{t_i} = G(Z_{t_i}, t_i)
    Z_{t_{i+1}} = Z_{t_i} + (t_{i+1} - t_i) * v_{t_i}
```

### 条件生成

条件生成中同样使用自适应时间步，外加观测值替换步骤[^src-flowts]。

## 与相关技术的比较

| 技术 | 原理 | 参数 | 来源 |
|------|------|------|------|
| **Adaptive Sampling** | 探索-利用权衡，$t^k$ 缩放 | $k \in (0, 1]$ | FlowTS (2025) |
| Logit-Normal Sampling | 从 Logit-Normal 分布采样 $t$ | $\mu, \sigma$ | SD3, DiTS |
| Uniform Sampling | 均匀采样 $t \sim \mathcal{U}[0, 1]$ | 无 | 标准做法 |
| Learned Schedule | 学习最优噪声调度 | 可学习参数 | StaTS (2026) |

## 相关页面

- [[flowts]] — FlowTS 模型实体
- [[rectified-flow-for-time-series]] — Rectified Flow 在时间序列中的应用
- [[flow-matching]] — Flow Matching 理论基础
- [[exploration-vs-exploitation]] — 强化学习中的探索与利用权衡

[^src-flowts]: [[source-flowts]]
