---
title: "Koopman Linearization for Forecasting"
type: concept
tags:
  - koopman-operator
  - time-series
  - dynamical-systems
  - linearization
  - forecasting
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Koopman Linearization for Forecasting

> **编排者备注（去重）**：本页是从 [[k2vae|K²VAE]] ingest 中提议的通用概念页。仓库中已有 [[micro-macro-coupled-koopman-modeling]] 包含一节 "Koopman Operator Theory"，并存在 `koopman-operator` 标签（无同名页面）。请将本页与 MMCKM 的 Koopman 理论段落合并/去重，作为多个 Koopman 时间序列模型（K²VAE、MMCKM）共享的概念枢纽。

**Koopman 线性化**是一类把非线性时间序列预测转化为**线性动力系统**建模的范式。其理论基础是 Koopman 理论 (Koopman, 1931)：对非线性系统 $x_{k+1}=f(x_k)$，存在一个（理论上无限维的）**测量函数空间** $\psi$，系统状态映射到该空间后，其演化可由一个**线性 Koopman 算子** $\mathcal{K}$ 描述[^src-k2vae]：

$$\psi(x_{k+1}) = \psi(f(x_k)) = \mathcal{K}\circ\psi(x_k)$$

## 为什么对预测有用

非线性使概率/确定性模型都难以写出简洁的状态转移方程，长期预测尤其困难。把系统提升 (lift) 到测量空间后获得线性演化，带来三个好处[^src-k2vae]：

1. **建模简化**：线性系统的状态转移、不确定性传播都有闭式或可微形式；
2. **长期稳健**：线性外推比逐步非线性回归更不易累积误差；
3. **高效**：避免扩散/流模型的多步迭代采样。

## 有限维近似

无限维 Koopman 算子需有限维近似。常见做法[^src-k2vae]：

- **DMD / eDMD (Dynamic Mode Decomposition / extended DMD)**：通过矩阵计算（伪逆）从观测数据拟合线性算子。K²VAE 用 **one-step eDMD** 拟合局部算子 $K_{loc}=X_{fore}(X_{back})^{\dagger}$，并加全局可学习项 $K_{glo}$ 兜底数值不稳定[^src-k2vae]。
- **神经网络测量函数**：用 MLP 等学习 $\psi$（及其逆 $\psi^{-1}$ 作为解码器），数据驱动地构造测量空间。

## 关键性质与代表模型

- **Markov 性 / history-free**：当测量函数时不变时，预测仅依赖当前状态而非历史轨迹（[[mmckm|MMCKM]] 据此实现 history-free 交通预测）[^src-mmckm]。
- **控制输入扩展**：可加入控制项 $z_{k+1}=Kz_k+Bu_k$，把外部影响或非线性残差注入线性系统——[[mmckm|MMCKM]] 用 CrossAttention 注入宏观流影响；[[k2vae|K²VAE]] 用 KalmanNet 注入非线性残差以精炼不确定性[^src-k2vae][^src-mmckm]。
- **代表模型**：[[k2vae|K²VAE]]（通用长期概率预测，Koopman + Kalman）、[[mmckm|MMCKM]]（交通流，微观/宏观双尺度 Koopman）、Koopa（非平稳确定性长期预测，K²VAE/MMCKM 的基线）[^src-k2vae][^src-mmckm]。

## 关联页面

- [[k2vae]] — Koopman 线性化 + Kalman 精炼的概率预测
- [[mmckm]]、[[micro-macro-coupled-koopman-modeling]] — 微观/宏观双尺度 Koopman 交通模型
- [[kalmannet-uncertainty-modeling]] — 在 Koopman 线性系统上做 Kalman 精炼
- [[generative-time-series-forecasting]] — 生成式预测范式

[^src-k2vae]: [[source-k2vae]]
[^src-mmckm]: [[source-mmckm]]

