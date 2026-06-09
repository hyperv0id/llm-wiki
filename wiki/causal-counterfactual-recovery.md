---
title: "Causal Counterfactual Recovery via Encode–Decode Flow (DoFlow)"
type: technique
tags:
  - causal-inference
  - counterfactual
  - continuous-normalizing-flow
  - abduction-action-prediction
  - structural-causal-model
  - iclr
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# 反事实恢复：基于编码-解码流的因果反事实生成（DoFlow）

这是 [[doflow|DoFlow]] 用连续归一化流（[[continuous-normalizing-flow|CNF]]）在时间序列上生成反事实轨迹的核心机制，以及其相应的理论保证（反事实恢复，Corollary 4.5）[^src-doflow]。

## Abduction–Action–Prediction

反事实预测遵循 Pearl 的标准三步法[^src-doflow]：

1. **Abduction（溯因）**：将每个观测到的事实值通过 CNF 的**前向过程**编码为潜表示 $z_{i,t}^F = \Phi_\theta(x_{i,t}^F; H_{i,t-1}^F)$，其中 $H_{i,t-1}^F$ 是从过去事实观测计算的事实隐藏状态。该潜变量隐式捕获了未观测的外生因素（如患者的基线健康状况）[^src-doflow]。
2. **Action（干预）**：施加指定的干预调度 $I$，即 $\text{do}(X_I:=\gamma_I)$[^src-doflow]。
3. **Prediction（预测）**：从溯因得到的潜变量 $z_{i,t}^F$ 出发，通过**反向过程**在反事实隐藏状态下解码 $\hat{x}_{i,t}^{CF}=\Phi_\theta^{-1}(z_{i,t}^F; \hat{H}_{i,t-1}^{CF})$[^src-doflow]。

关键区分（见原文 Figure 1）[^src-doflow]：
- **事实隐藏状态** $H_{i,t-1}^F$ 由观测到的事实值 $(x_{i,t-1}^F, x_{\text{pa}(i),t-1}^F)$ 更新，**仅用于编码**事实值为潜表示。
- **反事实隐藏状态** $\hat{H}_{i,t-1}^{CF}$ 由先前预测的反事实值 $(\hat{x}_{i,t-1}^{CF}, \hat{x}_{\text{pa}(i),t-1}^{CF})$ 自回归更新，**用于解码**。

被干预节点 $(i,t)\in I$ 直接设为 $\hat{x}_{i,t}^{CF}:=\gamma_{i,t}$，并相应更新其 RNN 状态[^src-doflow]。由于固定了从事实样本溯因得到的外生噪声，反事实在 DoFlow 中产生**单一确定性轨迹**（因此论文只对观测与干预计算 MMD/CRPS）[^src-doflow]。

## 反事实恢复定理

在以下假设下（Assumption 4.1）[^src-doflow]：
- **(A1)** 外生噪声 $U_t \perp\!\!\!\perp (X_{t^-}, X_{\text{pa},t^-})$。
- **(A2)** 结构方程 $f(\cdot, U_t)$ 关于 $U_t$ 严格单调且连续（在加性 SCM $X_t = f^*(\cdot)+U_t$ 下自动满足）[^src-doflow]。
- **(A3)** 编码潜变量满足 $p_\theta(Z_t\mid H_{t-1})=q(Z_t)$，即编码 $Z_t$ 与条件历史在分布上独立（无限数据与精确训练的极限下精确成立）[^src-doflow]。

**Proposition 4.3**：编码潜变量 $Z_t = \Phi_\theta(X_t; H_{t-1}) = g(U_t)$ 是外生噪声 $U_t$ 的一个连续可微双射函数，且**对 $H_{t-1}$ 不变**[^src-doflow]。直觉：CNF 把任意固定父历史下的 $X_t$ 都映射到同一基分布，故潜变量只编码外生噪声。

**Corollary 4.5（反事实恢复）**：在 Assumption 4.1 下，编码-解码过程几乎必然恢复真实反事实：$\hat{X}_t^{CF} := \Phi_\theta^{-1}(Z_t^F; \hat{H}_{t-1}^{CF}) = X_t^{CF}$[^src-doflow]。证明通过对时间步归纳完成[^src-doflow]。

## 与 BGM 的关系

(A1)–(A2) 与 Bijective Generation Mechanisms (BGM, Nasr-Esfahany et al. 2023) 的假设相呼应[^src-doflow]。但两者不同：BGM 额外要求观测分布匹配（(A4)）以获得**类级别**的模型无关可识别性；而 DoFlow 的 Proposition 4.3 与 Corollary 4.5 不假设 (A4)，转而依赖 CNF 特定的条件 (A3)，提供**模型特定的逐点恢复**[^src-doflow]。论文也给出一条替代路线：若额外加上 (A4) 观测匹配，则 DoFlow 落入 BGM 类，可直接援引 BGM 的反事实恢复结果[^src-doflow]。

## 链接

- [[doflow]] — DoFlow 主页
- [[source-doflow]] — source summary
- [[continuous-normalizing-flow]] — CNF 可逆编码-解码基础
- [[causal-time-series-forecasting]] — 因果时间序列预测范式（观测/干预/反事实）

[^src-doflow]: [[source-doflow]]
