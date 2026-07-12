---
title: "History-Conditional Manifold"
type: technique
tags:
  - flow-matching
  - manifold
  - source-distribution
  - probabilistic-forecasting
  - time-series
created: 2026-07-12
last_updated: 2026-07-12
source_count: 1
confidence: high
status: active
---

# History-Conditional Manifold

**History-Conditional Manifold (HCM)** 是 [[kite|KITE]] 提出的可学习源分布构造：用历史内生序列动态合成 \(Y_0\)，替换 Flow Matching 默认的上下文无关高斯，使生成起点在拓扑上贴近目标预测流形。[^src-kite]

## 问题

标准 [[flow-matching|Flow Matching]] 常取 \(z_0\sim N(0,I)\)。协变量条件预测分布高度局域、结构复杂时，从原点附近的噪声走到目标，路径长、非线性强，采样延迟上升且保真度受损。[^src-kite]

[[tsflow|TSFlow]] 用固定 GP 核规则做上下文相关先验，表达力受非可学习规则瓶颈。HCM 改为**端到端可学**的历史条件源。[^src-kite]

## 参数化

源样本合成：[^src-kite]

\[
Y_0 = \mu_{\text{hist}} + \sigma_{\text{hist}}\,\delta_{\text{hist}}.
\]

### Barycenter Mapping

\[
\mu_{\text{hist}} = f_\phi(X_{\text{endo}}) \in \mathbb{R}^{N\times F}
\]

线性层或 MLP 把历史映射到预测空间的一阶中心，抓住历史依赖的局部趋势。[^src-kite]

### Uncertainty Estimator

\[
\sigma_{\text{hist}} = \mathrm{Softplus}(g_\psi(X_{\text{endo}})) + \sigma_{\min}
\]

标量（或共享）尺度做异方差覆盖；\(\sigma_{\min}\) 保底探索宽度，敏感度上常用 \([0.05, 0.15]\)。[^src-kite]

### Manifold Projector

\[
\delta_{\text{hist}} = \alpha \frac{M z}{\|M z\|} + (1-\alpha)\epsilon
\]

- \(M\in\mathbb{R}^{(N\times F)\times r}\)：可学低秩流形基；
- \(z\sim N(0,I_r)\)、\(\epsilon\sim N(0,I)\)；
- \(\alpha\) 可学，平衡结构扰动与各向同性扰动，避免纯低秩过拟合。[^src-kite]

### Coverage Constraint

\[
L_{CC} = \frac12\log\sigma_{\text{hist}}^2 + \frac{\|Y_{\text{endo}}-\mathrm{detach}(\mu_{\text{hist}})\|^2}{2\sigma_{\text{hist}}^2}
\]

阻止方差塌缩，并强制源支撑盖住真值；对 \(\mu\) stop-gradient，避免该损失扭曲质心几何优化。[^src-kite]

### 路径合成

第 \(s\) 步：[^src-kite]

\[
Y_s = s\cdot Y_{\text{endo}} + (1-s)\cdot Y_0.
\]

## 理论含义

在「历史中心化带来的条件残差下降」大于噪声尺度项时：[^src-kite]

1. **Target Scale（Prop. 1）**：\(\mathbb{E}[\|Y-Y_0^H\|_1\mid h,c] < \mathbb{E}[\|Y-Y_0^G\|_1\mid h,c]\)——匹配目标更小，梯度更稳。
2. **Path Regularity（Prop. 2）**：路径管半径更小 → 路径局部 Jacobian budget 更低 → 局部误差放大更弱。

直觉：预报目标通常围着历史依赖的局部中心，而不是 \(N(0,I)\) 原点。[^src-kite]

## 在 KITE 链路中的位置

HCM 解决**从哪出发**；[[knowledge-guided-conditioning|KGC]] 解决**沿路听谁的**；[[classifier-free-guidance|CFG]] 解决**有条件与无条件差多少**。详见 [[kite-manifold-guidance-chain]]。[^src-kite]

## 与 GP 先验源的对比

| | TSFlow GP 源 | KITE HCM |
|--|--------------|----------|
| 结构来源 | 固定核（SE/OU/PE） | 可学 \(\mu,\sigma,M,\alpha\) |
| 历史用法 | 条件 GP 回归 / 规则 | 直接映射历史到 \((\mu,\sigma)\) |
| 外生未来 | 非原生设计重点 | 与 KGC/CFG 一体 |
| 可学性 | 核超参为主 | 全模块可微 |

[^src-kite]

## 相关页面

- [[kite]] / [[source-kite]]
- [[gaussian-process-prior-flow-matching]]
- [[flow-matching]]
- [[flow-matching-forecasting]]
- [[prototype-guided-flow-matching]] — Aurora 的原型条件源，另一条“信息源”路线

[^src-kite]: [[source-kite]]
