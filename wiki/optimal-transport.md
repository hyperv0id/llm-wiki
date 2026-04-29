---
title: "Optimal Transport"
type: concept
tags:
  - optimal-transport
  - mathematics
  - probability
  - wasserstein-distance
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Optimal Transport

**最优传输**（Optimal Transport, OT）是概率论和微分几何中的核心理论，研究如何以最小"成本"将一个概率分布映射到另一个概率分布[^src-flow-matching]。

## 核心问题：Monge 问题

给定两个概率分布 $\mu$ 和 $\nu$ 在 $\mathbb{R}^d$ 上，以及成本函数 $c(x, y): \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^+$，寻找一个映射 $T: \mathbb{R}^d \to \mathbb{R}^d$，使得：

$$
\inf_{T_\# \mu = \nu} \int_{\mathbb{R}^d} c(x, T(x)) \, d\mu(x)
$$

其中 $T_\# \mu = \nu$ 表示 $T$ 将 $\mu$ 的质量"推送"到 $\nu$（push-forward）。

### 特殊情况：二次成本

当 $c(x, y) = \|x - y\|^2$ 时，问题变为：

$$
\inf_{T_\# \mu = \nu} \mathbb{E}_{x \sim \mu}[\|x - T(x)\|^2]
$$

该问题的解 $T$ 称为**最优传输映射**。

---

## Wasserstein 距离

最优传输问题导出了 **Wasserstein 距离**（又称 Earth Mover's Distance）：

$$
W_2(\mu, \nu) = \left( \inf_{T_\# \mu = \nu} \int \|x - T(x)\|^2 \, d\mu(x) \right)^{1/2}
$$

### 性质

1. **度量**：$W_2(\mu, \nu) \geq 0$，且 $W_2(\mu, \nu) = 0 \iff \mu = \nu$
2. **三角不等式**：$W_2(\mu, \nu) \leq W_2(\mu, \omega) + W_2(\omega, \nu)$
3. **弱收敛**：$W_2(\mu_n, \mu) \to 0$ 蕴含弱收敛

---

## 最优传输与高斯分布

对于两个高斯分布 $\mu = \mathcal{N}(0, \Sigma_0)$ 和 $\nu = \mathcal{N}(\mu_1, \Sigma_1)$，最优传输映射有解析形式：

$$
T(x) = \mu_1 + A x, \quad A = \Sigma_0^{-1/2} (\Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2})^{1/2} \Sigma_0^{-1/2}
$$

### 简化情况

当 $\mu = \mathcal{N}(0, I)$，$\nu = \mathcal{N}(x_1, \sigma^2 I)$ 时：

$$
T(x) = x_1 + \sigma x
$$

即简单的平移和缩放。

---

## OT 位移插值 (Displacement Interpolation)

给定两个分布 $\mu_0$ 和 $\mu_1$，定义 OT 位移插值路径：

$$
\mu_t = [(1-t) \text{id} + t T]_\# \mu_0, \quad t \in [0,1]
$$

其中 $T$ 是从 $\mu_0$ 到 $\mu_1$ 的最优传输映射。

### 性质

- $\mu_0 = \mu$（初始分布）
- $\mu_1 = \nu$（目标分布）
- 粒子沿**直线**运动（从 $x$ 到 $T(x)$）
- 速度方向**恒定**（不随时间变化）

这正是 Flow Matching 中 OT 路径的理论基础！

---

## 在生成模型中的应用

### Flow Matching

Flow Matching 使用 OT 位移插值作为概率路径：

$$
p_t(x \mid x_1) = \mathcal{N}(x \mid t x_1, (1 - (1-\sigma_{\min})t)^2 I)
$$

对应的向量场（OT 条件向量场）：

$$
u_t(x \mid x_1) = \frac{x_1 - (1 - \sigma_{\min}) x}{1 - (1 - \sigma_{\min})t}
$$

### 优势

1. **直线轨迹**：比扩散路径的曲线轨迹更简单
2. **恒定方向**：向量场可分离为 $g(t) \cdot h(x)$，更容易拟合
3. **更快收敛**：实验表明 OT 路径训练更快、采样更高效

---

## 相关页面

- [[flow-matching]] — 流匹配框架
- [[continuous-normalizing-flow]] — 连续归一化流
- [[diffusion-model]] — 扩散模型

## 引用

[^src-flow-matching]: [[source-flow-matching]]