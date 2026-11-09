---
title: "Optimal Transport"
type: concept
tags:
  - optimal-transport
  - mathematics
  - probability
  - wasserstein-distance
created: 2026-04-28
last_updated: 2026-11-09
source_count: 5
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

## 求解方法

### 静态 vs 动态公式化

OT 的求解方法可分为两大类：

- **静态方法**：直接寻找映射 $T$ 或对偶势（如 Kantorovich 对偶、Sinkhorn 算法）
- **动态方法**：将 OT 转化为 Wasserstein 测地线的连续演化问题，代表为 Benamou-Brenier 算法

[[benamou-brenier-algorithm|Benamou-Brenier 算法]] 将 OT 转化为 $(d+1)$ 维时空上的凸变分问题，通过变量替换 $E = \mathbf{v}\varrho$ 使约束线性化，再以增广拉格朗日方法迭代求解[^src-benamou-brenier-blog]。

在路径空间层面，[[schrodinger-bridge|Schrödinger bridge (SB)]] 将 OT 推广为 **entropy-regularized optimal transport in path space**：最小化相对于 reference stochastic process 的 KL divergence，以 entropy 参数控制确定性与随机性之间的权衡[^src-schrodinger-bridges-generative-modeling]。经典 OT（$\varepsilon \to 0$）和纯扩散（$\varepsilon \to \infty$）是 SB 的两个极限。[[entropic-optimal-transport]] 给出了 static SB 的完整形式化，[[sinkhorn-algorithm|Sinkhorn/IPF 算法]] 是其核心求解器。

## 在生成模型与时序预测中的应用

### Flow Matching 与 Stochastic Interpolants

Flow Matching 使用 OT 位移插值作为概率路径：

$$
p_t(x \mid x_1) = \mathcal{N}(x \mid t x_1, (1 - (1-\sigma_{\min})t)^2 I)
$$

对应的向量场（OT 条件向量场）：

$$
u_t(x \mid x_1) = \frac{x_1 - (1 - \sigma_{\min}) x}{1 - (1 - \sigma_{\min})t}
$$

[[stochastic-interpolant|Stochastic Interpolants]] 进一步证明：固定任意插值路径时，概率流速度是二次目标 $G(\hat v)$ 的极小点；若再对插值做 $\max_I \min_{\hat v} G$，在可插值密度假设下恢复 [[benamou-brenier-algorithm|Benamou–Brenier]] 动态 OT 解，从而把仿真无关速度学习与动态最优传输显式桥接[^src-stochasticinterpolants]。

### 优势

1. **直线轨迹**：比扩散路径的曲线轨迹更简单
2. **恒定方向**：向量场可分离为 $g(t) \cdot h(x)$，更容易拟合
3. **更快收敛**：实验表明 OT 路径训练更快、采样更高效

### 时序预测：DistDF 的离散 Kantorovich 形式

[[distdf|DistDF]] 把 Wasserstein 距离从生成模型搬到预测模型的训练目标上，直接在离散概率测度上使用 Kantorovich 形式（论文 Definition 3.2 / 附录 B）。给定随机变量 $\alpha,\beta$ 的经验样本 $S_\alpha=[\alpha_1,\dots,\alpha_n]$、$S_\beta=[\beta_1,\dots,\beta_m]$，传输代价 $d_{i,j}=\|\alpha_i-\beta_j\|_p^p$，则

$$
W_p(\mathbb{P}_\alpha,\mathbb{P}_\beta) := \min_{P\in\Pi(\mathbb{P}_\alpha,\mathbb{P}_\beta)} \langle D,P\rangle,
$$

其中 $\Pi$ 要求行和为 $a_i$、列和为 $b_j$ 且 $p_{i,j}\ge 0$。直接对齐条件分布 $W_p(\mathbb{P}_{y\mid x},\mathbb{P}_{\hat y\mid x})$ 不可行：固定 $x$ 时通常只有一个观测 $y$ 和一个预测 $\hat y$，两边的经验集各只有一个样本。DistDF 改用联合分布差异 $W_p(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat y})$，它由 [[joint-distribution-wasserstein-alignment|联合分布 Wasserstein 对齐]]给出期望条件差异的上界：

$$
\int W_p(\mathbb{P}_{y\mid x},\mathbb{P}_{\hat y\mid x})\,\mathrm{d}\mathbb{P}(x) \le W_p(\mathbb{P}_{x,y},\mathbb{P}_{x,\hat y}),
$$

等式在 $p=1$ 或条件项对 $x$ 为常数时成立（Lemma 3.3，引自 Kim et al. 2022）。当联合分布取高斯假设时，平方 $W_2$ 化为闭式的 [[bures-wasserstein|Bures–Wasserstein]] 度量 $BW=\|\mu_{x,y}-\mu_{x,\hat y}\|_2^2+B(\Sigma_{x,y},\Sigma_{x,\hat y})$（Lemma 3.5），因此可以在 batch 上对拼接序列 $[X,Y]$ 与 $[X,\hat Y]$ 的均值、协方差求梯度，并与 MSE 混合成 $L=\gamma\,BW+(1-\gamma)L_{\mathrm{MSE}}$。[^src-distdf] 这里的角色与生成模型不同：OT 不是用来在噪声与数据之间搬运样本，而是把预测与标签的联合分布拉近，作为多步监督目标使用。

## 相关页面

- [[flow-matching]] — 流匹配框架
- [[continuous-normalizing-flow]] — 连续归一化流
- [[diffusion-model]] — 扩散模型
- [[instaflow]] — InstaFlow，reflow 通过降低凸传输代价改善噪声-图像配对 (ICLR 2024)
- [[rectified-flow|Rectified Flow]] — 通过独立耦合收敛到 OT 映射的直线 ODE 生成模型
- [[benamou-brenier-algorithm]] — Benamou-Brenier 算法，OT 的动态公式化
- [[schrodinger-bridge]] — Schrödinger bridge，路径空间中的熵正则最优传输
- [[entropic-optimal-transport]] — 熵正则最优传输的完整形式化
- [[sinkhorn-algorithm]] — Sinkhorn/IPF 求解静态 SB 的迭代算法
- [[stochastic-interpolant]] — 随机插值与二次速度目标
- [[interflow]] — InterFlow 生成模型
- [[distdf]] — DistDF，把联合分布 Wasserstein 差异作为时序预测的训练目标
- [[joint-distribution-wasserstein-alignment]] — DistDF 的联合分布 Wasserstein 对齐技术
- [[source-distdf]] — 论文 Hao Wang et al., arXiv:2510.24574, ICLR 2026
- [[bures-wasserstein]] — closed-form Gaussian W₂ metric

## 引用

[^src-flow-matching]: [[source-flow-matching]]
[^src-benamou-brenier-blog]: [[source-benamou-brenier-blog]]
[^src-schrodinger-bridges-generative-modeling]: [[source-schrodinger-bridges-generative-modeling]]
[^src-stochasticinterpolants]: [[source-stochasticinterpolants]]
[^src-distdf]: [[source-distdf]]