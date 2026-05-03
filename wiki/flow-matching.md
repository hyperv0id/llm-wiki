---
title: "Flow Matching"
type: concept
tags:
  - flow-matching
  - continuous-normalizing-flows
  - generative-model
  - optimal-transport
  - meta-ai
  - neurips-2023
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Flow Matching

**Flow Matching**（流匹配）是由 Meta AI 的 Yaron Lipman、Ricky T. Q. Chen 等人于 2022 年提出的生成模型训练框架，发表于 NeurIPS 2023[^src-flow-matching]。它提供了一种无需模拟（simulation-free）的方式来训练连续归一化流（CNF），同时统一了扩散模型和最优传输路径。

## 背景：连续归一化流 (CNF)

### 什么是归一化流？

归一化流通过一系列可逆变换，将简单分布（如标准高斯分布）转换为复杂的数据分布。设 $z \sim p(z)$ 为简单先验，$x = \phi(z)$ 为可逆变换，则 $x$ 的分布为：

$$
p(x) = p(z) \left| \det \frac{\partial z}{\partial x} \right| = p(\phi^{-1}(x)) \left| \det D\phi^{-1}(x) \right|
$$

### 连续版本：CNF

连续归一化流将离散的可逆变换序列推广为连续时间流。定义时间依赖的向量场 $v_t: \mathbb{R}^d \to \mathbb{R}^d$，通过常微分方程（ODE）定义流映射 $\phi_t$：

$$
\frac{d}{dt} \phi_t(x) = v_t(\phi_t(x)), \quad \phi_0(x) = x
$$

流 $\phi_t$ 将初始噪声分布 $p_0$ 推送（push-forward）到目标分布 $p_1$：

$$
p_t = [\phi_t]_* p_0
$$

其中 push-forward 算子定义为：

$$
[\phi_t]_* p_0(x) = p_0(\phi_t^{-1}(x)) \left| \det \frac{\partial \phi_t^{-1}}{\partial x}(x) \right|
$$

### 传统训练的困难

传统 CNF 使用最大似然训练，需要昂贵的 ODE 数值模拟来计算 log-likelihood 的梯度：

$$
\nabla_\theta \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
$$

这导致 CNF 难以扩展到高维数据（如高分辨率图像）。

---

## Flow Matching 核心思想

### 目标概率路径

Flow Matching 的核心是直接指定一个**概率路径** $p_t(x)$，满足：
- $p_0(x) = \mathcal{N}(x \mid 0, I)$ — 标准高斯噪声
- $p_1(x) \approx q(x)$ — 近似数据分布

概率路径是一个随时间 $t \in [0, 1]$ 变化的概率密度函数。

### Flow Matching 目标函数

给定目标概率路径 $p_t(x)$ 和生成它的向量场 $u_t(x)$，Flow Matching 目标为：

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x \sim p_t} \left\| v_t(x; \theta) - u_t(x) \right\|^2
$$

这个目标简单直观：**用神经网络 $v_t$ 回归目标向量场 $u_t$**。

### 问题：$u_t$ 难以获取

然而，我们通常不知道哪个向量场 $u_t$ 能生成我们想要的目标概率路径 $p_t$。直接优化 $\mathcal{L}_{\text{FM}}$ 是不可行的。

---

## 条件构造：从不可行到可行

### 条件概率路径

对于每个数据样本 $x_1 \sim q(x_1)$，定义**条件概率路径** $p_t(x \mid x_1)$，满足：
- $p_0(x \mid x_1) = \mathcal{N}(x \mid 0, I)$ — 噪声分布
- $p_1(x \mid x_1) = \mathcal{N}(x \mid x_1, \sigma_{\min}^2 I)$ — 集中在数据点

### 边缘化

通过对数据分布 $q(x_1)$ 边缘化条件概率路径，得到边缘概率路径：

$$
p_t(x) = \int p_t(x \mid x_1) \, q(x_1) \, dx_1
$$

特别地，$p_1(x) \approx q(x)$，即边缘概率路径在 $t=1$ 时近似数据分布。

### 边缘向量场

类似地，定义边缘向量场：

$$
u_t(x) = \int u_t(x \mid x_1) \frac{p_t(x \mid x_1) q(x_1)}{p_t(x)} \, dx_1
$$

其中 $u_t(x \mid x_1)$ 是生成条件概率路径 $p_t(\cdot \mid x_1)$ 的向量场。

### 定理 1：边缘化正确性

**定理 1**：如果 $u_t(x \mid x_1)$ 生成 $p_t(x \mid x_1)$，则边缘向量场 $u_t(x)$ 生成边缘概率路径 $p_t(x)$。

**证明**：使用连续性方程（continuity equation）。对于任意 $t$：

$$
\frac{\partial}{\partial t} p_t(x) = -\nabla \cdot (u_t(x) p_t(x))
$$

展开右侧：

$$
\begin{aligned}
\nabla \cdot (u_t(x) p_t(x)) &= \nabla \cdot \left( \int u_t(x \mid x_1) \frac{p_t(x \mid x_1) q(x_1)}{p_t(x)} p_t(x) \, dx_1 \right) \\
&= \nabla \cdot \left( \int u_t(x \mid x_1) p_t(x \mid x_1) q(x_1) \, dx_1 \right)
\end{aligned}
$$

而左侧：

$$
\frac{\partial}{\partial t} p_t(x) = \int \frac{\partial}{\partial t} p_t(x \mid x_1) q(x_1) \, dx_1 = \int -\nabla \cdot (u_t(x \mid x_1) p_t(x \mid x_1)) q(x_1) \, dx_1
$$

由于 $u_t(x \mid x_1)$ 生成 $p_t(x \mid x_1)$，两项相等，定理得证。$\square$

---

## 条件流匹配 (CFM)

### 核心���察

虽然边缘向量场 $u_t(x)$ 难以直接计算，但条件向量场 $u_t(x \mid x_1)$ 很容易定义！

### CFM 目标函数

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_1 \sim q,\, x \sim p_t(\cdot \mid x_1)} \left\| v_t(x; \theta) - u_t(x \mid x_1) \right\|^2
$$

### 定理 2：梯度等价

**定理 2**：在 $p_t(x) > 0$ 的假设下，有：

$$
\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)
$$

即优化 CFM 目标等价于优化原始 FM 目标！

**证明思路**：

展开两项：

$$
\begin{aligned}
\mathcal{L}_{\text{FM}} &= \mathbb{E}_{t,x} \|v_t(x) - u_t(x)\|^2 \\
&= \mathbb{E}_{t,x_1,x} \|v_t(x)\|^2 - 2\langle v_t(x), u_t(x) \rangle + \|u_t(x)\|^2
\end{aligned}
$$

$$
\begin{aligned}
\mathcal{L}_{\text{CFM}} &= \mathbb{E}_{t,x_1,x} \|v_t(x) - u_t(x \mid x_1)\|^2 \\
&= \mathbb{E}_{t,x_1,x} \|v_t(x)\|^2 - 2\langle v_t(x), u_t(x \mid x_1) \rangle + \|u_t(x \mid x_1)\|^2
\end{aligned}
$$

关键观察：利用边缘化公式（8），有

$$
\mathbb{E}_{x \sim p_t} [u_t(x)] = \mathbb{E}_{x_1 \sim q, x \sim p_t(\cdot \mid x_1)} [u_t(x \mid x_1)]
$$

因此两项关于 $\theta$ 的梯度相等。$\square$

**重要意义**：CFM 目标可以直接计算和采样，无需知道边缘概率路径或边缘向量场！

---

## 高斯条件概率路径

### 参数化

设条件概率路径为高斯分布：

$$
p_t(x \mid x_1) = \mathcal{N}\left(x \mid \mu_t(x_1), \sigma_t(x_1)^2 I\right)
$$

其中 $\mu_t: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ 是时间依赖的均值函数，$\sigma_t: [0,1] \times \mathbb{R} \to \mathbb{R}_{>0}$ 是时间依赖的标准差函数。

**边界条件**：
- $t = 0$: $\mu_0(x_1) = 0, \sigma_0(x_1) = 1$ → 标准高斯噪声
- $t = 1$: $\mu_1(x_1) = x_1, \sigma_1(x_1) = \sigma_{\min}$ → 集中在数据点

### 流映射

定义对应的流映射（仿射变换）：

$$
\psi_t(x) = \sigma_t(x_1) x + \mu_t(x_1)
$$

当 $x \sim \mathcal{N}(0, I)$ 时，$\psi_t(x) \sim \mathcal{N}(\mu_t(x_1), \sigma_t(x_1)^2 I)$，即：

$$
[\psi_t]_* \mathcal{N}(0, I) = p_t(\cdot \mid x_1)
$$

### 定理 3：生成高斯路径的向量场

**定理 3**：对于上述高斯条件概率路径，生成它的唯一向量场为：

$$
\boxed{u_t(x \mid x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1)) + \mu_t'(x_1)}
$$

其中 $\sigma_t' = \frac{d}{dt}\sigma_t$，$\mu_t' = \frac{d}{dt}\mu_t$。

**推导**：

由流映射的定义，有：

$$
\frac{d}{dt} \psi_t(x) = u_t(\psi_t(x) \mid x_1)
$$

代入 $\psi_t(x) = \sigma_t x + \mu_t$：

$$
\sigma_t' x + \mu_t' = u_t(\sigma_t x + \mu_t \mid x_1)
$$

令 $y = \sigma_t x + \mu_t$（即 $x = \frac{y - \mu_t}{\sigma_t}$），得：

$$
u_t(y \mid x_1) = \sigma_t' \cdot \frac{y - \mu_t}{\sigma_t} + \mu_t' = \frac{\sigma_t'}{\sigma_t}(y - \mu_t) + \mu_t'
$$

$\square$

---

## 两种特殊路径

### 路径 1：扩散路径 (Diffusion Path)

扩散模型中的概率路径可以通过高斯条件路径表示。

#### 方差保持 (VP) 路径

设 $\alpha_t = e^{-\frac{1}{2} \int_0^t \beta(s) ds}$，则：

$$
\mu_t(x_1) = \alpha_{1-t} x_1, \quad \sigma_t(x_1) = \sqrt{1 - \alpha_{1-t}^2}
$$

代入定理 3 的公式：

$$
\begin{aligned}
u_t(x \mid x_1) &= \frac{\frac{d}{dt}\sqrt{1 - \alpha_{1-t}^2}}{\sqrt{1 - \alpha_{1-t}^2}} (x - \alpha_{1-t} x_1) + \frac{d}{dt}(\alpha_{1-t} x_1) \\
&= \frac{-\alpha_{1-t} \alpha'_{1-t}}{\sqrt{1 - \alpha_{1-t}^2} \cdot \sqrt{1 - \alpha_{1-t}^2}} (x - \alpha_{1-t} x_1) - \alpha'_{1-t} x_1 \\
&= \frac{\alpha'_{1-t}}{\sqrt{1 - \alpha_{1-t}^2}} (\alpha_{1-t} x - x_1)
\end{aligned}
$$

其中 $\alpha'_{1-t} = -\frac{1}{2}\beta(1-t)$。

#### 路径特点

- 噪声减少主要发生在 $t \to 1$ 时
- 轨迹呈曲线（见论文图 3）
- 与概率流 ODE（Song et al., 2020）等价

---

### 路径 2：最优传输路径 (OT Path)

#### 线性插值

最简单的选择是线性变化：

$$
\boxed{\mu_t(x_1) = t x_1}, \quad \boxed{\sigma_t(x_1) = 1 - (1 - \sigma_{\min}) t}
$$

代入定理 3：

$$
\begin{aligned}
u_t(x \mid x_1) &= \frac{-(1 - \sigma_{\min})}{1 - (1 - \sigma_{\min})t} (x - t x_1) + x_1 \\
&= \frac{x_1 - (1 - \sigma_{\min}) x}{1 - (1 - \sigma_{\min})t}
\end{aligned}
$$

#### OT 路径的特点

1. **直线轨迹**：粒子从噪声到数据沿直线运动
2. **恒定方向**：向量场方向不随时间变化（$u_t = g(t) \cdot h(x \mid x_1)$）
3. **更简单**：比扩散路径更容易拟合

#### 为什么是最优传输？

对于两个高斯分布 $\mathcal{N}(0, I)$ 和 $\mathcal{N}(x_1, \sigma_{\min}^2 I)$，最优传输映射为：

$$
\psi_t(x) = (1-t)x + t(x_1 + \sigma_{\min} \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
$$

这正是线性插值形式（22）！

---

## 训练与采样

### 训练目标（CFM with OT）

代入 OT 路径的条件向量场：

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_1 \sim q, x_0 \sim \mathcal{N}(0,I)} \left\| v_t(\psi_t(x_0)) - \frac{x_1 - (1 - \sigma_{\min}) x_0}{1 - (1 - \sigma_{\min})t} \right\|^2
$$

其中 $\psi_t(x_0) = (1 - (1 - \sigma_{\min})t)x_0 + t x_1$。

### 采样

1. 采样噪声 $x_0 \sim \mathcal{N}(0, I)$
2. 通过 ODE 求解器从 $t=0$ 积分到 $t=1$：

$$
\frac{d}{dt} x(t) = v_t(x(t); \theta)
$$

3. 得到 $x_1 \approx \text{sample from } q(x)$

---

## 实验结果

| 数据集 | 方法 | NLL (bits/dim) ↓ | FID ↓ | NFE ↓ |
|--------|------|------------------|-------|-------|
| CIFAR-10 | DDPM | 3.12 | 7.48 | 274 |
| CIFAR-10 | FM w/ Diffusion | 3.10 | 8.06 | 183 |
| **CIFAR-10** | **FM w/ OT** | **2.99** | **6.35** | **142** |
| ImageNet 64×64 | DDPM | 3.32 | 17.36 | 264 |
| ImageNet 64×64 | FM w/ Diffusion | 3.33 | 16.88 | 187 |
| **ImageNet 64×64** | **FM w/ OT** | **3.31** | **14.45** | **138** |

### 关键发现

1. **FM w/ OT 全面优于扩散模型**：更好的 likelihood、更好的样本质量、更少的采样步数
2. **训练更快**：FM-OT 收敛更快，所需迭代更少
3. **采样更高效**：OT 路径的直线轨迹使 ODE 求解更稳定

---

## 与其他方法的关系

### vs 扩散模型

- 扩散模型是 Flow Matching 的**特例**（使用扩散路径）
- Flow Matching 提供了更灵活的概率路径选择
- 训练更稳定（直接回归向量场 vs 回归分数）

### vs 连续归一化流

- 传统 CNF 需要 ODE 模拟计算 log-likelihood
- Flow Matching 是**无需模拟**的训练方法
- 可以使用任意概率路径

### vs 最优传输

- OT 路径利用最优传输理论
- 直线轨迹是 OT 解的几何性质
- 边缘流不必是全局最优传输

---

## 相关页面

- [[continuous-normalizing-flow]] — 连续归一化流
- [[normalizing-flow]] — 归一化流基础
- [[diffusion-model]] — 扩散模型
- [[glow]] — Glow 流模型
- [[optimal-transport]] — 最优传输理论
- [[probability-flow-ode]] — ��率流 ODE

## 引用

[^src-flow-matching]: [[source-flow-matching]]
- [[aurora]] — Aurora 使用 Prototype-Guided Flow Matching 进行时间序列概率预测
- [[prototype-guided-flow-matching]] — Aurora 的原型引导流匹配技术
