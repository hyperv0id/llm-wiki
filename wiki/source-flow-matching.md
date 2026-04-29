---
title: "Flow Matching for Generative Modeling"
type: source-summary
tags:
  - flow-matching
  - continuous-normalizing-flows
  - cnf
  - optimal-transport
  - meta-ai
  - neurips-2023
created: 2026-04-28
last_updated: 2026-04-28
source_count: 0
confidence: medium
status: active
---

# Flow Matching

**Flow Matching** 是由 Meta AI 的 Yaron Lipman、Ricky T. Q. Chen 等人于 2022 年发表的论文（arXiv:2210.02747），提出了一种无需模拟的训练连续归一化流（CNF）的新范式[^src-flow-matching]。

## 核心贡献

### 1. Flow Matching 目标函数

直接回归目标概率路径的向量场：

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1], x \sim p_t} \| v_t(x; \theta) - u_t(x) \|^2
$$

其中 $u_t(x)$ 是生成目标概率路径 $p_t(x)$ 的向量场。

### 2. 条件概率路径构造

通过条件概率路径的混合构造边缘概率路径：

$$
p_t(x) = \int p_t(x \mid x_1) q(x_1) \, dx_1
$$

$$
u_t(x) = \int u_t(x \mid x_1) \frac{p_t(x \mid x_1) q(x_1)}{p_t(x)} \, dx_1
$$

### 3. 条件流匹配 (CFM)

由于边缘概率路径难以计算，引入条件流匹配目标：

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_1 \sim q, x \sim p_t(\cdot \mid x_1)} \| v_t(x) - u_t(x \mid x_1) \|^2
$$

**定理 2**：$\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$

### 4. 高斯条件概率路径

设 $p_t(x \mid x_1) = \mathcal{N}(x \mid \mu_t(x_1), \sigma_t(x_1)^2 I)$，边界条件：
- $\mu_0(x_1) = 0, \sigma_0(x_1) = 1$（标准高斯噪声）
- $\mu_1(x_1) = x_1, \sigma_1(x_1) = \sigma_{\min}$（集中在数据点）

对应的流映射：$\psi_t(x) = \sigma_t(x_1) x + \mu_t(x_1)$

**定理 3**：生成该高斯路径的唯一向量场为：

$$
u_t(x \mid x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1)) + \mu_t'(x_1)
$$

### 5. 两种特殊路径

#### 扩散路径 (Diffusion)

方差保持 (VP) 路径：
$$
\mu_t(x_1) = \alpha_{1-t} x_1, \quad \sigma_t(x_1) = \sqrt{1 - \alpha_{1-t}^2}
$$

$$
u_t(x \mid x_1) = \frac{\alpha'_{1-t}}{\sqrt{1 - \alpha_{1-t}^2}} (\alpha_{1-t} x - x_1)
$$

#### 最优传输路径 (OT)

线性插值：
$$
\mu_t(x_1) = t x_1, \quad \sigma_t(x_1) = 1 - (1 - \sigma_{\min}) t
$$

$$
u_t(x \mid x_1) = \frac{x_1 - (1 - \sigma_{\min}) x}{1 - (1 - \sigma_{\min}) t}
$$

OT 路径的优势：粒子沿直线运动，方向恒定，训练更快、采样更高效。

## 实验结果

| 数据集 | 方法 | NLL (bits/dim) ↓ | FID ↓ | NFE ↓ |
|--------|------|------------------|-------|-------|
| CIFAR-10 | DDPM | 3.12 | 7.48 | 274 |
| CIFAR-10 | FM w/ Diffusion | 3.10 | 8.06 | 183 |
| CIFAR-10 | **FM w/ OT** | **2.99** | **6.35** | **142** |
| ImageNet 64×64 | DDPM | 3.32 | 17.36 | 264 |
| ImageNet 64×64 | **FM w/ OT** | **3.31** | **14.45** | **138** |

## 意义

Flow Matching 统一了扩散模型和连续归一化流，提供了更灵活的概率路径选择。OT 路径在训练速度、采样效率和生成质量上均优于传统扩散路径。

## 引用

[^src-flow-matching]: Lipman et al. "Flow Matching for Generative Modeling". NeurIPS 2023. arXiv:2210.02747