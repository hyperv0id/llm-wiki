---
title: "SAGD: Learning What Matters — Steering Diffusion via Spectrally Anisotropic Forward Noise"
type: source-summary
tags:
  - diffusion-models
  - inductive-bias
  - frequency-domain
  - anisotropic-noise
  - sagd
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: high
status: active
---

# SAGD: Learning What Matters — Steering Diffusion via Spectrally Anisotropic Forward Noise

**Luca Scimeca, Thomas Jiralerspong, Berton Earnshaw, Jason Hartford, Yoshua Bengio** (Mila — Quebec AI Institute / Valence Labs)，arXiv:2510.09660v4，2025 年 12 月。[^src-sagd]

---

## 核心贡献

本文提出 **Spectrally Anisotropic Gaussian Diffusion (SAGD)**，将标准扩散模型中的各向同性前向噪声替换为**频域对角协方差**的结构化各向异性高斯噪声，从而系统性地塑造扩散模型的归纳偏置。这是同一团队 ICLR 2025 Workshop 论文（arXiv:2502.10236）的完整版，在理论深度和实验规模上均有实质性提升。[^src-sagd]

### 与 Workshop 版的关键区别

| 维度 | Workshop 版 (2502.10236) | SAGD (2510.09660) |
|------|------------------------|-------------------|
| 理论框架 | 经验性频域噪声塑形 | 各向异性高斯协方差 $\Sigma_w$ 的正式理论 |
| Score 收敛性 | 未证明 | 证明 $\Sigma_w \succ 0$ 时 $t \to 0$ score 收敛到真实数据 score |
| Score-$\epsilon$ 关系 | 标准形式 | 推导各向异性下的通用 score-$\epsilon$ 关系 $\nabla_{x_t} \log q_{w,t} = -\frac{1}{\sigma_t}\Sigma_w^{-1}\epsilon_\theta$ |
| 噪声算子 | 三种加权函数（幂律/指数/带通） | 两种具体算子：**plw-SAGD**（幂律加权）和 **bpm-SAGD**（带通掩码+两频段混合） |
| 数据集 | 5 个（MNIST/CIFAR-10/Quickdraw/WikiArt/CelebA） | 6 个（新增 **FFHQ** 和 **ImageNet-1k**） |
| 骨干网络 | DDPM fast sampling + U-Net | U-Net + **DiT** latent diffusion（DINOv2 潜空间） |
| 最大实验规模 | 32×32 | **256×256 ImageNet-1k**，196M 参数 DiT |
| 选择性忽略 | 实验展示 | 理论形式化：rank-deficient $\Sigma_w$ 下的 projected score |

---

## 理论贡献

### 1. 各向异性协方差的形式化

定义线性算子 $T_w = \mathcal{F}^{-1} \text{Diag}(w) \mathcal{F}$，前向噪声 $\epsilon^{(w)} = T_w \xi$ 服从：

$$\epsilon^{(w)} \sim \mathcal{N}(0, \Sigma_w), \quad \Sigma_w = T_w T_w^\top = \mathcal{F}^{-1} \text{Diag}(|w(f)|^2) \mathcal{F}$$

前向边缘分布：

$$q_w(x_t | x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\, x_0,\ \sigma_t^2 \Sigma_w\right)$$[^src-sagd]

### 2. Score-$\epsilon$ 关系（各向异性推广）

$$\nabla_{x_t} \log q_{w,t}(x_t) = -\frac{1}{\sigma_t} \Sigma_w^{-1}\, \mathbb{E}[\epsilon^{(w)} | x_t]$$

当 $\Sigma_w = I$ 时退化为标准 DDPM score 关系。**$\ell_2$ 训练目标不变**（最优预测仍为条件均值），$\Sigma_w^{-1}$ 仅在 score 转换时出现。[^src-sagd]

### 3. Score 收敛性（Theorem）

当 $\Sigma_w \succ 0$（满秩，即 $w(f) > 0$ 对所有 $f$）且数据分布 $q$ 有局部正 $C^1$ 密度时：

$$\lim_{t \to 0} \nabla_{x_t} \log q_{w,t}(x_t) = \nabla_x \log q(x) \quad \text{a.e.}$$

**意义**：塑形前向协方差改变的是概率流路径（path），而非终点（endpoint）score。[^src-sagd]

### 4. 选择性忽略的理论

当 $w$ 在某频段为零时 $\Sigma_w$ 奇异，模型学到的是 **projected score** $\Pi \nabla_x \log q(x)$，其中 $\Pi$ 为正交投影到 $\text{range}(\Sigma_w)$。这解释了选择性恢复的理论基础。[^src-sagd]

---

## 两种噪声算子

### plw-SAGD（幂律加权）

$$w_\alpha(f) = (r(f) + \varepsilon)^\alpha,\quad r(f) = \sqrt{f_x^2 + f_y^2}$$

- $\alpha > 0$：放大高频
- $\alpha < 0$：放大低频
- $\alpha = 0$：白噪声

RAPSD 满足 $\log \text{PSD}(r) \approx (2\alpha) \log r + \text{const}$。[^src-sagd]

### bpm-SAGD（带通掩码 + 两频段混合）

$$\epsilon^{(w)} = \gamma_l \epsilon_{[a_l,b_l]} + \gamma_h \epsilon_{[a_h,b_h]}, \quad \gamma_l + \gamma_h = 1$$

选择性忽略：设 $b_l < a_h$ 使 $[b_l, a_h]$ 频段在协方差中无支撑。[^src-sagd]

---

## 实验结果

### 频域受限信息学习（CIFAR-10）

CIFAR-10 仅保留高频信息后训练，SAGD 随 $\alpha$ 增大 FID 近乎单调下降，$\alpha=0.08$ 相较标准 DPM 基线 FID 降低约 **0.3**。[^src-sagd]

### 自然数据集（plw-SAGD）

| 数据集 | 最优 $\alpha$ | 基线 FID | SAGD FID | 改善 |
|--------|-------------|----------|----------|------|
| MNIST | -0.06 | 0.42 | **0.28** | ↓33% |
| DomainNet-Quickdraw | -0.04 | 0.60 | **0.49** | ↓18% |
| WikiArt | -0.001 | 1.06 | **1.02** | ↓4% |
| FFHQ | -0.001 | 1.11 | **1.04** | ↓6% |
| **ImageNet-1k** | **-0.04** | **8.68** | **7.55** | **↓13%** |

**5/6 数据集超越基线**。CIFAR-10 最优 $\alpha \approx 0$（持平）。[^src-sagd]

### ImageNet-1k DiT 实验（亮点）

256×256 分辨率，RAE DiT backbone in DINOv2 latent space（196M 参数）：
- 基线 ($\alpha=0$)：FID 8.68 ± 0.07
- SAGD ($\alpha=-0.04$)：FID **7.55 ± 0.06**
- 绝对改善 ≈ 1.1 FID（约 13% 相对提升）

证明 SAGD 在大规模高分辨率设定下同样有效。[^src-sagd]

### 选择性忽略（bpm-SAGD）

MNIST  频段破坏实验，8 个非重叠频段全部超越标准扩散，趋势与 workshop 版一致（高频破坏时修复更好，印证 MNIST 信息集中于低频）。[^src-sagd]

---

## 关键性质

1. **Gaussian 性保持**：$\epsilon^{(w)}$ 仍是高斯噪声（仅协方差不再是标量 $I$）
2. **Drop-in 兼容**：只改 forward noise 生成，训练目标/采样过程/网络架构不变
3. **Score 一致性**：满秩时 $t \to 0$ score 收敛到真实 score
4. **路径塑形而非终点塑形**：各向异性确定性改变 probability-flow ODE 轨迹
5. **与 EDM 正交**：SAGD 改频谱形状，EDM 改噪声量级——可组合

---

## 局限性

- 最优 $\alpha$ 依赖数据集特性，需经验搜索
- 空域-频域的感知映射不总是直观
- 时变频域策略（$\Sigma_{w_t}$）作为未来方向
- 仅图像数据验证，但理论上适用于任何有谱基的拓扑结构（1D/2D/3D grid, graph Laplacian）

## 引用

[^src-sagd]: [[source-sagd]]