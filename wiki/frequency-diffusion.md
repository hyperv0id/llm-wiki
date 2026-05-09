---
title: "频域扩散"
type: technique
tags:
  - diffusion-models
  - frequency-domain
  - inductive-bias
  - noise-schedule
  - sagd
created: 2026-05-09
last_updated: 2026-05-09
source_count: 2
confidence: high
status: active
---

# 频域扩散（Frequency Diffusion）

**频域扩散**是一种通过在前向加噪过程中对噪声的频谱进行操控来塑造扩散模型归纳偏置的方法。由 Jiralerspong 等人（Mila, 2025）提出。[^src-2502-10236]

---

## 核心思想

标准 DPM 前向过程使用各向同性白噪声，对各频率成分均匀施加噪声。频域扩散的核心观察是：**前向加噪过程中被抹除的信息，恰好是去噪模型有压力去学习的信息**。因此选择性增强/抑制特定频段的噪声，可以引导模型聚焦于数据分布的特定方面。[^src-2502-10236]

形式化地说，标准前向一步采样为：

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

频域扩散将 $\boldsymbol{\epsilon}$ 替换为 $\boldsymbol{\epsilon}^{(w)}$：

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}^{(w)}$$

---

## 频域塑形流程

完整的噪声频域塑形流程为：[^src-2502-10236]

$$\boldsymbol{\epsilon} \xrightarrow{\mathcal{F}} \mathbf{N}_{\text{freq}} \xrightarrow{w(\mathbf{f})} \mathbf{N}_{\text{freq}}^{(w)} \xrightarrow{\mathcal{F}^{-1}} \boldsymbol{\epsilon}^{(w)}$$

### 步骤 1：复高斯随机场

令 $\mathbf{x} \in \mathbb{R}^{H \times W}$ 为空域图像，$\mathcal{F}$ 为二维傅里叶变换。在频域构造复值随机场：

$$\mathbf{N}_{\text{freq}} = \mathbf{N}_{\text{real}} + i\mathbf{N}_{\text{imag}},\quad \mathbf{N}_{\text{real}}, \mathbf{N}_{\text{imag}} \sim \mathcal{N}(0, \mathbf{I})$$

其中每个像素（频率 bin）独立同分布。[^src-2502-10236]

### 步骤 2：频域加权

令 $\mathbf{f} = (f_x, f_y)$ 表示频域坐标，其中：

$$f_x = \frac{k_x}{W},\quad f_y = \frac{k_y}{H},\quad k_x, k_y \in \mathbb{Z}$$

引入加权函数 $w(\mathbf{f})$ 逐元素缩放各频率分量：

$$\mathbf{N}_{\text{freq}}^{(w)}(\mathbf{f}) = \mathbf{N}_{\text{freq}}(\mathbf{f}) \odot w(\mathbf{f})$$

### 步骤 3：逆变换回空域

$$\boldsymbol{\epsilon}^{(w)} = \Re\left(\mathcal{F}^{-1}\left(\mathbf{N}_{\text{freq}}^{(w)}\right)\right)$$

其中 $\Re(\cdot)$ 取实部，确保输出纯实数噪声。[^src-2502-10236]

### 高斯性保持

高斯分布经傅里叶变换后仍是高斯。频域加权是线性操作，因此 $\boldsymbol{\epsilon}^{(w)}$ 仍然是高斯噪声（只是协方差矩阵不再是标量 $\mathbf{I}$）。标准白噪声是此框架的特例（$w(\mathbf{f}) = 1$）。[^src-2502-10236]

---

## 频域加权函数详解

### 1. 幂律加权（Power-Law Weighting）

$$w(\mathbf{f}) = \|\mathbf{f}\|^\alpha$$

- $\alpha > 0$：放大高频，噪声更"粗糙"，模型被迫学习高频纹理/边缘
- $\alpha < 0$：放大低频，噪声更"平滑"，模型被迫学习低频形状/结构
- $\alpha = 0$：退化为白噪声

灵感来源于自然图像能量谱通常遵循幂律分布。[^src-2502-10236]

### 2. 指数衰减加权（Exponential Decay Weighting）

$$w(\mathbf{f}) = \exp\left(-\beta \|\mathbf{f}\|^2\right),\quad \beta > 0$$

- $\beta \to 0$：保留所有频率（近似白噪声）
- $\beta$ 增大：高频被指数抑制，产生类似低通滤波的效果
- 实际效果：施加空间相关性，$\beta$ 大时噪声变得平滑（相邻像素相关）

### 3. 带通掩码与两频段混合（Band-Pass Mask & Two-Band Mixture）

二进制掩码：

$$w(\mathbf{f}) = \mathbf{M}_{[a,b]}(f_x, f_y) = \begin{cases} 1, & a \leq d(f_x, f_y) \leq b \\ 0, & \text{otherwise} \end{cases}$$

其中径向距离：

$$d(f_x, f_y) = \sqrt{\left(f_x - \frac{1}{2}\right)^2 + \left(f_y - \frac{1}{2}\right)^2}$$

实验使用的两频段混合形式：

$$\boldsymbol{\epsilon}_f = \gamma_l\,\boldsymbol{\epsilon}_{[a_l,b_l]} + \gamma_h\,\boldsymbol{\epsilon}_{[a_h,b_h]}$$

其中 $\boldsymbol{\epsilon}_{[a,b]}$ 是经过 $[a,b]$ 带通滤波后的噪声。[^src-2502-10236]

### 标准白噪声作为特例

当参数取以下值时，两频段混合退化为标准白噪声：

$$\gamma_l = \gamma_h = 0.5,\quad a_l = 0,\ b_l = 0.5,\ a_h = 0.5,\ b_h = 1$$

---

## SAGD：各向异性高斯扩散（完整版理论框架）

SAGD（Spectrally Anisotropic Gaussian Diffusion）是同一团队（Scimeca, Jiralerspong, Bengio 等，arXiv:2510.09660, 2025）提出的完整版，将频域噪声控制形式化为**各向异性高斯前向协方差**框架。[^src-sagd]

### 协方差形式化

定义线性算子 $T_w = \mathcal{F}^{-1} \text{Diag}(w) \mathcal{F}$，前向噪声 $\epsilon^{(w)} = T_w \xi$ 服从：

$$\epsilon^{(w)} \sim \mathcal{N}(0, \Sigma_w), \quad \Sigma_w = T_w T_w^\top = \mathcal{F}^{-1} \text{Diag}(|w(f)|^2) \mathcal{F}$$

前向边缘分布：

$$q_w(x_t | x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\, x_0,\ \sigma_t^2 \Sigma_w\right)$$[^src-sagd]

### Score-$\epsilon$ 关系（各向异性推广）

$$\nabla_{x_t} \log q_{w,t}(x_t) = -\frac{1}{\sigma_t} \Sigma_w^{-1}\, \mathbb{E}[\epsilon^{(w)} | x_t]$$

当 $\Sigma_w = I$ 时退化为标准 DDPM score 关系。**$\ell_2$ 训练目标不变**，$\Sigma_w^{-1}$ 仅在 score 转换时出现。[^src-sagd]

### Score 收敛性定理

当 $\Sigma_w \succ 0$（满秩，即 $w(f) > 0$ 对所有 $f$）且数据分布有局部正 $C^1$ 密度时：

$$\lim_{t \to 0} \nabla_{x_t} \log q_{w,t}(x_t) = \nabla_x \log q(x) \quad \text{a.e.}$$

**意义**：塑形前向协方差改变的是概率流路径（path），而非终点（endpoint）score。[^src-sagd]

### 两种具体算子

**plw-SAGD（幂律加权）**：$w_\alpha(f) = (r(f) + \varepsilon)^\alpha$，$\alpha \in \mathbb{R}$ 控制频谱斜率。

**bpm-SAGD（带通掩码+两频段混合）**：$\epsilon^{(w)} = \gamma_l \epsilon_{[a_l,b_l]} + \gamma_h \epsilon_{[a_h,b_h]}$。

### 选择性忽略的理论

当 $w$ 在某频段为零时 $\Sigma_w$ 奇异，模型学到 **projected score** $\Pi \nabla_x \log q(x)$，$\Pi$ 为正交投影到 $\text{range}(\Sigma_w)$。[^src-sagd]

### ImageNet-1k DiT 实验（SAGD 亮点）

256×256 分辨率，RAE DiT backbone in DINOv2 latent space（196M 参数）：

| 设定 | FID |
|------|-----|
| 基线 ($\alpha=0$) | 8.68 ± 0.07 |
| SAGD ($\alpha=-0.04$) | **7.55 ± 0.06** |

绝对改善 ≈ 1.1 FID（约 13% 相对提升），证明 SAGD 在大规模高分辨率设定下有效。[^src-sagd]

---

## 实验设计

固定 $a_l = 0, b_h = 1, b_l = a_h = 0.5$，扫描 $\gamma_l \in [0.1, 0.2, \dots, 0.9]$ 和 $\gamma_h = 1 - \gamma_l$。每个数据集训练 9 个模型（8 种频域调度 + 标准扩散基线），3 个随机种子取平均。[^src-2502-10236]

---

## 数据特异性

| 数据集 | 最优 $\gamma_l$ | 偏置方向 | 可能的解释 |
|--------|----------------|----------|-----------|
| MNIST | 0.4–0.5 | 平衡略偏高频 | 数字笔画需要清晰边缘 |
| CIFAR-10 | 0.5–0.6 | 平衡 | 物体类别+背景纹理各占一半 |
| DomainNet-Quickdraw | 0.3–0.4 | 偏高频 | 线条草图几乎无低频结构 |
| WikiArt | 0.6–0.7 | 偏低频 | 大色块/构图比纹理更重要 |
| CelebA | 0.3–0.4 | 偏高频 | 面部特征/表情需要高频细节 |

FID 趋势近乎单调，验证了频域操控与数据信息分布的对应关系。[^src-2502-10236]

---

## 选择性学习机制

### 问题设定

数据在特定频段被噪声破坏：

$$\mathbf{x}' = \mathbf{x} + \gamma_c\,\boldsymbol{\epsilon}_{[a_c, b_c]},\quad \gamma_c = 1$$

标准 DPM 训练无法区分破坏噪声和真实数据，会近似整个被破坏分布。[^src-2502-10236]

### 频域扩散的解法

训练时设置频段参数使前向噪声**跳过**破坏频段：

$$b_l = a_c,\quad a_h = b_c$$

这样：
- 低频噪声 $\boldsymbol{\epsilon}_{[a_l,b_l]}$ 覆盖 $[0, a_c)$
- 高频噪声 $\boldsymbol{\epsilon}_{[a_h,b_h]}$ 覆盖 $(b_c, 1]$
- 破坏频段 $[a_c, b_c]$ 在前向过程中**不被额外噪声影响**

由于去噪模型只需学习前向过程抹除的信息，而破坏频段的信息在前向过程中保持完好，模型无需学习该频段的分布，从而成功恢复原始无噪声分布。[^src-2502-10236]

### 实验结果

在 MNIST 的 8 个非重叠频段测试中，频域扩散的 FID/KID 全面低于标准扩散。高频破坏时修复效果更好（FID 更低），验证了 MNIST 的信息主要集中于低频。[^src-2502-10236]

---

## 与其他频域方法的对比

| 方法 | 修改位置 | 机制 | 与频域扩散的关系 |
|------|---------|------|-----------------|
| [[frequency-enhanced-attention\|FEA]] | 注意力计算 | Fourier/Wavelet 域交叉注意力替代标准注意力 | 正交——频域扩散改前向过程 |
| [[frequency-enhanced-block\|FEB]] | 注意力计算 | 频域自注意力替代 | 正交 |
| [[adaptive-frequency-modulation\|AFM]] | 模型内部 | Beta 分布谱滤波器分频段 | 互补——AFM 模型内，频域扩散数据层 |
| EDM 噪声调度 | 前向过程 | 调整 $\sigma(t)$ 数值 | **正交可组合**——EDM 改量级，频域扩散改频谱形状 |
| 冷扩散（Cold Diffusion） | 前向过程 | 非高斯退化 | 不同方向——退化类型 vs 频谱形状 |

## 链接

- [[diffusion-model]] — 扩散模型基础
- [[edm-design-space]] — EDM 设计空间
- [[frequency-based-noise-control]] — 频域噪声控制概念
- [[inductive-bias-shaping]] — 归纳偏置塑造
- [[two-band-mixture-noise]] — 两频段混合噪声参数化
- [[source-sagd]] — SAGD 完整版论文（各向异性高斯扩散）
- [[spectral-bias-training-dynamics]] — 扩散模型训练谱偏置理论
- [[freqflow]] — FreqFlow，频率感知流匹配
- [[frequency-aware-conditioning]] — 频率感知条件化概念

## 引用

[^src-2502-10236]: [[source-2502-10236]]
[^src-sagd]: [[source-sagd]]