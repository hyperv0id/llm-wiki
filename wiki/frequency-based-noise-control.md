---
title: "频域噪声控制"
type: concept
tags:
  - diffusion-models
  - frequency-domain
  - inductive-bias
  - noise-schedule
created: 2026-05-09
last_updated: 2026-05-09
source_count: 2
confidence: medium
status: active
---

# 频域噪声控制（Frequency-Based Noise Control）

**频域噪声控制**是一种在扩散模型前向过程中通过操控噪声的频谱分布来显式设置模型归纳偏置的技术。由 Jiralerspong 等人（Mila, 2025）提出。[^src-2502-10236]

---

## 核心原理

### 标准 DPM 前向过程

扩散模型的前向过程逐步向数据添加高斯噪声：

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t;\ \sqrt{\alpha_t}\,\mathbf{x}_{t-1},\ (1-\alpha_t)\mathbf{I}\right)$$

一步采样形式：

$$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

标准实现使用白噪声 $\boldsymbol{\epsilon}$，对各频率成分**均匀**施加噪声。[^src-2502-10236]

### 频域噪声控制的核心洞见

> **前向加噪过程中被抹除的信息 ≈ 去噪模型有压力去学习的信息**

因此，通过选择性增强/抑制/跳过特定频段的噪声，可以引导模型学习数据分布的特定方面。[^src-2502-10236]

### 频域塑形流程

将白噪声 $\boldsymbol{\epsilon}$ 替换为频域塑形噪声 $\boldsymbol{\epsilon}^{(w)}$：

$$\boldsymbol{\epsilon} \xrightarrow{\mathcal{F}} \mathbf{N}_{\text{freq}} \xrightarrow{w(\mathbf{f})} \mathbf{N}_{\text{freq}}^{(w)} \xrightarrow{\mathcal{F}^{-1}} \boldsymbol{\epsilon}^{(w)}$$

其中各步骤的数学形式为：[^src-2502-10236]

**复高斯随机场：**

$$\mathbf{N}_{\text{freq}} = \mathbf{N}_{\text{real}} + i\mathbf{N}_{\text{imag}},\quad \mathbf{N}_{\text{real}}, \mathbf{N}_{\text{imag}} \sim \mathcal{N}(0, \mathbf{I})$$

**频域加权（逐元素乘法）：**

$$\mathbf{N}_{\text{freq}}^{(w)}(\mathbf{f}) = \mathbf{N}_{\text{freq}}(\mathbf{f}) \odot w(\mathbf{f})$$

**逆变换回空域：**

$$\boldsymbol{\epsilon}^{(w)} = \Re\left(\mathcal{F}^{-1}\left(\mathbf{N}_{\text{freq}}^{(w)}\right)\right)$$

### 高斯性保持

高斯分布的傅里叶变换仍是高斯，频域加权是线性操作，因此 $\boldsymbol{\epsilon}^{(w)}$ 仍满足高斯假设。**无需修改扩散框架的其他部分**（训练目标、网络架构、采样过程均不变）。[^src-2502-10236]

---

## 频域加权函数族

### 幂律加权

$$w(\mathbf{f}) = \|\mathbf{f}\|^\alpha$$

- $\alpha > 0$：放大高频，噪声更粗糙，模型被迫学习高频纹理/边缘
- $\alpha < 0$：放大低频，噪声更平滑，模型被迫学习低频形状/结构
- $\alpha = 0$：退化为白噪声

### 指数衰减加权

$$w(\mathbf{f}) = \exp\left(-\beta\|\mathbf{f}\|^2\right),\quad \beta > 0$$

- $\beta \to 0$：保留所有频率
- $\beta$ 增大：高频被指数抑制，噪声变得平滑（相邻像素相关）

### 带通掩码

$$w(\mathbf{f}) = \mathbf{M}_{[a,b]}(f_x, f_y) = \begin{cases} 1, & a \leq d(f_x, f_y) \leq b \\ 0, & \text{otherwise} \end{cases}$$

其中径向距离 $d(f_x, f_y) = \sqrt{(f_x - 1/2)^2 + (f_y - 1/2)^2}$。

### 两频段混合（实验使用）

$$\boldsymbol{\epsilon}_f = \gamma_l\,\boldsymbol{\epsilon}_{[a_l,b_l]} + \gamma_h\,\boldsymbol{\epsilon}_{[a_h,b_h]}$$

标准白噪声是此公式的特例（$\gamma_l = \gamma_h = 0.5$）。[^src-2502-10236]

---

## 三种控制模式

| 控制模式 | 数学实现 | 对模型的影响 |
|---------|---------|-------------|
| **强调**某频段 | $\gamma_l \gg \gamma_h$ 或 $\gamma_h \gg \gamma_l$ | 该频段信息被更多破坏→模型被迫更强地学习该频段 |
| **抑制**某频段 | $\gamma_l \ll \gamma_h$ 或 $\gamma_h \ll \gamma_l$ | 该频段信息被较少破坏→模型较少关注该频段 |
| **跳过**某频段 | $w(\mathbf{f}) = 0$ 对特定 $\mathbf{f}$ | 该频段信息在前向过程中保持完好→模型无压力学习该频段（选择性忽略） |

---

## 与相关概念的关系

### 频域原理 / 谱偏置（F-Principle / Spectral Bias）

神经网络天然倾向先学习低频成分（Xu et al., 2019; Rahaman et al., 2019）。频域噪声控制可以：
- **增强**这种自然偏置：使用偏低频的噪声调度，让模型更聚焦低频
- **对抗**这种自然偏置：使用偏高频的噪声调度，迫使模型学习高频细节

### EDM 设计空间

EDM 调整噪声量级 $\sigma(t)$ 和信号缩放 $s(t)$，频域噪声控制调整噪声的**频谱形状**。两者**正交可组合**——可在 EDM 框架内对每个噪声水平 $\sigma$ 使用不同频域加权 $w_\sigma(\mathbf{f})$，实现噪声量级与频谱形状的独立调控。[^src-2502-10236]

### 其他频域方法

- **[[frequency-enhanced-attention\|FEA]] / [[frequency-enhanced-block\|FEB]]**（FEDformer）：修改注意力计算，频域噪声控制修改数据层面的噪声分布——互不干扰
- **[[adaptive-frequency-modulation\|AFM]]**（UniExtreme）：在模型内部分类频段，频域噪声控制在前向过程操控噪声——互补
- **冷扩散（Cold Diffusion）**：使用非高斯退化，频域噪声控制保持高斯假设——不同方向

---

## 实验证据

### 频域匹配实验

5 个数据集（MNIST, CIFAR-10, DomainNet-Quickdraw, WikiArt, CelebA）中 3 个显著受益于频域控制。FID 趋势近乎单调，验证了频域操控与数据信息分布的对应关系。[^src-2502-10236]

### 选择性学习实验

对 MNIST 在特定频段 $[a_c, b_c]$ 添加破坏噪声 $\mathbf{x}' = \mathbf{x} + \gamma_c\boldsymbol{\epsilon}_{[a_c,b_c]}$。训练时设置 $b_l = a_c, a_h = b_c$ 使前向噪声跳过破坏频段。在 8 个非重叠频段测试中全面超越标准扩散。[^src-2502-10236]

---

## 局限性

- 最优频域调度依赖于数据集特性，无通用公式——需经验探索
- 空域内容与频域感知之间的映射并非总是直观
- 仅验证了简单的两频段带通滤波器，更复杂的加权方案（幂律、指数衰减、动态时变）尚未实验验证
- 动态时变频域调度（如低频→高频渐进式变化）是未来方向
- 仅在图像数据上验证，未扩展到时间序列、3D 等其他拓扑结构数据

## 链接

- [[frequency-diffusion]] — 频域扩散（具体技术实现）
- [[two-band-mixture-noise]] — 两频段混合噪声参数化
- [[diffusion-model]] — 扩散模型基础
- [[edm-design-space]] — EDM 设计空间
- [[inductive-bias-shaping]] — 归纳偏置塑造
- [[freqflow]] — FreqFlow，另一种频率感知方法，通过双分支架构在流匹配中显式建模频率（2026）
- [[frequency-aware-conditioning]] — 频率感知条件化概念

## 引用

[^src-2502-10236]: [[source-2502-10236]]