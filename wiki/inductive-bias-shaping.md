---
title: "归纳偏置塑造"
type: concept
tags:
  - deep-learning
  - inductive-bias
  - diffusion-models
  - frequency-domain
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: medium
status: active
---

# 归纳偏置塑造（Inductive Bias Shaping）

**归纳偏置塑造**是指通过显式设计训练过程中的某些组件来引导模型学习数据分布的特定方面，而非依赖模型架构的隐式偏置。在扩散模型中，Jiralerspong 等人（2025）提出通过操控前向加噪算子（[[frequency-based-noise-control|频域噪声控制]]）来实现这一目标。[^src-2502-10236]

---

## 背景与动机

### 深度学习中的隐式归纳偏置

深度学习模型天然具有多种隐式归纳偏置：[^src-2502-10236]

**1. 频域原理（F-Principle / Spectral Bias）**

神经网络在训练过程中倾向先学习数据的低频成分，再学习高频成分（Xu et al., 2019; Rahaman et al., 2019）。这意味着：
- 低频结构（形状、轮廓）被优先捕获
- 高频细节（纹理、边缘）需要更多训练步数
- 模型可能对低频信息过拟合，忽略高频判别特征

**2. 简约偏置 / 捷径学习（Simplicity Bias / Shortcut Learning）**

模型倾向使用简单、易学、但可能是虚假相关的特征进行预测（Geirhos et al., 2020; Scimeca et al., 2021）。例如：
- 在 ImageNet 训练中，CNN 倾向使用纹理而非形状进行分类
- 当训练数据中存在虚假相关性时，模型会优先利用这些捷径

**3. 几何自适应调和表示**

Kadkhodaie 等人（2023）发现扩散模型中的去噪网络隐式学到几何自适应的调和表示（geometry-adaptive harmonic representations），这有助于模型在训练数据之外泛化。

**4. 奖励优化中的主次偏置**

Zhang 等人（2024）发现扩散模型在奖励优化中存在主次偏置（primacy bias），导致过度优化问题。

### 隐式偏置的问题

如果不被显式控制，这些偏置可能导致模型学习数据分布的"错误"方面，产生有偏的生成结果。[^src-2502-10236]

---

## 扩散模型中的归纳偏置塑造

### 核心假设

Jiralerspong 等人的核心假设是：[^src-2502-10236]

> **前向加噪过程中被抹除的信息 ≈ 去噪模型有压力去学习的信息**

形式化地，标准 DPM 的前向过程为：

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

训练目标为预测噪声 $\boldsymbol{\epsilon}$：

$$L = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

去噪模型 $\boldsymbol{\epsilon}_\theta$ 必须学会从 $\mathbf{x}_t$ 中恢复被噪声 $\boldsymbol{\epsilon}$ 抹除的信息。因此，**$\boldsymbol{\epsilon}$ 的统计特性决定了模型需要学习的内容**。[^src-2502-10236]

### 三种塑造策略

**策略 1：强调特定频段**

使用偏重低频的噪声调度（$\gamma_l \gg \gamma_h$）：

$$\boldsymbol{\epsilon}_f = 0.9\,\boldsymbol{\epsilon}_{[0,0.5]} + 0.1\,\boldsymbol{\epsilon}_{[0.5,1]}$$

→ 低频信息被更多破坏 → 模型被迫更强地学习低频结构

**策略 2：抑制特定频段**

使用偏重高频的噪声调度（$\gamma_h \gg \gamma_l$）：

$$\boldsymbol{\epsilon}_f = 0.1\,\boldsymbol{\epsilon}_{[0,0.5]} + 0.9\,\boldsymbol{\epsilon}_{[0.5,1]}$$

→ 高频信息被更多破坏 → 模型被迫更强地学习高频细节

**策略 3：跳过特定频段**

使用带通掩码跳过某频段：

$$w(\mathbf{f}) = 0,\quad \forall \mathbf{f} \in [a_c, b_c]$$

→ 该频段信息在前向过程中保持完好 → 模型无压力学习该频段

### 选择性学习的数学形式

当数据被频段噪声破坏时：

$$\mathbf{x}' = \mathbf{x} + \gamma_c\,\boldsymbol{\epsilon}_{[a_c, b_c]}$$

训练时设置 $b_l = a_c, a_h = b_c$，使前向噪声跳过 $[a_c, b_c]$ 频段。这样破坏噪声所在频段在前向过程中不被额外抹除，模型无需学习该频段，从而恢复原始分布。[^src-2502-10236]

---

## 其他归纳偏置塑造方法对比

| 方法 | 操控对象 | 数学形式 | 与频域噪声控制的关系 |
|------|---------|---------|---------------------|
| 噪声调度调整 (Sahoo et al., 2024) | 噪声量级 $\sigma(t)$ | $\sigma(t)$ 可学习 | **正交**——量级 vs 频谱形状 |
| 非高斯噪声 (Bansal et al., 2022) | 噪声分布类型 | $\boldsymbol{\epsilon} \sim \text{非高斯}$ | **不同方向**——分布类型 vs 频谱形状 |
| 奖励优化对齐 (Zhang et al., 2024) | 模型输出偏好 | 事后微调 | **补充**——事后调整 vs 训练时塑造 |
| 频域噪声控制（本方法） | 噪声频谱形状 | $w(\mathbf{f})$ 加权 | — |

---

## 更广泛的意义

归纳偏置塑造的思想不局限于扩散模型。核心洞见——**训练过程中信息抹除的模式决定了模型学习的内容**——可以推广到任何涉及前向加噪/信息破坏的生成模型框架：[^src-2502-10236]

- **[[flow-matching|Flow Matching]]**：概率路径的选择决定了模型学习的变换方向
- **VAE**：编码器引入的噪声分布影响潜变量的语义结构
- **Masked Autoencoder**：掩码模式决定了模型需要重建的信息
- **Denoising Autoencoder**：噪声类型影响学到的特征表示

## 链接

- [[frequency-based-noise-control]] — 频域噪声控制（具体实现）
- [[frequency-diffusion]] — 频域扩散（具体技术）
- [[diffusion-model]] — 扩散模型
- [[edm-design-space]] — EDM 设计空间
- [[spurious-patterns]] — 虚假模式/捷径学习

## 引用

[^src-2502-10236]: [[source-2502-10236]]