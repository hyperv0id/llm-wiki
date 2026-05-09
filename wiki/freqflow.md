---
title: "FreqFlow — Frequency-Aware Flow Matching"
type: technique
tags:
  - flow-matching
  - frequency-domain
  - dual-branch
  - image-generation
  - frequency-aware
  - arxiv-2026
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: medium
status: active
---

# FreqFlow — Frequency-Aware Flow Matching

**FreqFlow**（Frequency-Aware Flow Matching）是由 Ren 等人（Johns Hopkins University & ByteDance, 2026）提出的频率感知流匹配生成框架。其核心创新在于将显式的频率条件引入 flow matching 过程，通过双分支架构分离低频全局结构与高频细节处理，并使用时变自适应加权机制动态平衡两者的贡献。ImageNet-256 上 FID 1.38，超越 DiT (+0.79) 和 SiT (+0.58)。[^src-freqflow]

---

## 背景与动机

### Flow Matching 回顾

流匹配模型学习从高斯噪声到数据分布的确定性变换。给定图像 $X$（或其潜变量）和噪声 $N \sim \mathcal{N}(0, I)$，中间潜变量通过线性插值构造：[^src-freqflow]

$$
X_t = (1 - t) \cdot X + t \cdot N, \quad t \in [0, 1]
$$

相应的速度场（velocity field）定义为时变导数：

$$
V_t = \frac{dX_t}{dt} = N - X
$$

训练目标为回归速度场：

$$
\mathcal{L} = \|f_\theta(X_t, t) - V_t\|^2
$$

### 频率视角的观察

论文通过对预训练的 SiT 模型进行频率分析，发现以下关键现象：[^src-freqflow]

1. **非均匀频率恢复**：低频成分（全局形状/颜色）在早期迅速生成，高频成分（纹理/边缘）在后期才出现
2. **高频误差主导**：SiT 的低频误差仅 0.08，但高频误差高达 0.69，说明模型难以恢复细节
3. **缺乏显式频率引导**：模型在空域操作，未对频域特性进行显式建模

为此，论文定义了频率误差来量化：

$$
\boxed{E = \mathbb{E}[|F_{\text{real}}|] - \mathbb{E}[|F_{\text{gen}}|]}
$$

其中 $F$ 表示傅里叶变换。[^src-freqflow]

---

## FreqFlow 架构

FreqFlow 采用双分支架构，由频率分支和空间分支组成，整体结构如图 4 所示。

### 3.1 频率分支 (Frequency Branch)

#### 频域分解

给定噪声图像 $X_t$（尺寸 $H \times W$），首先通过离散傅里叶变换（DFT）将其转换到频域：[^src-freqflow]

$$
F_t(u, v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} X_t(x, y) e^{-j2\pi\left(\frac{ux}{H} + \frac{vy}{W}\right)}
$$

其中 $F_t(u, v)$ 是位置 $(u, v)$ 处的复数值频域分量，$j$ 为虚数单位。

#### 高斯滤波分离频率成分

应用高斯高通和低通滤波器分离频率成分。对于高频分量 $H_t$ 和低频分量 $L_t$：[^src-freqflow]

$$
\boxed{H_t(u, v) = F_t(u, v) \cdot \left(1 - e^{-\frac{(u - \frac{H}{2})^2 + (v - \frac{W}{2})^2}{2\sigma_H^2}}\right)}
$$

$$
\boxed{L_t(u, v) = F_t(u, v) \cdot e^{-\frac{(u - \frac{H}{2})^2 + (v - \frac{W}{2})^2}{2\sigma_L^2}}}
$$

其中 $\sigma_H$ 和 $\sigma_L$ 控制高低通滤波的截止频率。$\sigma_L = 8$，$\sigma_H = 2$（默认）。高通滤波器增强边缘和纹理，低通滤波器保留全局结构并通过。

#### IDFT 重建空域表示

通过逆离散傅里叶变换（IDFT）将滤波后的频域分量重建回空域：[^src-freqflow]

$$
X_t^H(x, y) = \frac{1}{HW} \sum_{u=0}^{H-1} \sum_{v=0}^{W-1} H_t(u, v) e^{j2\pi\left(\frac{ux}{H} + \frac{vy}{W}\right)}
$$

$$
X_t^L(x, y) = \frac{1}{HW} \sum_{u=0}^{H-1} \sum_{v=0}^{W-1} L_t(u, v) e^{j2\pi\left(\frac{ux}{H} + \frac{vy}{W}\right)}
$$

#### 统一频率分支

论文未使用两个独立的网络分别处理高低频，而是设计了一个统一的频率分支 $f_{\text{freq}}$（Vision Transformer），同时处理两个频率分量：[^src-freqflow]

$$
\hat{V}_t^H, \hat{V}_t^L, h_t = f_{\text{freq}}(X_t^H, X_t^L, t, c)
$$

其中 $t$ 为时间步，$c$ 为类别条件，$\hat{V}_t^L$ 和 $\hat{V}_t^H$ 分别是低/高频预测速度场，$h_t$ 为融合的特征表示。统一分支相比分离分支 FID 提升 0.49（2.95 vs 3.44）。[^src-freqflow]

#### 自适应频率整合 (Adaptive Frequency Integration)

核心机制：使用可学习的时变权重 $\omega_t$ 动态平衡低频和高频分量的贡献：[^src-freqflow]

$$
\boxed{\omega_t = \sigma(\text{MLP}(h_t^L, h_t^H, t))}
$$

$$
\boxed{h_t = \omega_t \odot h_t^L + (1 - \omega_t) \odot h_t^H}
$$

其中 $\sigma$ 为 sigmoid 激活函数，MLP 为处理拼接特征和时间步的多层感知机，$\odot$ 为逐元素乘法。

**权重演化模式**：在生成早期（时间步 1000→~600），$\omega_t$（低频权重）较大，模型优先建立全局结构；随着生成推进（~600→0），$(1 - \omega_t)$ 逐渐增大，模型逐步转向高频细节的精细调整。这一行为证明了 FreqFlow 显式实现了"先粗后细"的生成轨迹。[^src-freqflow]

---

### 3.2 空间分支 (Spatial Branch)

空间分支负责在潜域中合成图像，接收来自频率分支的特征 $h_t$ 作为引导。[^src-freqflow]

#### 融合与处理

$$
\boxed{\hat{V}_t = f_{\text{spatial}}(\text{merge}(X_t, h_t), t, c)}
$$

其中 $\text{merge}$ 通过逐元素加法实现（实验证明优于交叉注意力和通道拼接）。$f_{\text{spatial}}$ 使用 ConvNeXt 实现，因为 ConvNeXt 在捕获高频细节（边缘、纹理）方面优于 ViT。[^src-freqflow]

**融合方式消融（FID）**：[^src-freqflow]

| 融合方式 | FID↓ | IS↑ |
|---------|------|-----|
| 交叉注意力 | 3.95 | 198.6 |
| 通道拼接 | 3.46 | 224.8 |
| **逐元素加法** | **2.95** | **231.5** |

---

### 3.3 双域监督训练策略 (Dual-domain Supervision)

FreqFlow 同时使用空间域损失和频率域损失进行训练，确保模型在双域中都能有效学习。[^src-freqflow]

#### 损失函数定义

空间域 L2 损失：
$$
\boxed{\mathcal{L}_s(y, \hat{y}) = \|y - \hat{y}\|_2^2}
$$

频率域 L2 损失（直接监督频域输出）：
$$
\boxed{\mathcal{L}_f(y, \hat{y}) = \|\text{FFT}(y) - \text{FFT}(\hat{y})\|_2^2}
$$

#### 完整训练目标

最终损失函数整合了空间分支和频率分支的监督信号：[^src-freqflow]

$$
\boxed{\mathcal{L} = \mathcal{L}_s(\hat{V}_t, V_t) + \mathcal{L}_f(\hat{V}_t, V_t) + \alpha\big(\mathcal{L}_s(\hat{V}_t^H, V_t^H) + \mathcal{L}_s(\hat{V}_t^L, V_t^L) + \mathcal{L}_f(\hat{V}_t^H, V_t^H) + \mathcal{L}_f(\hat{V}_t^L, V_t^L)\big)}
$$

其中：
- 前两项 $\mathcal{L}_s(\hat{V}_t, V_t) + \mathcal{L}_f(\hat{V}_t, V_t)$：空间分支的全局双域监督
- 中间项 $\mathcal{L}_s(\hat{V}_t^H, V_t^H) + \mathcal{L}_s(\hat{V}_t^L, V_t^L)$：频率分支的逐分量空间监督
- 末两项 $\mathcal{L}_f(\hat{V}_t^H, V_t^H) + \mathcal{L}_f(\hat{V}_t^L, V_t^L)$：频率分支的逐分量频域监督
- $\alpha = 0.5$（默认，对 $\alpha$ 取值相对不敏感）

**频域损失的重要性消融**：[^src-freqflow]

| 配置 | FID↓ | IS↑ |
|-----|------|-----|
| 去掉频域监督 | 4.67 | 198.4 |
| 完整双域监督 | 2.95 | 231.5 |

FID 提升 1.72，证明了频率域监督的关键作用。

---

## 模型变体

FreqFlow 提供三种模型变体，参数量和深度随模型级别递增：[^src-freqflow]

| 模型 | 频率分支深度 | 空间分支深度 | 频率分支隐藏维度 | 空间分支隐藏维度 | 总参数量 |
|-----|------------|------------|---------------|---------------|---------|
| FreqFlow-B | 15 | 12 | 768 | 384 | 134M |
| FreqFlow-L | 39 | 20 | 960 | 480 | 507M |
| FreqFlow-H | 57 | 29 | 1152 | 576 | 1.08B |

训练超参数：优化器 AdamW ($\beta_1=0.99, \beta_2=0.99$)，权重衰减 0.03，batch size 2048，学习率 2e-4（常数调度），总 epoch 800，warmup 5 epoch，类别标签 dropout 0.1。推理采用 ODE 模式，250 步。[^src-freqflow]

---

## 实验结果

### ImageNet-64 (pixel space)

| 模型 | 参数量 | FID↓ | IS↑ |
|-----|--------|------|-----|
| U-ViT-M/4 | 131M | 5.85 | 33.71 |
| U-ViT-L/4 | 287M | 4.26 | 40.66 |
| DiMR-M/3R | 133M | 3.65 | 42.41 |
| DiMR-L/3R | 284M | 2.21 | 55.73 |
| **FreqFlow-B** | **134M** | **1.92** | **59.34** |

FreqFlow-B 以 134M 参数超越 284M 参数的 DiMR-L/3R。[^src-freqflow]

### ImageNet-256 (latent space, w/ CFG)

| 模型 | 类型 | 参数量 | FID↓ |
|-----|------|--------|------|
| DiT-XL/2 | 扩散 | 675M | 2.27 |
| SiT-XL/2 | 流匹配 | 675M | 2.06 |
| DiMR-G/2R | 扩散 | 1.06B | 1.63 |
| MAR-H | AR | 943M | 1.55 |
| **FreqFlow-L** | **流匹配** | **507M** | **1.54** |
| **FreqFlow-H** | **流匹配** | **1.08B** | **1.38** |

FreqFlow-L (507M) 超越 DiT-XL/2 (675M) 和 SiT-XL/2 (675M)，FreqFlow-H 以 1.38 FID 为基于流匹配的生成模型 SOTA。[^src-freqflow]

### ImageNet-256 (w/o CFG)

| 模型 | 参数量 | FID↓ |
|-----|--------|------|
| DiT-XL/2 | 675M | 9.62 |
| U-ViT-H/2 | 501M | 6.58 |
| DiMR-G/2R | 1.06B | 3.56 |
| **FreqFlow-L** | **507M** | **3.12** |
| **FreqFlow-H** | **1.08B** | **2.45** |

无 CFG 情况下 FreqFlow-L 超越 1.06B 参数的 DiMR-G 0.44 FID。[^src-freqflow]

### ImageNet-512

| 模型 | 参数量 | FID↓ |
|-----|--------|------|
| ADM-U | 731M | 3.85 |
| DiT-XL/2 | 675M | 3.04 |
| DiMR-XL/3R | 525M | 2.89 |
| DiffiT | 561M | 2.67 |
| **FreqFlow-L** | **507M** | **2.02** |

FreqFlow-L 超越 DiT-XL/2 (+1.02 FID) 和 DiffiT (+0.65 FID)。[^src-freqflow]

---

## 频率误差分析

论文定量比较了 FreqFlow 与 SiT 的频域误差：[^src-freqflow]

| 模型 | 低频误差 | 高频误差 |
|-----|---------|---------|
| SiT | 0.08 | 0.69 |
| **FreqFlow** | **0.06** | **0.48** |

FreqFlow 在低/高频上误差均更低，高频误差降低 30%，有效证明了频率感知设计的优势。[^src-freqflow]

---

## 相关对比

### vs [[frequency-based-noise-control|频域噪声控制]] (2025)

二者都关注频率域，但方法论不同：

| 维度 | 频域噪声控制 | FreqFlow |
|------|------------|----------|
| 修改对象 | 前向加噪的噪声频谱 | 模型架构（双分支）和训练（双域损失） |
| 框架 | 扩散模型 | 流匹配模型 |
| 机制 | 操控 $\boldsymbol{\epsilon} \to \boldsymbol{\epsilon}^{(w)}$ | 显式频率分支 + 自适应整合 |
| 频域处理 | 仅在前向过程 | 在前向、逆向、损失中均涉及 |
| 是否修改架构 | 否（即插即用） | 是（全新双分支架构） |

二者可互补：噪声频谱操控 + 频率感知架构。[^src-freqflow]

### vs [[flow-matching|Flow Matching]] (2023)

FreqFlow 建立在 Flow Matching 基础上：
- 保留 Flow Matching 的线性插值路径 $X_t = (1-t) \cdot X + t \cdot N$
- 标准 FM 目标 $\mathcal{L} = \|f_\theta(X_t, t) - V_t\|^2$ 被替换为双域损失
- 模型输出从单一速度场 $\hat{V}_t$ 扩展到多分量（$\hat{V}_t, \hat{V}_t^H, \hat{V}_t^L$）

### vs SiT (2024)

SiT 将 DiT 架构与 Flow Matching 结合，但仍在纯空域操作。FreqFlow 显式引入频率分支，以额外参数量换取频率感知能力。在类似参数量下（FreqFlow-L 507M vs SiT-XL 675M），FreqFlow-L 的 FID 更优（1.54 vs 2.06）。[^src-freqflow]

---

## 关键洞见总结

1. **"先粗后细"的生成轨迹**：频率分支的自适应权重 $\omega_t$ 从初期偏向低频逐步过渡到后期重视高频，与人类视觉感知的"先识别形状再注意细节"一致
2. **加法融合的简洁性**：在三种融合方式中（交叉注意力、通道拼接、加法），最简单的加法最优
3. **频域损失的不可替代性**：去掉频率域损失导致 FID 从 2.95 劣化到 4.67，说明纯空域损失不足以引导频率学习
4. **统一分支优于分离分支**：统一的 ViT 处理高低频比两个独立网络更有效

## 引用

[^src-freqflow]: [[source-freqflow]]
