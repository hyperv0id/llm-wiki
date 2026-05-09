---
title: "FreqFlow: Frequency-Aware Flow Matching for High-Quality Image Generation"
type: source-summary
tags:
  - flow-matching
  - frequency-domain
  - image-generation
  - transformer
  - arxiv-2026
created: 2026-05-09
last_updated: 2026-05-09
source_count: 1
confidence: medium
status: active
---

# FreqFlow: Frequency-Aware Flow Matching for High-Quality Image Generation

Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, Liang-Chieh Chen. Johns Hopkins University & ByteDance. arXiv:2604.15521v1, April 2026.[^src-freqflow]

## 核心贡献

FreqFlow 提出了一种显式频率感知的流匹配生成框架，采用双分支架构（频率分支 + 空间分支），在 ImageNet-256 上取得 **FID 1.38** 的 SOTA 结果，超越 DiT (+0.79) 和 SiT (+0.58)。[^src-freqflow]

### 三⼤创新

1. **频率分支**：通过离散傅里叶变换（DFT）将噪声图像分解为低频（全局结构）和高频（纹理细节）分量，使用时变自适应加权机制 $\omega_t = \sigma(\text{MLP}(h_t^L, h_t^H, t))$ 动态平衡两者的贡献。
2. **空间分支**：在潜域中合成图像，接收频率分支的输出 $h_t$ 作为引导，通过逐元素加法融合。
3. **双域监督**：同时使用空间域损失 $\mathcal{L}_s(y, \hat{y}) = \|y - \hat{y}\|_2^2$ 和频率域损失 $\mathcal{L}_f(y, \hat{y}) = \|\text{FFT}(y) - \text{FFT}(\hat{y})\|_2^2$ 进行训练。[^src-freqflow]

## 方法论

### Flow Matching 预备

流匹配学习从高斯噪声到数据分布的确定性变换。中间潜变量 $X_t = (1-t) \cdot X + t \cdot N$，速度场 $V_t = dX_t/dt = N - X$，训练目标为 $\mathcal{L} = \|f_\theta(X_t, t) - V_t\|^2$。[^src-freqflow]

### 从频率视角分析

论文发现流匹配模型先在早期生成低频成分（全局结构），晚期才生成高频细节（纹理）。频率误差分析显示 SiT 的低频误差为 0.08 而高频误差高达 0.69，表明模型难以恢复细节。[^src-freqflow]

### 双分支架构

- **频率分支**：使用 DFT $F_t(u, v)$ 转换噪声图像，通过高斯高通/低通滤波器分离频率成分，用统一 ViT 处理两者。自适应权重 $\omega_t$ 在早期偏向低频，后期逐步转向高频。
- **空间分支**：使用 ConvNeXt 处理合并了频率特征 $h_t$ 的噪声图像。逐元素加法融合优于交叉注意力和通道拼接。[^src-freqflow]

### 完整训练目标

$$\mathcal{L} = \mathcal{L}_s(\hat{V}_t, V_t) + \mathcal{L}_f(\hat{V}_t, V_t) + \alpha(\mathcal{L}_s(\hat{V}_t^H, V_t^H) + \mathcal{L}_s(\hat{V}_t^L, V_t^L) + \mathcal{L}_f(\hat{V}_t^H, V_t^H) + \mathcal{L}_f(\hat{V}_t^L, V_t^L))$$

其中 $\alpha = 0.5$。[^src-freqflow]

## 实验结果

| 分辨率 | 模型 | 参数量 | FID↓ |
|-------|------|--------|------|
| 64×64 | FreqFlow-B | 134M | 1.92 |
| 256×256 | FreqFlow-L | 507M | 1.54 (w/ CFG) / 3.12 (w/o CFG) |
| 256×256 | FreqFlow-H | 1.08B | 1.38 (w/ CFG) / 2.45 (w/o CFG) |
| 512×512 | FreqFlow-L | 507M | 2.02 |

在 ImageNet-256 上，FreqFlow-H (1.38 FID) 超越 DiT-XL (2.27)、SiT-XL (2.06)、DiMR-G (1.63)、MAR-H (1.55) 等基线。[^src-freqflow]

## 消融研究关键发现

1. **低+高频同时使用最佳**：单独低频 FID 3.55、单独高频 3.12、二者结合 2.95
2. **加法融合最优**：加法 2.95 FID vs 交叉注意力 3.95 vs 通道拼接 3.46
3. **频域损失关键**：去掉频域损失 FID 从 2.95 劣化至 4.67
4. **统一频率分支优于分离分支**：统一 2.95 vs 分离 3.44 FID[^src-freqflow]

## 局限性

受限于计算资源，最大模型 FreqFlow-H 仅约 1B 参数，更大规模的扩展留作未来工作。[^src-freqflow]

## 引用

[^src-freqflow]: [[source-freqflow]]
