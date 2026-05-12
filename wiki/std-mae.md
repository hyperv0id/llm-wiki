---
title: "STD-MAE: Spatial-Temporal-Decoupled Masked AutoEncoder"
type: technique
tags:
  - masked-pre-training
  - spatiotemporal-forecasting
  - self-supervised-learning
  - traffic-prediction
  - autoencoder
created: 2026-05-12
last_updated: 2026-05-12
source_count: 1
confidence: high
status: active
---

# STD-MAE: Spatial-Temporal-Decoupled Masked AutoEncoder

**STD-MAE**（Spatial-Temporal-Decoupled Masked AutoEncoder）是一种面向时空预测的时空解耦掩码预训练框架，由 Gao 等人于 IJCAI-2024 提出。核心思想是在预训练阶段通过两个独立的掩码自编码器分别沿空间和时间维度重建时空序列，学习清晰的异质性表征以增强下游任意架构的预测器[^src-2312-00516-std-mae]。

## 方法概览

```
                    ┌─────────────┐
 长输入序列 ──────► │ Patch Embed │ ──► E ∈ ℝ^(Tp×N×D)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         S-Mask       保留全部    T-Mask
              │                     │
              ▼                     ▼
        ┌──────────┐         ┌──────────┐
        │ S-MAE    │         │ T-MAE    │
        │ (空间自注)│         │ (时间自注)│
        └────┬─────┘         └────┬─────┘
             │                     │
          H^(S)                H^(T)
             │                     │
              └──────┬──────┘
                     ▼
            MLP 投影 + 预测器融合
                     │
                     ▼
              增强预测结果
```

## 核心技术

### 时空解耦掩码

给定输入 $X \in \mathbb{R}^{T \times N \times C}$，掩码比率 $r$，定义两种独立的伯努利采样[^src-2312-00516-std-mae]：

- **S-Mask**：$\tilde{X}^{(S)} = \sum_{n=1}^{N} \mathcal{B}_S(1-r) \cdot X[:, n, :]$ — 随机保留部分传感器的时间序列
- **T-Mask**：$\tilde{X}^{(T)} = \sum_{t=1}^{T} \mathcal{B}_T(1-r) \cdot X[t, :, :]$ — 随机保留部分时间步

### Patch Embedding 与位置编码

为处理长期输入（如 864 时间步），首先通过非重叠 patch 窗口 $L$ 将序列分割为 $T_p = T_{long}/L$ 个 patch，然后通过全连接层映射到嵌入维度 $D$。使用**二维正弦位置编码**同时编码时间和空间位置[^src-2312-00516-std-mae]：

$$E_{pos}[t, n, 2i] = \sin(t / 10000^{4i/D})$$
$$E_{pos}[t, n, 2i+1] = \cos(t / 10000^{4i/D})$$
$$E_{pos}[t, n, 2j + D/2] = \sin(n / 10000^{4j/D})$$
$$E_{pos}[t, n, 2j + 1 + D/2] = \cos(n / 10000^{4j/D})$$

### 非对称解码器

遵循 MAE 设计哲学，使用轻量级解码器（1 层 Transformer + 回归层）以减少预训练时间。解码器包含三个组件[^src-2312-00516-std-mae]：

1. **Padding 层**：用共享可学习 mask 令牌 $V \in \mathbb{R}^D$ 填充被掩码位置
2. **Transformer 层**：对完整 patch 集合做自注意力
3. **回归层**：在 patch 级别重建时间序列

损失函数仅在掩码部分计算 MAE：

$$\mathcal{L}_S = \frac{1}{T_p N_M L} \sum_{t,n,l} |\hat{Q}^{(S)}[t,n,l] - Q^{(S)}[t,n,l]|$$
$$\mathcal{L}_T = \frac{1}{T_M N L} \sum_{t,n,l} |\hat{Q}^{(T)}[t,n,l] - Q^{(T)}[t,n,l]|$$

### 下游集成

预训练后，将 S-MAE 和 T-MAE 编码器输出的表征通过 MLP 投影到预测器隐藏维度，与预测器自身隐藏表示相加[^src-2312-00516-std-mae]：

$$H^{(Aug)} = MLP(H'^{(S)}) + MLP(H'^{(T)}) + H^{(F)}$$

此操作无需修改预测器原始架构，实现即插即用。

## 关键发现

| 超参数 | 最优值 | 比较 |
|--------|--------|------|
| 掩码比率 $r$ | 0.25 | 不同于 CV-MAE 的 75% 和 BERT 的 15% |
| 预训练长度 $T_{long}$ | 864（3 天） | 多数数据集 3 天最优，部分需要 7 天 |
| Transformer 层数（编码器） | 4 | — |
| Transformer 层数（解码器） | 1 | 轻量设计 |
| 嵌入维度 $D$ | 96 | — |
| 注意力头数 | 4 | — |
| Patch 大小 $L$ | 12 | 与预测输入长度对齐 |

## 相关技术

- MAE (Masked AutoEncoder) — CV 中的掩码自编码器
- [[spatiotemporal-mirage]] — 时空幻象问题
- [[traffic-forecasting]] — 交通预测综述
- [[spatio-temporal-decoupling]] — 时空解耦（不同语境，共形预测）

[^src-2312-00516-std-mae]: [[source-2312-00516-std-mae]]
