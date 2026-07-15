---
title: "Mamba"
type: entity
tags:
  - state-space-model
  - sequence-modeling
  - linear-complexity
created: 2026-05-08
last_updated: 2026-07-21
source_count: 3
confidence: medium
status: active
---

# Mamba

**Mamba** 是一种选择性状态空间模型（Selective State Space Model, SSM），通过输入依赖的参数化机制解决了传统 SSM 在内容感知建模上的局限。Mamba 在语言、视觉等多个领域展现出与 Transformer 竞争的性能，同时保持线性计算复杂度[^src-demystify-mamba-linear-attention-2024]。

## 与线性注意力的统一

Han et al. (NeurIPS 2024) 证明 Mamba 可以重新表述为线性注意力 Transformer 的一个变体，两者之间存在 **六项关键差异**[^src-demystify-mamba-linear-attention-2024]：

1. **输入门**（$\mathbf{\Delta}_i$）：逐元素门控输入，控制信息进入隐藏状态的比例
2. **遗忘门**（$\widetilde{\mathbf{A}}_i$）：输入依赖的指数衰减，控制历史信息的保留程度
3. **捷径连接**（$D \odot x_i$）：从输入到输出的残差连接
4. **无注意力归一化**：Mamba 缺少线性注意力中的归一化分母 $Q_i Z_i$
5. **单头**：Mamba 使用单头而非多头注意力
6. **修改的块设计**：子块顺序交换 + 拼接 + 统一归一化

## 核心机制

Mamba 的选择性机制通过输入依赖的参数 $\mathbf{\Delta}_i$、$\mathbf{B}_i$、$\mathbf{C}_i$ 实现内容感知的序列建模[^src-demystify-mamba-linear-attention-2024]：

$$h_i = \widetilde{A}_i \odot h_{i-1} + B_i(\Delta_i \odot x_i)$$
$$y_i = C_i h_i + D \odot x_i$$

其中遗忘门 $\widetilde{A}_i = \text{diag}(\exp(\Delta_i A))$ 位于 $(0,1)$ 区间，实现输入依赖的指数衰减。

## 关键发现

消融实验表明，Mamba 在视觉任务中的成功主要归功于[^src-demystify-mamba-linear-attention-2024]：

- **块设计**（+1.8%）是最有影响力的单一因素
- **遗忘门**（+0.8%）提供局部偏置和输入依赖的位置信息，但可通过位置编码替代以保持并行性
- **归一化的缺失**导致 -5.2% 的准确率下降，说明 Mamba 通过其他机制补偿了归一化

## Mamba as Diffusion Backbone

SSD-TS (KDD 2025) demonstrates Mamba's effectiveness as a denoising backbone in conditional diffusion models for time series imputation, replacing Transformer/S4 backbones with Mamba-based modules [^src-ssdts]. Two key advantages over attention-based diffusion backbones:

1. **Better noise-signal discrimination**: During early diffusion steps when data is mostly noise, attention weights derived from noisy input similarity are misleading. Mamba's state update $h_t = A_t h_{t-1} + B_t x_t$ is independent of content similarity, and its gating mechanism helps filter noise [^src-ssdts].

2. **Controllable frequency response**: SSMs offer frequency response $\hat{K}(\omega) = C(i\omega I-A)^{-1}B$, enabling selective suppression of broadband noise while preserving signal in specific frequency bands — impossible with frequency-unbiased attention [^src-ssdts].

SSD-TS introduces BAM (bidirectional Mamba + temporal attention for intra-channel) and CMB (unidirectional Mamba for inter-channel) as specialized Mamba modules for the diffusion denoising context [^src-ssdts].

## 相关页面

- [[s-mamba|S-Mamba]] — 首个 Mamba-based 多变量时间序列预测框架 (Neurocomputing 2024)
- [[dst-mamba|DST-Mamba]] — 分解式时空 Mamba，用于长程交通预测 (AAAI 2025)
- [[ssd-ts|SSD-TS]] — Mamba 作为扩散模型去噪 backbone 用于时间序列插补 (KDD 2025)
- [[bam|BAM]] — 双向注意力 Mamba，扩散去噪的通道内建模模块
- [[cmb|CMB]] — 通道 Mamba 块，扩散去噪的通道间建模模块
- [[mila|MILA]] — 受 Mamba 启发的线性注意力模型
- [[linear-attention-unified-framework|Mamba ↔ Linear Attention 统一框架]]
- [[forget-gate-in-sequential-models|遗忘门在序列模型中的作用]]
- [[mamba-block-design|Mamba 块设计]]
- [[e2-cstp|E²-CSTP]] — GCN+Mamba 混合时空编码器，17-56% 效率提升 vs Transformer (NeurIPS 2025)
- [[rivermamba|RiverMamba]] — 首个全球 0.05° 河流流量和洪水预报 Mamba 模型，空间填充曲线 + LOAN + Hindcast-Forecast 架构 (NeurIPS 2025)

[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
[^src-ssdts]: [[source-ssdts]]
[^src-rivermamba]: [[source-rivermamba]]
