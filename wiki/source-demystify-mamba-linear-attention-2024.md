---
title: "Demystify Mamba in Vision: A Linear Attention Perspective"
type: source-summary
tags:
  - mamba
  - linear-attention
  - vision-transformer
  - state-space-model
created: 2026-05-08
last_updated: 2026-05-08
source_count: 1
confidence: medium
status: active
---

# Demystify Mamba in Vision: A Linear Attention Perspective

**Authors**: Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, Gao Huang (Tsinghua University & Alibaba Group)

**Venue**: NeurIPS 2024

## 核心贡献

本文的核心贡献是将 Mamba（选择性状态空间模型）与线性注意力 Transformer 统一到一个框架中，证明 Mamba 本质上是线性注意力的一个变体，两者之间存在 **六项关键差异**[^src-demystify-mamba-linear-attention-2024]。

## 统一框架

论文通过数学推导建立了 Mamba 与线性注意力的对应关系[^src-demystify-mamba-linear-attention-2024]：

| Mamba 组件 | 线性注意力对应 |
|-----------|---------------|
| $C_i$（输出投影） | $Q_i$（查询） |
| $B_j$（输入投影） | $K_j^\top$（键） |
| $\Delta_j \odot x_j$（输入门控值） | $V_j$（值） |
| $\prod_{k=j+1}^i \widetilde{A}_k$（遗忘门） | 隐式恒等（线性注意力中无遗忘） |
| $D \odot x_i$（捷径连接） | 无对应 |
| $Q_i Z_i$（归一化分母） | Mamba 中缺失 |

## 六项差异的消融分析

论文通过系统消融实验量化每项差异的影响[^src-demystify-mamba-linear-attention-2024]：

1. **输入门**（$\mathbf{\Delta}_i$）：对输入进行逐元素门控，ImageNet-1K 准确率 +0.9%，但吞吐量从 1152 降至 106 im/s
2. **遗忘门**（$\widetilde{\mathbf{A}}_i$）：输入依赖的指数衰减，+0.8% 准确率（77.6→78.4），但吞吐量降至 743 im/s（因循环计算）
3. **捷径连接**（$D \odot x_i$）：残差连接，+0.2% 准确率
4. **无注意力归一化**：移除归一化导致准确率从 77.6 暴跌至 72.4（-5.2%），证明归一化对线性注意力至关重要
5. **单头**：Mamba 使用单头而非多头，影响为边际/负面
6. **修改的块设计**：交换子块顺序 + 拼接 + 统一归一化，+1.8% 准确率（77.6→79.4），是最有影响力的设计

## MILA 架构

基于上述分析，论文提出 **MILA（Mamba-Inspired Linear Attention）**，将遗忘门替换为位置编码（LePE、CPE、RoPE），保留线性注意力的可并行计算特性[^src-demystify-mamba-linear-attention-2024]。MILA 在 ImageNet-1K、COCO 目标检测和 ADE20K 语义分割上全面超越 Vision Mamba 基线。

## 局限性

- 分析基于视觉任务，Mamba 在语言建模中的优势可能来自不同机制
- 遗忘门在长序列语言任务中的位置编码替代方案尚未充分验证

[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
