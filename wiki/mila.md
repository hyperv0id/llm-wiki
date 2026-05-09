---
title: "MILA"
type: entity
tags:
  - linear-attention
  - vision-transformer
  - mamba-inspired
created: 2026-05-08
last_updated: 2026-05-08
source_count: 1
confidence: medium
status: active
---

# MILA

**MILA（Mamba-Inspired Linear Attention）** 是 Han et al. (NeurIPS 2024) 提出的视觉架构，基于 Mamba 与线性注意力的统一框架，将 Mamba 的关键设计融入线性注意力中，同时保留可并行计算的优势[^src-demystify-mamba-linear-attention-2024]。

## 核心设计

MILA 从 Mamba 中提取两个最有价值的设计，并将其适配到线性注意力框架中[^src-demystify-mamba-linear-attention-2024]：

1. **遗忘门 → 位置编码替换**：Mamba 的遗忘门 $\widetilde{A}_i$ 提供局部偏置和输入依赖的位置信息。MILA 使用 **LePE（Local Enhanced Positional Encoding）**、**CPE（Conditional Positional Encoding）** 和 **RoPE** 替代遗忘门的功能，从而避免循环计算，保持并行化
2. **块设计**：采用 Mamba 的修改块设计（交换子块顺序 + 拼接 + 统一归一化）

## 架构变体

MILA 提供三种规模[^src-demystify-mamba-linear-attention-2024]：

| 变体 | 参数量 | ImageNet-1K Top-1 |
|------|--------|-------------------|
| MILA-T (Tiny) | — | — |
| MILA-S (Small) | — | 81.5% |
| MILA-B (Base) | — | 83.4% |

## 性能对比

MILA 在多个视觉任务上全面超越 Vision Mamba 基线[^src-demystify-mamba-linear-attention-2024]：

- **ImageNet-1K 分类**：MILA-B 83.4% vs VMamba-B 82.2%；MILA-S 81.5% vs VMamba-S 80.4%
- **COCO 目标检测与实例分割**：全面超越所有 Mamba 基线
- **ADE20K 语义分割**：全面超越所有 Mamba 基线
- **推理速度**：由于可并行计算，MILA 显著快于 Vision Mamba 模型

## 意义

MILA 证明了 Mamba 在视觉任务中的成功并非来自 SSM 的独特优势，而是来自其特定的设计选择（遗忘门和块设计），这些设计可以移植到线性注意力中，获得更好的性能和效率[^src-demystify-mamba-linear-attention-2024]。

## 相关页面

- [[mamba|Mamba]] — 选择性状态空间模型
- [[linear-attention-unified-framework|Mamba ↔ Linear Attention ��一框架]]
- [[forget-gate-in-sequential-models|遗忘门在序列模型中的作用]]
- [[mamba-block-design|Mamba 块设计]]

[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
