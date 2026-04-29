---
title: "Contract-and-Broadcast Transformer (CBT)"
type: entity
tags:
  - transformer
  - interpretable-attention
  - cbsa
  - vision-transformer
  - neurips-2025
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# Contract-and-Broadcast Transformer (CBT)

**CBT**（Contract-and-Broadcast Transformer）是基于 CBSA 构建的 Transformer 架构，通过堆叠 CBSA 层与 ISTA（Iterative Shrinkage-Thresholding Algorithm）模块实现完全可解释的视觉模型[^src-cbsa]。

## 架构设计

### 整体结构

CBT 由交替的 CBSA 注意力层和 ISTA 前馈网络组成：

```
Input Image → Conv Embed → [CBSA → ISTA] × L → Output
```

- **CBSA**：实现 token 间的信息交互（token mixer），通过压缩目标驱动[^src-cbsa]
- **ISTA 模块**：通过算法展开导出，处理稀疏性惩罚和展开项，是 CBSA 的配套前馈模块[^src-cbsa]

### 与标准 ViT 的对比

| 组件 | ViT | CBT |
|------|-----|-----|
| Token Mixer | MHSA (Multi-Head Self-Attention) | CBSA (Contract-and-Broadcast) |
| Channel Mixer | MLP | ISTA |
| 可解释性 | 黑盒 | 白盒（每步对应优化目标） |
| 复杂度 | O(N²) | O(N)（固定 m） |

## 模型规模

CBT 系列提供四个规模[^src-cbsa]：

| 模型 | 参数量 | FLOPs (ImageNet) | ImageNet-1K Top-1 |
|------|--------|------------------|-------------------|
| CBT-Tiny | 1.8M | 1.1G | 63.2 |
| CBT-Small | 6.7M | 4.0G | 71.4 |
| CBT-Base | 25.7M | 15.1G | 73.4 |
| CBT-Large | 83.1M | 47.3G | 74.4 |

## 实验结果

### ImageNet-1K 分类

CBT-Small 以 ViT-S 30% 的参数量和 40% 的 FLOPs 达到可比精度（71.4 vs 72.4）[^src-cbsa]：

| 模型 | 参数量 | FLOPs | ImageNet | CIFAR-10 | CIFAR-100 |
|------|--------|-------|----------|----------|-----------|
| ViT-S | 22.1M | 9.8G | 72.4 | 97.2 | 83.2 |
| CRATE-B | 22.8M | 12.6G | 70.8 | 96.8 | 82.7 |
| **CBT-S** | **6.7M** | **4.0G** | **71.4** | **96.3** | **80.4** |

### 语义分割（ADE20K）

CBT decoder 在 ADE20K 数据集上以更少的计算量超越 Segmenter 和 DEPICT[^src-cbsa]：

- CBT decoder 以 Segmenter 20% 的 FLOPs 实现 +1.5% mIoU 提升
- pairwise similarity 计算量仅为 Segmenter 的 0.06%

### 预训练适配

实验探索将预训练 ViT（ImageNet-21K 预训练）适配到 CBSA 风格：
- CBSA∗：使用 ViT 预训练的三个独立投影矩阵计算 Q、K、V
- CBSA∨：CBSA∗ 移除收缩步骤（等效于 Agent Attention）

结果：CBSA 实现与 linear attention 可比的性能，FLOPs 相近[^src-cbsa]。

## 可解释性验证

### 1. 紧凑结构化表示

在合成数据（10 类，每类样本来自一维子空间 + 噪声）上，CBSA 迭代确实将 token 压缩到低维子空间，实现良好分离[^src-cbsa]。

### 2. 涌现的分割特性

- 仅使用标准监督分类训练，CRATE（MSSA）就涌现出分割特性
- CBT 在早期层关注更多语义有意义区域，但分割特性在后续层衰减
- **CRATE + CBT 混合模型**：前半使用 MSSA，后半使用 CBSA，分割特性不仅涌现而且持续增强[^src-cbsa]

### 3. 抗参数扰动

CBSA 的注意力头建模为低维子空间，扰动投影矩阵（子空间基）不会显著改变其张成的子空间。实验验证参数扰动下 CBT 性能下降远小于 Segmenter（黑盒方法）[^src-cbsa]。

## 与 CRATE 的关系

| 方面 | CRATE | CBT |
|------|-------|-----|
| 注意力机制 | MSSA（Multi-head Subspace Self-Attention） | CBSA |
| 复杂度 | O(N²)（Gram 矩阵） | O(N)（固定 m） |
| 注意力变体 | 仅 softmax | softmax / linear / channel / agent |
| 性能（ImageNet） | 70.8 (B) | 73.4 (B) |

## 引用

[^src-cbsa]: [[source-cbsa]]