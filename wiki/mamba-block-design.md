---
title: "Mamba Block Design"
type: technique
tags:
  - mamba
  - block-design
  - architecture
created: 2026-05-08
last_updated: 2026-05-08
source_count: 1
confidence: medium
status: active
---

# Mamba Block Design

**Mamba 的块设计（Block Design）** 是 Mamba 架构中与标准 Transformer 块的关键结构差异。Han et al. (NeurIPS 2024) 通过消融实验证明，块设计是 Mamba 在视觉任务中成功的最重要单一因素，贡献了 +1.8% 的准确率提升[^src-demystify-mamba-linear-attention-2024]。

## 三项结构差异

Mamba 的块设计与标准 Transformer 块存在三项差异[^src-demystify-mamba-linear-attention-2024]：

### 1. 子块顺序交换

标准 Transformer 块中，注意力子块在前、线性子块在后。Mamba 将其交换：**线性子块在前，注意力类子块在后**。

### 2. 拼接替代相加

标准 Transformer 使用残差相加（$x + \text{Attention}(x)$）。Mamba 将两个子块的输出进行**拼接（Concatenation）**，而非相加。

### 3. 统一归一化

标准 Transformer 在每个子块后分别应用层归一化。Mamba 在拼接后使用**单一的统一归一化**。

## 消融实验结果

论文通过逐步替换线性注意力块中的组件来量化块设计的影响[^src-demystify-mamba-linear-attention-2024]：

| 替换内容 | ImageNet-1K Top-1 | 变化 |
|---------|-------------------|------|
| 基线线性注意力 | 77.6% | — |
| + 替换注意力子块设计 | 79.4% | **+1.8%** |
| + 替换线性子块设计 | 78.3% | +0.7% |
| + 替换两者 | 79.4% | +1.8% |

关键发现：替换注意力子块设计即可获得全部 +1.8% 的提升，替换线性子块设计仅贡献 +0.7%，且两者同时替换不叠加。

## 与标准 Transformer 的对比

| 特性 | 标准 Transformer | Mamba |
|------|-----------------|-------|
| 子块顺序 | 注意力 → 线性 | 线性 → 注意力 |
| 子块融合 | 残差相加 | 拼接 |
| 归一化位置 | 每个子块后分别归一化 | 拼接后统一归一化 |
| 对准确率贡献 | — | +1.8% |

## 相关页面

- [[mamba|Mamba]] — 选择性状态空间模型
- [[mila|MILA]] — 采用此块设计的线性注意力模型
- [[linear-attention-unified-framework|Mamba ↔ Linear Attention 统一框架]]

[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
