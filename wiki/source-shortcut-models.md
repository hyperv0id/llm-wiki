---
title: "One Step Diffusion via Shortcut Models"
type: source-summary
tags:
  - shortcut-models
  - one-step-generation
  - flow-matching
  - distillation
  - uc-berkeley
  - arxiv-2025
created: 2026-04-28
last_updated: 2026-04-28
source_count: 0
confidence: medium
status: active
---

# One Step Diffusion via Shortcut Models

**Shortcut Models** 是由 UC Berkeley 的 Kevin Frans、Danijar Hafner、Sergey Levine、Pieter Abbeel 于 2025 年发表的论文（arXiv:2410.12557），提出了一种单网络、单训练阶段的少步/单步生成模型[^src-shortcut-models]。

## 核心贡献

### 1. 核心思想

传统扩散模型需要数十到数百步迭代去噪。Shortcut Models 的关键洞察是：**不仅根据噪声水平，还根据期望的步长来调节网络**，使其能够"跳过"生成过程中的多个步骤。

### 2. 自一致性约束

Shortcut 模型满足自一致性性质：

$$
s(x_t, t, 2d) = \frac{1}{2}s(x_t, t, d) + \frac{1}{2}s(x'_{t+d}, t+d, d)
$$

其中 $x'_{t+d} = x_t + s(x_t, t, d) \cdot d$ 是沿预测方向的一步。

### 3. 训练目标

$$
\mathcal{L}_S = \mathbb{E}_{x_0 \sim \mathcal{N}, x_1 \sim D, (t,d)} \left[ \|s_\theta(x_t, t, 0) - (x_1 - x_0)\|^2 \right] + \left\| s_\theta(x_t, t, 2d) - s_{\text{target}} \right\|^2
$$

其中 $s_{\text{target}} = \text{stopgrad}\left( \frac{s_\theta(x_t, t, d) + s_\theta(x'_{t+d}, t+d, d)}{2} \right)$

- 第一项：d=0 时的 Flow Matching 目标
- 第二项：d>0 时的自一致性目标

### 4. 优势

- **单阶段训练**：无需预训练+蒸馏的两阶段流程
- **灵活推理预算**：训练后可选择任意步数生成
- **计算效率高**：仅比基础扩散模型多约 16% 计算量

## 实验结果

| 数据集 | 方法 | 128-Step FID | 4-Step FID | 1-Step FID |
|--------|------|--------------|------------|------------|
| CelebA-HQ-256 | Diffusion | 23.0 | 123.4 | 39.7 |
| CelebA-HQ-256 | Flow Matching | 7.3 | 63.3 | 280.5 |
| CelebA-HQ-256 | Consistency Training | 53.7 | 19.0 | 33.2 |
| **CelebA-HQ-256** | **Shortcut Models** | **6.9** | **13.8** | **20.5** |
| ImageNet-256 | Diffusion | 39.7 | 464.5 | 467.2 |
| ImageNet-256 | Flow Matching | 17.3 | 108.2 | 324.8 |
| ImageNet-256 | Consistency Training | 42.8 | 43.0 | 69.7 |
| **ImageNet-256** | **Shortcut Models** | **15.5** | **28.3** | **40.3** |

## 引用

[^src-shortcut-models]: Frans et al. "One Step Diffusion via Shortcut Models". arXiv:2410.12557v3, 2025.