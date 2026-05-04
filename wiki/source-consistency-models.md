---
title: "Consistency Models"
type: source-summary
tags:
  - diffusion-models
  - fast-inference
  - one-step-generation
  - icml-2023
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Consistency Models

**Consistency Models** 是由 Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever 于 2023 年发表在 ICML 的论文（arXiv:2303.01469），提出了一类新的生成模型，支持快速单步生成，同时保留多步采样和零样本编辑能力[^src-consistency-models]。

## 核心贡献

### 1. 单步生成能力

Consistency Models 通过学习 PF ODE（概率流 ODE）轨迹上任意点到起点的映射，实现单步生成：
- 输入：纯噪声 $x_T \sim \mathcal{N}(0, T^2 I)$
- 输出：$x_0 = f_\theta(x_T, T)$

### 2. 自一致性约束

核心思想：同一轨迹上的点应映射到相同起点
$$
f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \text{对于同一轨迹}
$$

### 3. 两种训练模式

| 模式 | 描述 | 优势 |
|------|------|------|
| **Consistency Distillation (CD)** | 蒸馏预训练扩散模型 | 可利用已有扩散模型知识 |
| **Consistency Training (CT)** | 独立训练 | 无需预训练模型，作为独立生成模型 |

### 4. 零样本编辑能力

无需显式训练即可执行：
- 图像修复 (inpainting)
- 上色 (colorization)
- 超分辨率 (super-resolution)
- 笔触引导编辑 (stroke-guided editing)

## 实验结果

| 数据集 | 步数 | FID | IS |
|--------|------|-----|-----|
| CIFAR-10 | 1 | 3.55 | — |
| CIFAR-10 | 2 | 2.93 | — |
| ImageNet 64×64 | 1 | 6.20 | — |
| ImageNet 64×64 | 2 | 4.70 | — |

## 与其他方法的对比

- **vs Progressive Distillation**: 一致性模型在单步/少步采样上优于渐进蒸馏
- **vs GANs**: 作为非对抗生成模型，性能可比甚至超越 GANs
- **vs 传统扩散模型**: 保留多步采样和零样本编辑能力

## 引用

[^src-consistency-models]: Yang Song et al. "Consistency Models". ICML 2023. arXiv:2303.01469