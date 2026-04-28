---
title: "Noise Conditional Score Networks (NCSN)"
type: entity
tags:
  - generative-model
  - score-based
  - diffusion
  - neural-network
created: 2026-04-28
last_updated: 2026-04-28
source_count: 2
confidence: high
status: active
---

# Noise Conditional Score Networks (NCSN)

**NCSN** 是一种基于分数的生成模型，通过估计数据分布的对数密度梯度（分数）来进行生成建模[^src-ncsn]。

## 核心思想

NCSN 的核心创新在于：

1. **多噪声水平扰动**：使用一系列几何递增的高斯噪声 σ₁ > σ₂ > ... > σ_L 扰动数据
2. **条件分数网络**：训练单一神经网络 s_θ(x, σ) 同时估计所有噪声水平下的分数
3. **退火采样**：从高噪声开始逐步降低噪声水平，利用大噪声填充低密度区域

## 架构设计

NCSN 采用为密集预测设计的架构：
- **U-Net 结构**：带跳跃连接的编码器-解码器架构
- **膨胀卷积**：扩大感受野同时保持特征图分辨率
- **条件实例归一化 (CondInstanceNorm++)**：根据���声水平 σ 动态调整归一化参数

## 与其他生成模型的对比

| 模型类别 | 代表方法 | 优点 | 缺点 |
|----------|----------|------|------|
| 基于似然 | VAE, Flows, PixelCNN | 训练稳定 | 架构受限 |
| 对抗训练 | GANs | 样本质量高 | 训练不稳定 |
| **基于分数** | **NCSN, DDPM** | **无需对抗、架构灵活** | **采样慢** |

## 后续发展

NCSN 是扩散概率模型 (Diffusion Probabilistic Models) 的重要前身：
- **[[ddpm|DDPM]]** (Ho et al., 2020)：将分数匹配与扩散过程结合，建立了与去噪得分匹配的数学等价性
- **[[score-based-sde|Score-Based SDE]]** (Song et al., ICLR 2021)：统一了 NCSN 和 DDPM 的 SDE 框架，NCSN 对应 VE SDE

## 相关页面

- [[smld]] — SMLD 框架，NCSN 是其主要实现
- [[score-based-generative-modeling]] — 基于分数的生成建模概念
- [[score-based-sde]] — Score-Based SDE，NCSN 是其框架的一个实例
- [[score-matching]] — 分数匹配技术原理
- [[langevin-dynamics]] — 朗之万动力学采样基础
- [[annealed-langevin-dynamics]] — 退火朗之万动力学采样方法
- [[diffusion-model]] — 扩散模型，NCSN 是其重要前身
- [[ddpm-simplified-training-objective]] — DDPM 的简化训练目标
- [[fokker-planck-equation]] — 福克-普朗克方程，分析收敛性的数学工具

## 引用

[^src-ncsn]: [[source-ncsn]]