---
title: "DDPM"
type: entity
tags:
  - diffusion-models
  - generative-model
  - nips-2020
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# DDPM (Denoising Diffusion Probabilistic Models)

**DDPM** 是扩散模型领域里程碑式的工作，由 Jonathan Ho, Ajay Jain, Pieter Abbeel 于 2020 年发表在 NeurIPS。首次证明了扩散模型能够生成与 GAN 相媲美的高质量图像。

## 核心创新

1. **简化训练目标**：使用 $L_{\text{simple}}$ 预测噪声而非预测均值
2. **与得分匹配的等价性**：建立了扩散模型与去噪得分匹配之间的数学联系
3. **高质量样本**：CIFAR-10 达到 IS 9.46, FID 3.17

## 技术特点

- **T=1000 步扩散**：从纯噪声逐步去噪
- **U-Net 架构**：带自注意力和组归一化
- **噪声调度**：$\beta_t$ 从 $10^{-4}$ 线性增加到 $0.02$

## 后续发展

- **[[score-based-sde|Score-Based SDE]]** (Song et al., ICLR 2021)：将 DDPM 重新解释为 VP SDE 的离散化

## 相关页面

- [[diffusion-model]] — 扩散模型概念
- [[ncsn|NCSN]] — DDPM 的重要前身
- [[score-based-sde|Score-Based SDE]] — 统一 SMLD 和 DDPM 的 SDE 框架
- [[score-based-generative-modeling]] — 基于分数的生成模型
- [[annealed-langevin-dynamics]] — 退火朗之万动力学

## 引用

[^src-ddpm]: [[source-ddpm]]