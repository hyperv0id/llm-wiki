---
title: "Tutorial on Diffusion Models for Imaging and Vision"
type: source-summary
tags:
  - diffusion-models
  - tutorial
  - imaging
  - vision
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Tutorial on Diffusion Models for Imaging and Vision

Stanley Chan 于 2024 年撰写的扩散模型教程，面向图像和视觉领域，系统覆盖 DDPM、SMLD、得分匹配、朗之万动力学和福克-普朗克方程等核心内容。[^src-tutorial]

## 核心贡献

- **统一视角**：将 DDPM 和 SMLD 置于同一数学框架下，阐明两者同属扩散过程的不同参数化
- **数学深度**：从朗之万方程出发，推导福克-普朗克方程，证明退火朗之万动力学收敛于数据分布的理论基础
- **逆问题应用**：讨论得分匹配在图像恢复中的潜力，指出训练得分网络本质上等同于训练图像去噪器[^src-tutorial]

## 章节结构

1. **引言**：扩散模型发展概述
2. **DDPM**：前向扩散与反向去噪，ELBO 推导
3. **得分匹配与朗之万动力学**：SMLD 基础
4. **SDE 视角**：连续时间扩散过程
5. **福克-普朗克方程**：概率分布演化的偏微分方程描述

## 局限性

- 未涵盖 latent diffusion models
- 对 classifier-free guidance 等工程实践着墨较少

## 引用

[^src-tutorial]: [[source-chan-2025-diffusion-tutorial]]
