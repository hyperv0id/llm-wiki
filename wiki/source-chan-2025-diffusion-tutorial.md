---
title: "Tutorial on Diffusion Models for Imaging and Vision"
type: source-summary
tags:
  - diffusion-models
  - tutorial
  - imaging
  - vision
created: 2026-04-28
last_updated: 2026-11-22
source_count: 1
confidence: medium
status: active
---

# Tutorial on Diffusion Models for Imaging and Vision

Stanley Chan 于 2024 年撰写的扩散模型教程（arXiv:2403.18103v3），面向对扩散模型研究或应用感兴趣的本研学生，系统覆盖 VAE、DDPM、SMLD、得分匹配、朗之万动力学和福克-普朗克方程。[^src-chan-2025-diffusion-tutorial]

## 核心贡献

- **统一视角**：将 DDPM 和 SMLD 置于同一数学框架下，阐明两者同属扩散过程的不同参数化
- **数学深度**：从朗之万方程出发，推导福克-普朗克方程，证明退火朗之万动力学收敛于数据分布的理论基础
- **逆问题应用**：讨论得分匹配在图像恢复中的潜力，指出训练得分网络本质上等同于训练图像去噪器
- **工程实践**：涵盖训练技巧与 DDIM 等加速方法[^src-chan-2025-diffusion-tutorial]

## 章节结构

1. **变分自编码器（VAE）**：编码器—解码器结构、ELBO 推导与优化方法，作为后续推导的铺垫
2. **DDPM**：前向扩散与反向去噪、噪声预测范式、ELBO 推导，以及 DDIM 加速采样
3. **得分匹配与朗之万动力学**：朗之万采样方程、Stein 分数函数、去噪分数匹配（DSM）、噪声条件分数网络（NCSN），构成 SMLD 基础
4. **SDE 视角**：从迭代算法到 ODE，给出 DDPM 与 SMLD 的 SDE 表示及数值求解器
5. **朗之万与福克-普朗克方程**：布朗运动、Master 方程、Kramers-Moyal 展开与福克-普朗克方程，用概率分布演化解释采样动力学

## 局限性

- 未涵盖 latent diffusion models，对 classifier-free guidance 等工程实践着墨较少
- 侧重理论推导与基础方法，对文本到图像等最新应用覆盖较少，并假设读者具备概率论与微积分基础

## 引用

[^src-chan-2025-diffusion-tutorial]: [[source-chan-2025-diffusion-tutorial]]
