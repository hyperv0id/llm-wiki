---
title: "Tutorial on Diffusion Models for Imaging and Vision"
type: source-summary
tags:
  - diffusion-models
  - generative-models
  - tutorial
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Tutorial on Diffusion Models for Imaging and Vision

本文是 Stanley Chan 编写的扩散模型教程，发表于 arXiv (arXiv:2403.18103v3)，面向对扩散模型研究或应用感兴趣的本研学生[^src-chan-diffusion-tutorial]。

## 核心内容

教程系统介绍了扩散模型的数学基础与实现原理，涵盖以下主题：

### 1. 变分自编码器 (VAE)
- 编码器-解码器结构
- 证据下界 (ELBO) 推导
- 优化方法

### 2. 去噪扩散概率模型 (DDPM)
- 前向扩散过程
- 逆向生成过程
- 噪声预测范式
- DDIM 加速采样

### 3. 分数匹配朗之万动力学 (SMLD)
- 朗之万采样方程
- Stein 分数函数
- 去噪分数匹配 (DSM)
- 噪声条件分数网络 (NCSN)

### 4. 随机微分方程 (SDE)
- 从迭代算法到 ODE
- DDPM 和 SMLD 的 SDE 表示
- 数值求解器

### 5. 朗之万与福克-普朗克方程
- 布朗运动与朗之万方程
- Master 方程
- Kramers-Moyal 展开
- 福克-普朗克方程

## 核心贡献

1. **统一视角**：将 DDPM 和 SMLD 统一到 SDE 框架下
2. **数学严谨**：提供完整的定理证明与推导
3. **物理直觉**：通过福克-普朗克方程解释采样动力学
4. **工程实践**：涵盖训练技巧与加速方法

## 局限性

- 侧重理论，对最新应用（如文本到图像）覆盖较少
- 假设读者具备概率论与微积分基础

## 参考文献

- Kingma & Welling (2014): VAE 原创工作
- Ho et al. (2020): DDPM
- Song & Ermon (2019, 2020): SMLD 系列工作
- Vincent (2011): 去噪分数匹配

---

[^src-chan-diffusion-tutorial]: [[source-chan-diffusion-tutorial]]