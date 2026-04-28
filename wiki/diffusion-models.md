---
title: "扩散模型 (Diffusion Models)"
type: concept
tags:
  - generative-models
  - diffusion
  - deep-learning
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# 扩散模型

**扩散模型**是一类通过逐步向数据添加噪声再逆转过程来生成新样本的生成模型。核心思想来自非平衡热力学。[^src-tutorial]

## 两类主流方法

### DDPM (Denoising Diffusion Probabilistic Models)

- 正向过程：固定方差的高斯噪声逐步添加
- 逆向过程：学习去噪网络
- 训练目标：ELBO 下界，等价于去噪得分匹配

### SMLD (Score Matching Langevin Dynamics)

- 正向过程：多尺度高斯噪声扰动
- 逆向过程：朗之万动力学采样
- 训练目标：多尺度得分匹配

## SDE 统一视角

连续时间下，扩散过程由随机微分方程描述：

$$
dx = f(x, t) dt + g(t) dw
$$

- **DDPM**：对应方差爆炸型 (VE) SDE
- **SMLD**：对应方差保持型 (VP) SDE

逆向过程对应反向时间 SDE，福克-普朗克方程描述概率密度的演化。[^src-tutorial]

## 应用领域

- 图像生成（DALL-E, Stable Diffusion 的底层技术）
- 音频合成
- 药物分子设计
- 逆问题求解（去模糊、超分辨率、修复）

## 挑战与未来方向

- 采样速度慢（需要数十到数百步）
- 与物理世界的物理一致性
- 信息取证与深度伪造检测

知识蒸馏和快速 ODE 求解器是加速采样的主要方向。[^src-tutorial]

## 引用

[^src-tutorial]: [[source-chan-2025-diffusion-tutorial]]
