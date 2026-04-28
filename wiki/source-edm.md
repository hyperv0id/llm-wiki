---
title: "EDM: Elucidating the Design Space of Diffusion-Based Generative Models"
type: source-summary
tags:
  - diffusion
  - sampling
  - preconditioning
  - nerips-2022
created: 2026-04-28
last_updated: 2026-04-28
source_count: 0
confidence: high
status: active
---

# EDM: Elucidating the Design Space of Diffusion-Based Generative Models

**论文信息**：Karras et al., NeurIPS 2022

## 核心贡献

EDM 系统性地梳理了扩散模型的设计空间，将 VP (DDPM)、VE (NCSN)、iDDPM、DDIM 等变体统一到一个公共框架下[^src-edm]。通过解耦各组件，EDM 提出了三方面的改进：

1. **采样改进**：使用 Heun 二阶方法替代 Euler 方法，配合优化的时间步长调度（ρ=7），大幅降低所需网络评估次数
2. **网络预处理**：提出 cskip/cout/cin 预处理公式，使输入输出保持单位方差，减少误差放大
3. **训练改进**：对数正态噪声分布 + 平衡损失权重 + 非泄漏数据增强

## 主要成果

- CIFAR-10 无条件：FID 1.97（35 NFE）
- CIFAR-10 条件：FID 1.79
- ImageNet-64：FID 1.36（新 SOTA）

## 局限性

- 高分辨率数据集需要重新调整参数
- 随机采样的 churn 参数需要针对每个模型单独调参

## 源文件
[^src-edm]: [[raw/edm]]
