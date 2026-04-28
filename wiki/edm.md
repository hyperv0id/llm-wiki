---
title: "EDM (Elucidating the Design Space)"
type: entity
tags:
  - diffusion
  - sampling
  - training
  - nerips-2022
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# EDM

EDM（Elucidating the Design Space of Diffusion-Based Generative Models）是 Karras 等人在 NeurIPS 2022 发表的论文[^src-edm]，系统性地梳理了扩散模型的设计空间。

## 核心贡献

EDM 将 VP (DDPM)、VE (NCSN)、iDDPM、DDIM 等扩散模型变体统一到一个公共框架下[^src-edm]，提出三方面改进：

1. **采样改进**：Heun 二阶 ODE 求解器 + 优化时间步长调度
2. **网络预处理**：cskip/cout/cin 预处理公式
3. **训练改进**：对数正态噪声分布 + 平衡损失权重 + 非泄漏增强

## 主要成果

- CIFAR-10 无条件：FID 1.97（35 NFE）
- CIFAR-10 条件：FID 1.79
- ImageNet-64：FID 1.36

## 链接

- [[source-edm]] — 论文摘要
- [[diffusion-model]] — 扩散模型基础
- [[heun-sampler]] — Heun 采样器
- [[edm-preconditioning]] — 预处理技术

[^src-edm]: [[source-edm]]