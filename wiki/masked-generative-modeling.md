---
title: "Masked Generative Modeling"
type: concept
tags:
  - generative-modeling
  - masked-autoencoder
  - diffusion
  - image-generation
  - video-generation
created: 2026-07-21
last_updated: 2026-08-30
source_count: 1
confidence: low
status: active
---

# Masked Generative Modeling

**掩码生成建模（Masked Generative Modeling）** 是一种通过随机掩码部分 token 并训练模型从可见 token 重建掩码 token 的生成范式，最早由 MaskGIT（Chang et al., CVPR 2022）和 MAGVIT（Yu et al., CVPR 2023）在计算机视觉中提出，后被 [[omnicast|OmniCast]] 成功应用于天气预测[^src-omnicast]。

## 核心原理

与自回归生成（逐 token 串行解码）不同，掩码生成建模采用**迭代并行解码**[^src-omnicast]：

1. **训练**：随机掩码 50–100% 的 token，训练模型预测被掩码 token 的分布（在连续空间中用扩散头，在离散空间中用 softmax）
2. **推理**：从全掩码序列开始，按预定义 schedule 每轮并行解码随机子集的掩码 token，直到全部揭示

## 优势

- **长程依赖**：双向注意力使模型能同时关注序列中所有位置，而非仅左侧上下文
- **无累积误差**：并行生成避免了自回归模型中的错误逐步放大问题。在 [[omnicast|OmniCast]] 中，跨时空的联合随机解码显著缓解了自回归误差累积问题[^src-omnicast]
- **可控多样性**：随机解码顺序引入额外随机性，产生更分散的集合预报

## 连续 vs 离散掩码生成

| 维度 | 离散（MaskGIT/MAGVIT） | 连续（OmniCast/MAR） |
|:-----|:-------------------|:--------------------|
| 潜空间 | VQ-VAE 离散码本 | 连续 VAE 潜变量 |
| 分布建模 | Cross-entropy / softmax | 扩散模型（per-token MLP 头） |
| 重建质量 | 受码本大小限制 | 更高（无量化损失）[^src-omnicast] |
| 适用场景 | 图像/视频（3 通道） | 多变量科学数据（100+ 通道） |

在天气数据上，连续 VAE 的压缩比为 ~100:1，而 VQ-VAE 需 ~3938:1（100 变量 × 32 bit / 13 bit），后者重建误差高 2–3×[^src-omnicast]。这使连续掩码生成成为气象预测的关键技术选择。

## 与其他生成范式的对比

| 范式 | 解码方式 | 典型代表 | 天气预测适用性 |
|:-----|:--------|:--------|:-------------|
| 自回归 | 逐 token 串行 | PanguWeather, [[graphcast|GraphCast]] | 中期可，S2S 累积误差严重 |
| 完整扩散 | 从噪声迭代 | GenCast | 精度高，但推理慢（50 步完整前向）[^src-omnicast] |
| **掩码生成** | **迭代并行解码** | **OmniCast** | **精度与速度兼得** |
| 流匹配 | ODE 积分 | CoGenCast, Sundial | 一步采样快，天气应用待探索（无来源，课程对比） |

## 相关页面

- [[omnicast]] — 掩码生成 + 扩散头在天气预测的应用
- [[source-omnicast]] — OmniCast 论文摘要
- [[mae]] — 掩码自编码器（掩码生成的前序预训练范式）
- [[latent-diffusion-models]] — 潜扩散模型
- [[ddpm]] — 扩散建模基础

[^src-omnicast]: [[source-omnicast]]
