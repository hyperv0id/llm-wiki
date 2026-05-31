---
title: "Scalable Diffusion Models with Transformers (DiT)"
type: source-summary
tags:
  - diffusion
  - transformer
  - generative-model
  - iccv-2023
  - scaling-law
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

**DiT**（Diffusion Transformer）是 William Peebles 和 Saining Xie（UC Berkeley）发表于 **ICCV 2023**（arXiv:2212.09748）的论文[^src-dit]。核心贡献是用 Vision Transformer（ViT）骨架替换扩散模型中的卷积 U-Net，使扩散模型获得 NLP 领域的 scaling law 特性——更大的计算量（Gflops）可预测地带来更好的生成质量。

## 核心贡献

1. **架构统一**：首次证明纯 Transformer（无卷积归纳偏置）可以作为条件扩散模型的主干网络，在隐空间（VAE latent space）中去噪，性能超越同等 Gflops 的 U-Net[^src-dit]
2. **adaLN-Zero 条件注入**：提出自适应 Layer Norm + 零初始化残差缩放的条件注入机制，在所有方案中效果最佳（FID 19.47 vs cross-attention 26.14），且训练极其稳定[^src-dit]
3. **Gflops-based Scaling Law**：12 个变体（S/B/L/XL × p=8/4/2）系统性实验表明，FID 与模型 Gflops 强负相关（r=-0.93），而与参数量不直接相关——减半 patch size（四倍 token 数）≈ 加深加宽网络[^src-dit]
4. **SOTA 生成质量**：ImageNet 256×256 FID=2.27（cfg=1.50），512×512 FID=3.04[^src-dit]

## 方法概述

DiT 运行在 Stable Diffusion 的预训练 VAE 隐空间（f8 下采样，32×32×4）中。输入隐变量经 patchify（p×p patch → 线性投影）变为 token 序列，送入 N 层 Transformer blocks。每层 block 包含多头自注意力 + MLP，条件信息（时间步 t + 类别标签 c）通过 adaLN 机制回归 Layer Norm 的缩放/偏移参数和残差缩放参数（α₁, α₂），α 零初始化使 block 在训练开始时等价于恒等映射[^src-dit]。

扩散框架完全沿用 ADM 和 IDDPM：T=1000 步线性调度、ϵ 预测 L_simple、协方差用完整 VLB 训练。分类器自由引导（CFG）在 s=1.50 时效果最佳[^src-dit]。

配置范围：DiT-S（33M, 12 层 384d）→ DiT-XL（675M, 28 层 1152d），patch size p∈{8,4,2} 对应 T∈{16,64,256} 个 token[^src-dit]。

## 关键发现

- **Model compute > Sampling compute**：DiT-XL/2 用 128 步采样（15.2 Tflops）优于 DiT-L/2 用 1000 步采样（80.7 Tflops），证明增加模型规模比增加采样步数更有效[^src-dit]
- **无需正则化的稳定训练**：所有变体均无 lr warmup、无 dropout、无 weight decay，由 adaLN-Zero 的恒等初始化保证稳定性[^src-dit]
- **VAE 解码器弱依赖**：三种不同 VAE 解码器下 FID 仅差 0.19，说明生成质量来自 Transformer 本身[^src-dit]

## 后续影响

DiT 成为 2023-2024 年扩散架构统一的标志性工作：OpenAI Sora（视频生成）、Stable Diffusion 3、Flux、PixArt-α 均采用 DiT backbone；UrbanDiT 将其引入城市时空预测领域[^src-dit]。

## 局限性

仅支持类别条件生成（无文本条件），严重依赖预训练 VAE 的质量，自注意力的 O(T²) 复杂度对高分辨率仍有挑战[^src-dit]。

[^src-dit]: [[source-dit]]
