---
title: "JiT — Just image Transformers"
type: entity
tags:
  - diffusion
  - transformer
  - pixel-space
  - mit
  - x-prediction
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: high
status: active
---

# JiT — Just image Transformers

JiT（Just image Transformers）是由 MIT 的 Tianhong Li 和 Kaiming He 提出的像素空间扩散模型，核心思想是将标准 Vision Transformer (ViT) 直接应用于原始像素块，使用 **x-prediction**（预测干净图像）而非 ε-/v-prediction。[^src-2511-13720]

## 核心设计

- **架构**：标准 ViT（plain Transformer），无卷积、无 U-Net、无层级设计
- **预测目标**：x-prediction（直接输出干净图像），损失使用 v-loss
- **patch 大小**：与分辨率成正比，保持序列长度固定为 16×16
  - 256² → p=16 (768-d per patch)
  - 512² → p=32 (3072-d per patch)
  - 1024² → p=64 (12288-d per patch)
- **条件化**：adaLN-Zero + 可选的 in-context class conditioning（32 tokens）
- **通用改进**：SwiGLU、RMSNorm、RoPE、qk-norm
- **bottleneck 嵌入**：线性嵌入层使用 128-d bottleneck（B/L）或 256-d（H/G）
- **训练**：logit-normal t 分布 (µ=-0.8, σ=0.8)，50-step Heun 采样器，CFG + CFG interval

关键算法（Tab. 1(3)(a)）：
1. 训练：$x_\theta = \text{net}(z_t, t)$, $v_\theta = (x_\theta - z_t)/(1-t)$, $\mathcal{L} = \|v_\theta - v\|^2$
2. 采样：$x_\theta = \text{net}(z_t, t)$, $v_\theta = (x_\theta - z_t)/(1-t)$, $z_{t+\Delta t} = z_t + \Delta t \cdot v_\theta$[^src-2511-13720]

## 关键发现

JiT 的核心贡献不是新架构，而是揭示了一个被忽视的根本问题：**x-prediction 与 ε-/v-prediction 在流形假设下存在根本差异**。在高维空间（patch 维度 > 网络隐藏维度）中，只有 x-prediction 能工作：

- ε-prediction：FID 372+（灾难性失败）
- v-prediction：FID 96+（灾难性失败）
- x-prediction：FID 8.62（合理），通过通用改进降至 4.37，扩展到 G 模型降至 1.82[^src-2511-13720]

## 模型变体

| 模型 | 分辨率 | 参数量 | Gflops | FID-50K (600ep) |
|------|--------|--------|--------|-----------------|
| JiT-B/16 | 256² | 131M | 25 | 3.66 |
| JiT-L/16 | 256² | 459M | 88 | 2.36 |
| JiT-H/16 | 256² | 953M | 182 | 1.86 |
| JiT-G/16 | 256² | 2B | 383 | 1.82 |
| JiT-B/32 | 512² | 133M | 26 | 4.02 |
| JiT-L/32 | 512² | 462M | 89 | 2.53 |
| JiT-H/32 | 512² | 956M | 183 | 1.94 |
| JiT-G/32 | 512² | 2B | 384 | 1.78 |
| JiT-B/64 | 1024² | 141M | 30 | 4.82 |

## 意义

1. **自包含生成范式**：无需 tokenizer、预训练、额外损失——纯 diffusion + Transformer
2. **计算效率**：计算成本几乎不随分辨率增长（仅 patch embedding 变化），JiT-G/32 在 512² 上仅需 384 Gflops
3. **通用性**：设计思路可推广到其他 tokenizer 难以设计的领域（蛋白质、分子、天气）
4. **与 [[elf|ELF]] 的呼应**：同实验室的连续 DLM 也采用 x-prediction，验证了跨模态普适性

[^src-2511-13720]: [[source-back-to-basics-let-denoising-generative-models-denoise]]