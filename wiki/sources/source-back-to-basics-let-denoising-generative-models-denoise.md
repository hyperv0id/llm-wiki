---
title: "Back to Basics: Let Denoising Generative Models Denoise"
type: source-summary
tags:
  - diffusion
  - x-prediction
  - manifold
  - pixel-space
  - mit
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: high
status: active
---

# Back to Basics: Let Denoising Generative Models Denoise

> **arXiv: 2511.13720, MIT, 2026**
> Tianhong Li, Kaiming He

## 核心问题

现代扩散模型并不真正"去噪"——它们预测噪声（ε-prediction）或流速度（v-prediction），而非直接预测干净图像。这篇论文追问：**在流形假设下，预测干净数据和预测含噪量在根本上是否不同？如果是，那么让网络直接预测干净数据（x-prediction）能否彻底改变高维空间扩散模型的设计？**[^src-2511-13720]

## 核心洞察

### 流形假设下的预测目标差异

根据流形假设，自然数据（如图像）位于高维空间中的低维流形上，而噪声 ε 或速度 v = x − ε 本质上散布在整个高维空间中（off-manifold）。因此：

- **x-prediction**：网络只需保留低维流形信息，过滤掉噪声。有限容量网络即可胜任。
- **ε-/v-prediction**：网络需要在高维空间中保留噪声的全部信息，需要高容量。当观测维度超过网络容量时，会灾难性失败。[^src-2511-13720]

### 九种组合的系统枚举

论文系统分析了损失空间和预测空间的九种组合（Tab. 1），证明所有九种都是合法的生成器，但只有 x-prediction 能在高维下正常工作。[^src-2511-13720]

### 玩具实验验证

用 d 维底层数据嵌入 D 维观测空间（d << D）的玩具实验表明：当 D 增大时，只有 x-prediction 能产生合理结果；ε-/v-prediction 在 D=16 时已开始退化，在 D=512 时完全崩溃（使用 256-dim MLP）。[^src-2511-13720]

## JiT: Just image Transformers

论文提出 **JiT**（Just image Transformers）——直接将标准 Vision Transformer (ViT) 应用于像素块，使用 x-prediction + v-loss（Tab. 1(3)(a)）。核心设计：

- 无 tokenizer、无预训练、无额外损失
- 使用大块大小（16×16 或 32×32 像素），序列长度固定为 16×16
- adaLN-Zero 时间/类别条件化
- 使用 SwiGLU、RMSNorm、RoPE、qk-norm 等通用 Transformer 改进
- 可选的 in-context class conditioning（多 token 前置）
- bottleneck 线性嵌入层（默认 128-d）反而有益[^src-2511-13720]

### 关键发现

| 发现 | 意义 |
|------|------|
| **x-prediction 是高维下的唯一选择** | JiT-B/16 (768-d per patch)，只有 x-prediction 在所有损失空间下工作；ε-/v-prediction 全部灾难性失败（FID 372+ vs 8.62） |
| **bottleneck 设计有益** | 将 768-d 嵌入压缩到 32~512-d 反而提升 FID，最优在 128-d |
| **增加 hidden units 不必要** | 即使 patch 维度高达 12288 (1024×1024, JiT-B/64)，x-prediction 仍然工作 |
| **噪声水平调整不解决根本问题** | 即使优化 t 分布，ε-/v-prediction 仍灾难性失败 |
| **EDM pre-conditioner 同样失败** | 因为 cskip ≠ 0 意味着网络输出不是纯 x-prediction |

### 实验结果

| 模型 | 分辨率 | FID-50K | 参数量 | Gflops |
|------|--------|---------|--------|--------|
| JiT-B/16 | 256 | 4.37 (200ep) / 3.66 (600ep) | 131M | 25 |
| JiT-L/16 | 256 | 2.79 / 2.36 | 459M | 88 |
| JiT-H/16 | 256 | 2.29 / 1.86 | 953M | 182 |
| JiT-G/16 | 256 | 2.15 / 1.82 | 2B | 383 |
| JiT-B/32 | 512 | 4.64 / 4.02 | 133M | 26 |
| JiT-L/32 | 512 | 3.06 / 2.53 | 462M | 89 |
| JiT-H/32 | 512 | 2.51 / 1.94 | 956M | 183 |
| JiT-G/32 | 512 | 2.11 / 1.78 | 2B | 384 |
| JiT-B/64 | 1024 | 4.82 | 141M | 30 |

所有 JiT 模型在相同序列长度（16×16）下运行，不同分辨率仅改变 patch 维度。计算成本几乎不随分辨率增长。[^src-2511-13720]

### 与现有方法的对比

JiT 在所有像素空间扩散方法中实现了最佳的计算效率-FID 平衡：
- JiT-G/16 (383 Gflops) vs SiD2 UViT/2 (653 Gflops) — 更少计算，更好 FID（1.82 vs 2.12）
- 不依赖任何预训练组件（tokenizer、VGG 感知损失、DINOv2 表示对齐）
- 对比 latent 方法（DiT、SiT、DDT）需要 SD-VAE tokenizer + VGG 感知损失

## 意义

1. **对扩散模型设计的根本反思**：预测目标的选择不仅影响损失加权，更决定网络能否有效利用其容量。在高维空间中，x-prediction 不是选项之一，而是必要条件。[^src-2511-13720]
2. **自包含范式的推动**：JiT 展示了"Diffusion + Transformer"在原生数据上自包含工作的可能性，无需领域特定的 tokenizer 设计。这对于 tokenizer 难以设计的科学应用（蛋白质、分子、天气）尤为重要。[^src-2511-13720]
3. **与 ELF 的呼应**：ELF（同实验室的连续 DLM 工作）也采用 x-prediction，验证了该设计在不同模态间的普适性。

## 局限性

- 实验仅在 ImageNet 上进行，未验证在更大规模、更多样化数据集上的表现
- 不使用额外损失或预训练，但论文附录表明分类辅助损失可进一步提升（JiT-L: 2.79→2.50），说明仍有优化空间
- 大模型（H/G）需要 dropout 和 early stopping 来防止过拟合
- 未探索条件生成（如 text-to-image）场景

[^src-2511-13720]: [[source-back-to-basics-let-denoising-generative-models-denoise]]