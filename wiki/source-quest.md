---
title: "QUEST: A Robust Attention Formulation Using Query-Modulated Spherical Attention"
type: source-summary
tags:
  - transformer
  - attention
  - robustness
  - vision-transformer
  -iclr-2026
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# QUEST: A Robust Attention Formulation

QUEST（Query-modulated Spherical Attention）是一种新的注意力机制，通过对键（keys）进行 ℓ2 归一化来解决 Transformer 训练不稳定性和注意力崩溃问题[^src-quest]。

## 核心贡献

### 问题识别
标准注意力机制中，查询（queries）和键（keys）的向量范数可以任意增长，导致注意力 logit 爆炸（attention logit explosion），进而引发训练不稳定[^src-quest]。此外，当数据中存在易于学习的虚假模式（spurious patterns）时，模型会陷入次优解，注意力过度集中于少数 token[^src-quest]。

### 解决方案
QUEST 将键约束到超球面潜在空间（hyperspherical latent space），同时保持查询未归一化，��而允许每个 token 独立控制其注意力分布的锐度[^src-quest]：

$$A = \text{softmax}(Q\bar{K}^T)$$

其中 $\bar{K}$ 是 ℓ2 归一化后的键。这种设计的直觉是：
- 查询范数控制注意力分布的锐度
- 键范数不再"窃取"全局注意力
- 注意力排名完全由查询和键之间的向量对齐（余弦相似度）决定[^src-quest]

## 实验结果

### 训练稳定性
- 标准注意力在 ViT-Base 和 ViT-Large 上训练崩溃
- QUEST 可稳定训练所有规模的 ViT 模型[^src-quest]

### 性能提升（ImageNet）
| 模型 | 标准注意力 | QUEST |
|------|-----------|-------|
| ViT-S/16 | 79.6% | 80.2% |
| ViT-B/16 | 79.0% | 84.9% |
| ViT-L/16 | 72.5% | 78.2% |
| ViT-H/14 | 83.2% | 83.4% |

### 鲁棒性
- 对抗攻击（FGSM、PGD、SPSA、Auto）：QUEST 显著优于标准注意力
- 数据损坏（IN-C）：QUEST 获得更低的 MCE
- 注意力更加均匀分布在相关目标区域[^src-quest]

## 与相关工作的对比

| 方法 | 机制 | 局限性 |
|------|------|--------|
| 标准注意力 | 缩放点积 + softmax | 训练不稳定、注意力崩溃 |
| QKNorm-HS | Q、K 均归一化 + 头级别缩放 | 所有 token 锐度相同 |
| QKNorm-DS | Q、K 均归一化 + 维度级别缩放 | 小模型上性能下降 |
| **QUEST** | **仅 K 归一化** | **每个查询独立控制锐度** |

## 核心洞见

1. **查询范数的作用**：缩放所有注意力 logit，控制注意力分布锐度[^src-quest]
2. **键范数的作用**：控制"全局"注意力的贡献，高范数键会"窃取"注意力[^src-quest]
3. **交叉依赖**：大键范数 → 高注意力 → 大查询范数 → 注意力熵崩溃[^src-quest]
4. **虚假模式问题**：模型容易学习位置+偏差向量的组合，而非真正的语义特征[^src-quest]

## 引用

[^src-quest]: Govindarajan, S., Sidén, P., Roll, J., & Lindsten, F. (2026). QUEST: A robust attention formulation using query-modulated spherical attention. ICLR 2026.