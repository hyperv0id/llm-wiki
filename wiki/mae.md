---
title: "MAE (Masked Autoencoder)"
type: technique
tags:
  - masked-autoencoding
  - self-supervised-learning
  - vision-transformer
  - representation-learning
  - pretraining
created: 2026-05-31
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# MAE (Masked Autoencoder)

**MAE**（Masked Autoencoder，掩码自编码器）是 He 等人于 2022 年（ArXiv 2021）提出的视觉自监督预训练方法。核心思想极其简单：随机遮住图像的大部分 patch（75%），然后让模型从可见部分重建被遮住的像素[^src-mae]。

## 为什么视觉需要不同的掩码策略

MAE 论文对"为什么 BERT 式掩码自编码在视觉上不 work"给出了三重诊断[^src-mae]：

1. **信息密度差异**：语言是高信息密度的人造信号，遮住 15% 的词就够难了。但图像是自然信号，空间冗余极高——相邻 patch 高度相关，简单的插值就能填补缺失区域。低比例的掩码（<50%）创建的 pretext task 太简单，模型不需要学习高层语义[^src-mae]。

2. **架构错配已解但设计未优化**：Vision Transformer（ViT）出现前，CNN 难以自然地集成 mask token。ViT 扫清了架构障碍，但之前的 mask image encoding 方法（包括 ViT 自己的实验、BEiT）编码器仍然处理全部 token（包括 mask token），导致计算量不随掩码比例下降[^src-mae]。

3. **解码器角色不同**：NLP 中 decoder 预测的词本身就是语义实体。视觉中 decoder 要重建像素（低级信号），如果解码器设计不当，低级重建需求会"污染"编码器的 latent representation[^src-mae]。

## 核心设计

### 设计一：编码器不看 mask token

MAE 做了一个看似"不用想"但此前没人坚决执行的决定：把 mask token 完全从编码器中删除[^src-mae]。

实现方式：生成所有 patch 的 token → random shuffle → 取前 25%（49/196 for ViT-L/16）→ 送入编码器 → 编码后追加共享的可学习 mask token → unshuffle 恢复原空间顺序 → 送入解码器[^src-mae]。

效果：
- **FLOPs 降至 1/3.3**：75% mask → 编码器只处理 25% 的 token → 自注意力复杂度从 O(196²) 降至 O(49²)[^src-mae]
- **Train-wall-clock 加速 2.8-4.1×**（取决于模型大小和解码器深度）[^src-mae]
- **Linear probing +13.9pp**（59.6% → 73.5%）——mask token 造成严重的训练/推理分布偏移[^src-mae]

### 设计二：75% 超高掩码比例

MAE 中 75% 掩码比例是最关键的发现之一[^src-mae]：

| 掩码比例 | Fine-tuning | Linear Probing |
|---------|------------|----------------|
| 10% | 83.2 | 54.6 |
| 30% | 83.0 | 61.7 |
| 50% | 84.5 | 69.9 |
| **75%** | **84.9** | **73.5** |
| 80% | 84.7 | 71.8 |
| 90% | 84.7 | 67.0 |

Fine-tuning 在 40%-80% 间均鲁棒（标准差仅 0.6%），但 linear probing 在 75% 达到尖峰——证实高掩码比例迫使编码器学习全局结构理解而非局部纹理插值[^src-mae]。

和 BERT 的 15% 对比：语言信息密度高，15% 足以创建有意义的学习任务。图像空间冗余极高，需要 75% 的极端掩码来消除冗余[^src-mae]。

### 设计三：轻量解码器作为"计算缓冲"

解码器默认配置：8 层 Transformer block、512-d 宽度（编码器为 24 层 1024-d ViT-L），每 token 计算量仅为编码器的 9%[^src-mae]。

解码器深度对 linear probing 影响显著（1 层 65.5% → 8 层 73.5%），但对 fine-tuning 几乎无影响（84.3-84.9%）[^src-mae]。这揭示了自编码器的深层规律：深层解码器充当"计算缓冲"——它消化低级纹理重建需求，保护编码器 latent 的语义抽象性[^src-mae]。

### 重建目标：Normalized Pixel

MAE 默认重建 per-patch 归一化的像素值（每个 patch 减均值除标准差），MSE 仅在 masked patch 上计算[^src-mae]。

全尺度/全任务对比中，normalized pixel 与 BEiT 的 dVAE token 差距 ≤0.2%（统计无差异），但 normalized pixel 不需要额外的 250M 图像预训练 tokenizer[^src-mae]。Per-patch normalization 额外贡献 +0.5% fine-tuning 精度——因为它保留了高频信息（纹理、边缘、局部对比度），这些是模型建立精确空间理解的线索[^src-mae]。

PCA 目标（仅低频）反而更差——说明高频细节对视觉表示学习并非噪声[^src-mae]。

## 与对比学习的分道扬镳

MAE ViT-L linear probing 73.5% vs MoCo v3 ViT-L 77.6%（输 4.1pp），但 fine-tuning 84.9% vs 84.1%（赢 0.8pp）[^src-mae]。

更深层的现象在部分微调实验中显现[^src-mae]：

| 微调 Blocks | MAE | MoCo v3 | 差距 |
|-----------|-----|---------|------|
| 0 (linear) | 73.5 | 77.6 | -4.1 |
| 1 | 81.0 | ~80.8 | +0.2 |
| 4 | ~84.2 | ~81.6 | +2.6 |
| 24 (full) | 84.9 | 84.1 | +0.8 |

仅微调 1 个 block，MAE 就追平 MoCo v3；微调 4 个 block，反超 2.6pp[^src-mae]。这揭示了两种范式的根本差异：
- **对比学习**优化"类间分离"→ 天然奖励线性可分性
- **MAE** 优化"从碎片重建整体"→ 学习全局结构理解，但特征天然非线性

MAE 的特征在非线性变换下更强大——这暗示人类视觉更接近 MAE 而非对比学习：人类不是通过"这张图和那张图像不像"学会识别猫，而是在观看世界的过程中从部分推断整体[^src-mae]。

## 迁移学习优势

| 任务 | 指标 | 监督预训练 | MAE | 提升 |
|------|------|-----------|-----|------|
| COCO 检测 | APbox (ViT-L) | 49.3 | 53.3 | +4.0 |
| ADE20K 分割 | mIoU (ViT-L) | 49.9 | 53.6 | +3.7 |

检测和分割这类需要空间理解的任务上，自监督 MAE 大幅超越有标签的分类预训练——说明 MAE 的"从部分重建整体"与密集预测任务在认知上同构[^src-mae]。

## 相关页面

- [[source-mae]] — MAE 论文摘要
- [[masked-generative-modeling]] — 掩码生成建模，MAE 掩码思想向生成任务的延伸（OmniCast 天气预测应用）
- [[std-mae]] — STD-MAE，时空解耦掩码自编码器（时空预测应用）
- [[videomae]] — VideoMAE，视频掩码自编码器（MAE 的视频扩展）
- [[dit]] — DiT，继承 ViT/MAE patchify + Transformer 设计的扩散模型架构
- [[gpt-st]] — GPT-ST，将 MAE 范式适配到时空图数据的预训练框架（NeurIPS 2023）
- [[contrastive-learning]] — 对比学习范式
- [[spatiotemporal-mirage]] — 时空幻象问题（STD-MAE 上下文）

[^src-mae]: [[source-mae]]
