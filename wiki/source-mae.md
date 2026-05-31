---
title: "Masked Autoencoders Are Scalable Vision Learners (MAE)"
type: source-summary
tags:
  - self-supervised-learning
  - masked-autoencoding
  - vision-transformer
  - representation-learning
  - cvpr-2022
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# Masked Autoencoders Are Scalable Vision Learners (MAE)

**MAE** (Masked Autoencoder) 是由 Kaiming He、Xinlei Chen、Saining Xie、Yanghao Li、Piotr Dollár 和 Ross Girshick（FAIR）于 CVPR 2022 发表的视觉自监督学习论文，arXiv 首次发布于 2021 年 11 月[^src-mae]。论文提出了一种简单且可扩展的掩码自编码器方法，通过随机掩码大量图像 patch 并重建像素来学习强大的视觉表示[^src-mae]。

## 核心贡献

MAE 基于两个核心设计[^src-mae]：

1. **非对称编码器-解码器架构**：编码器仅处理可见 patch（不使用 mask token），轻量解码器从 latent 表示和 mask token 重建完整图像。编码器仅处理约 25% 的 token，训练加速 3× 以上。

2. **高掩码比例（75%）**：极高的掩码比例消除了图像的空间冗余，创建了一个需要整体理解的、非平凡的 self-supervisory task。

## 核心方法

**Patch 化与随机掩码**：输入图像 224×224 切分为 14×14=196 个非重叠 16×16 patch。随机均匀采样 75% 的 patch 进行掩码，仅保留 49 个 patch 输入编码器[^src-mae]。

**编码器**：标准 ViT 架构，但仅处理可见 patch（无 mask token）。通过 shuffle-unshuffle 机制实现高效实现——shuffle token 列表，取前 25% 送入编码器，编码后在解码器端 unshuffle 恢复空间顺序，无需任何稀疏操作[^src-mae]。

**轻量解码器**：默认 8 层 Transformer blocks、512-d 宽度、16 头注意力。每 token 计算量仅为编码器的 9%[^src-mae]。

**训练目标**：仅在掩码 patch 上计算 MSE 损失。默认重建 per-patch 归一化像素值[^src-mae]。

**极简数据增广**：仅需随机裁剪（RandomResizedCrop），无需 color jittering（反而有害）。随机掩码本身就是最强的"增广"[^src-mae]。

## 关键实验结果

### ImageNet-1K 分类（fine-tuning）

| 模型 | MAE 精度 | 对比 |
|------|---------|------|
| ViT-B | 83.6% | scratch 82.3% |
| ViT-L | 85.9% | scratch 82.6%, MoCo v3 84.1% |
| ViT-H (224) | 86.9% | scratch 83.1% |
| ViT-H (448) | **87.8%** | 纯 IN1K 数据 SOTA |

### 关键消融实验

- **掩码比例 75% 最优**：fine-tuning 84.9%，linear probing 73.5%。40%-80% fine-tuning 均鲁棒[^src-mae]。
- **编码器不带 mask token**：linear probing 从 59.6% 跳至 73.5%（+13.9pp），fine-tuning 84.2% → 84.9%（+0.7pp），FLOPs 降至 1/3.3[^src-mae]。
- **解码器深度**：1→8 层 linear probing 从 65.5% 单调升至 73.5%，fine-tuning 对各深度均鲁棒（84.3-84.9%）[^src-mae]。
- **重建目标**：per-patch 归一化像素最优（85.4% fine-tuning），与 dVAE token（85.3%）统计无差异[^src-mae]。
- **无增广仍可工作**：仅 center crop 达 84.0%，加 color jitter 反降至 84.3%[^src-mae]。

### 迁移学习

- **COCO 目标检测**：ViT-L 53.3 APbox，比监督预训练高 4.0 APbox[^src-mae]。
- **ADE20K 语义分割**：ViT-L 53.6 mIoU，比监督预训练高 3.7 mIoU[^src-mae]。

### 与对比学习的定性差异

MAE 的 linear probing（73.5%）低于 MoCo v3（77.6%），但微调 1 个 block 即追平，微调 4 个 block 反超 2.6pp[^src-mae]。这揭示了 MAE 学习的特征非线性强、更适合 fine-tuning——与对比学习优化"线性可分性"不同，MAE 学的是"全局结构理解"[^src-mae]。

## 局限性

- 解码器仅在预训练使用，微调时丢弃——存在结构性算力浪费[^src-mae]。
- Linear probing 不如对比学习方法（天然弱点）[^src-mae]。
- 高掩码比例在"简单"数据（如固定视角医学影像）上可能不适用[^src-mae]。
- 像素重建目标缺乏严格的生成理论支撑[^src-mae]。
- 预训练与微调之间的分辨率切换需要位置编码插值[^src-mae]。

## 历史意义

MAE 打破了 2021 年前视觉自监督领域由对比学习垄断的局面，将前沿从"如何设计更好的增广/负样本"拉回到"掩码→重建"的简单范式[^src-mae]。论文发表后，masked image modeling 在 NeurIPS/CVPR 的自监督论文占比从近零暴涨至三成以上[^src-mae]。后续工作包括 SimMIM、MaskFeat、ConvNeXt V2、[[videomae]]、MAE+CLIP 多模态融合等。

[^src-mae]: [[source-mae]]
