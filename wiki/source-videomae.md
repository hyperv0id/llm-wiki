---
title: "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
type: source-summary
tags:
  - masked-autoencoding
  - self-supervised-learning
  - video-understanding
  - vision-transformer
  - spatiotemporal-modeling
  - neurips-2022
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

# VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

**VideoMAE** 由 Zhan Tong、Yibing Song、Jue Wang 和 Limin Wang（南京大学 / Tencent AI Lab / Shanghai AI Lab）于 NeurIPS 2022 发表（arXiv: 2203.12602）[^src-videomae]。论文将 MAE 的掩码自编码器范式扩展到视频域，提出 tube masking 与极高掩码比例（90%-95%）两个关键设计，使 vanilla ViT 无需任何外部监督数据即可在视频数据本身上进行高效预训练[^src-videomae]。

## 问题背景

视频 Transformer（ViViT、TimeSformer）严重依赖大规模图像监督预训练（如 ImageNet-21K），从视频数据从零训练效果极差（ViT-B on SSV2 from scratch 仅 32.6%）[^src-videomae]。视频自监督预训练（SSVP）被对比学习垄断，但对比方法依赖强数据增广和 3D-CNN backbone，计算昂贵且在小数据集上效果有限[^src-videomae]。

## 核心方法

### Tube Masking

生成 2D 空间随机 mask，沿时间轴复制到所有帧——同一空间位置在所有时间步要么全保留、要么全 mask。动机：堵死时间信息泄露——若用 plain random masking，mask 区域在相邻帧可能对应位置未被 mask，模型可"抄邻居"完成重建，无需高层时空理解[^src-videomae]。Tube masking 90% vs random masking 90%：SSV2 +1.3%（69.6 vs 68.3）[^src-videomae]。

### 极高掩码比例（90%-95%）

视频同时具有空间冗余和时间冗余，信息密度远低于图像（图像 75%）和语言（BERT 15%）。掩码比例从 50% 到 90%，fine-tuning 精度持续上升——真正的跃迁在 75%→90%（SSV2 68.0% → 69.6%），95% 仍接近最优[^src-videomae]。

### Cube Embedding 与时间降采样

每个时空 token 为 2×16×16 cube（2 帧跨度），通过 3D 卷积投影。时间降采样（Kinetics τ=4，SSV2 τ=2）去除相邻帧冗余，让可见 token 包含有意义的运动差异[^src-videomae]。

### 不对称编码器-解码器

编码器仅处理可见 token（90% mask → 仅 ~10% token），浅层解码器（4 层 384-d vs MAE 的 8 层 512-d）处理全量 token + 可学习 [[mask]] token。重建目标：per-cube 归一化像素值的 MSE loss，仅在被 mask 的 cube 上计算[^src-videomae]。

## 关键实验结果

### 主实验 vs from scratch 和 MoCo v3（Table 2）

| 数据集 | 训练量 | scratch | MoCo v3 | VideoMAE | vs MoCo |
|--------|--------|---------|---------|----------|---------|
| K400 | 240k | 68.8 | 74.2 | **80.0** | +5.8 |
| SSV2 | 169k | 32.6 | 54.2 | **69.6** | +15.4 |
| UCF101 | 9.5k | 51.4 | 81.7 | **91.3** | +9.6 |
| HMDB51 | 3.5k | 18.0 | 39.2 | **62.6** | +23.4 |

数据集越小，VideoMAE 相对对比学习的优势越大[^src-videomae]。

### 效率对比

800 epoch 预训练仅需 19.5 小时（64 V100），而 MoCo v3 300 epoch 需 61.7 小时——3.2× 训练加速[^src-videomae]。加速根源：编码器每步仅处理 5-10% 的 token，joint space-time attention 复杂度从 O(1568²) 降至 O(157²)[^src-videomae]。

### SSV2 / K400 SOTA（Table 6/7）

VideoMAE ViT-L 无需任何外部数据：SSV2 74.3%（32 帧 75.4%），超越所有依赖 ImageNet-21K / DALL-E 的方法[^src-videomae]。ViT-H K400 86.6%（320² 87.4%），超越 ViViT-H（JFT-300M 有监督预训练）的 84.9%[^src-videomae]。

## 三大核心发现

1. **信息密度阶梯**：语言 15% mask → 图像 75% → 视频 90%+，模态信息密度越低，所需 mask 比例越高[^src-videomae]。

2. **数据效率惊人**：仅 3.5k 视频的 HMDB51 上 VideoMAE 达 62.6%，MoCo v3 仅 39.2%（gap 23.4pp）。因为 mask 本身即天然增广——每次迭代随机选择不同 tube 位置，小数据集产生近乎无限的 mask 组合[^src-videomae]。

3. **数据质量 > 数据数量**：SSV2 域内 42k 视频预训练（68.7%）> K400 域外 240k 视频预训练（68.5%）——域相关性比数据规模更重要[^src-videomae]。

## 局限性

- 解码器微调时丢弃，存在结构性算力浪费[^src-videomae]。
- Kinetics 上时序建模增益有限（+0.5% vs random masking），因为 K400 偏场景识别而非运动理解[^src-videomae]。
- 默认仅处理 16 帧（约 1-2 秒），长视频建模能力受限[^src-videomae]。
- Linear probing 弱于对比学习（38.9% vs MoCo v3 33.7% 略有优势，但 fine-tuning 后差距巨大），特征非线性程度高[^src-videomae]。
- 仅用 RGB 模态，未利用音频和文本[^src-videomae]。

## 后续影响

VideoMAE 与 ST-MAE 一起，将视频自监督主流范式从对比学习（CVRL/ρBYOL/ρMoCo）拉回到 masked autoencoding。后续发展包括 VideoMAE V2（扩展到 128 帧）、InternVideo（masked reconstruction + video-text alignment）等[^src-videomae]。

[^src-videomae]: [[source-videomae]]
