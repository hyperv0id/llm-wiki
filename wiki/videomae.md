---
title: "VideoMAE (Video Masked Autoencoder)"
type: technique
tags:
  - masked-autoencoding
  - self-supervised-learning
  - video-understanding
  - vision-transformer
  - spatiotemporal-modeling
  - pretraining
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# VideoMAE (Video Masked Autoencoder)

**VideoMAE**（Video Masked Autoencoder）是 Tong 等人于 NeurIPS 2022 提出的视频自监督预训练方法，将 [[mae|MAE]] 的掩码自编码器范式扩展到视频域[^src-videomae]。核心思想：在视频上做更大更聪明的 masking——先用极高比例（90-95%）mask 时空 cube，再用 tube masking 堵死时间维度的信息泄露捷径，迫使 vanilla ViT 从极少数可见 token 中学习全局时空理解[^src-videomae]。

## 为什么视频需要不同的掩码设计

VideoMAE 基于对视频数据三个特性的诊断[^src-videomae]：

1. **时间冗余**：相邻帧语义变化极慢，若 mask 比例和图像一样（75%），模型可从相邻帧对应位置"抄答案"，无需高层时空理解[^src-videomae]。
2. **时间相关性导致信息泄露**：plain random masking 下，某空间位置在 frame t 被 mask、在 frame t+1 可能未被 mask——像素几乎一样，模型只需浅层时间匹配[^src-videomae]。
3. **计算瓶颈**：joint space-time attention 复杂度 O((T×N)²)，但极高 mask 比例反而可将编码器输入 token 降至 5-10%，复杂度崩塌式下降[^src-videomae]。

## 核心设计

### 设计一：Tube Masking — 堵死信息泄露

生成 2D 空间随机 mask → 沿时间轴复制到所有帧 → 每个空间位置在所有时间步要么全 mask、要么全保留[^src-videomae]。

数学表示：$\mathbb{I}[p_{x,y,\cdot} \in \Omega] \sim \text{Bernoulli}(\rho_{\text{mask}})$，不同时间 $t$ 共享相同值[^src-videomae]。

三种 mask 策略对比（Table 1b）[^src-videomae]：

| 策略 | mask 比例 | SSV2 | K400 |
|------|----------|------|------|
| tube masking | 90% | **69.6** | **80.0** |
| random masking | 90% | 68.3 | 79.5 |
| frame masking | 87.5% | 61.5 | 76.5 |

差距在运动中心数据集 SSV2 上尤为显著（+1.3%）——证明 tube masking 真正提升的是时空联合推理[^src-videomae]。

### 设计二：90%-95% 极高掩码比例 — 信息密度阶梯

VideoMAE 最反直觉的发现：最优 mask 比例从图像的 75% 跃升至视频的 90%-95%[^src-videomae]。

| 模态 | 方法 | 最优 mask | 信息密度 |
|------|------|----------|---------|
| 语言 | BERT | 15% | 高（离散语义） |
| 图像 | MAE | 75% | 中（空间冗余） |
| 视频 | VideoMAE | 90-95% | 低（空间+时间冗余） |

每级冗余都比上一级大一个数量级——mask 比例不是随意调的，而是由模态内在信息密度决定的[^src-videomae]。

SSV2 上 mask 比例演变（Figure 3）[^src-videomae]：
- 50%: ~65.2%
- 75%: ~68.0%（75%-90% 是真正的跃迁）
- **90%**: **~69.6%**（峰值）
- 95%: ~69.3%（略降但仍在高位）
- 98%: ~67.8%

75%-90% 是跳跃区间——75% 时可见 token 的时间邻居仍提供"免费答案"，只有 mask 到"邻居也全被遮住"的程度，模型才被迫学习真正的时空理解[^src-videomae]。

### 设计三：Cube Embedding + 时间降采样

**Cube Embedding**：每个 token 覆盖 2×16×16 时空体（2 帧×16×16 像素），通过 3D 卷积（kernel 2×16×16, stride 2×16×16）投影到 D 维。总 token 数 = T/2 × H/16 × W/16，相比逐帧 2D patch embedding 减半[^src-videomae]。

**时间降采样**：从原始视频取 t 帧连续帧 → 隔 τ 帧取一帧 → 得 T 帧。Kinetics τ=4（16 帧涵盖 ~2.1 秒），SSV2 τ=2（动作更快更短，~1.1 秒）[^src-videomae]。

### 设计四：不对称编码器-解码器

- **编码器**：vanilla ViT joint space-time attention，仅处理 visible token（~5-10% 总量）。90% mask + ViT-B 16 帧 → 约 157 visible token（总量 1568），self-attention 复杂度从 O(1568²) 降至 O(157²)，约 100× 加速[^src-videomae]。
- **解码器**：4 层 384-d（MAE 用 8 层 512-d），因为 tube masking 已堵死"低级特征重建"捷径，不需深解码器作计算缓冲[^src-videomae]。
- **解码器深度消融**：1 层 68.5%，2 层 69.2%，4 层 69.6%（最优），8 层 69.3%（下降）——视频场景下过深反而有害[^src-videomae]。
- **重建目标**：per-cube 归一化像素的 MSE loss，仅在被 mask 的 cube 上计算[^src-videomae]。

## 数据效率

VideoMAE 最震撼的结果在小数据集上[^src-videomae]：

| 数据集 | 视频数 | Scratch | MoCo v3 | VideoMAE | vs MoCo |
|--------|-------|---------|---------|----------|---------|
| HMDB51 | 3.5k | 18.0 | 39.2 | **62.6** | **+23.4** |
| UCF101 | 9.5k | 51.4 | 81.7 | **91.3** | +9.6 |
| SSV2 | 169k | 32.6 | 54.2 | **69.6** | +15.4 |
| K400 | 240k | 68.8 | 74.2 | **80.0** | +5.8 |

数据集越小，VideoMAE 相对对比学习的优势越大。原因：对比学习需大规模正负样本对，masked autoencoding 靠随机 mask 产生"无限增广"——3.5k 视频每次迭代选不同 tube 位置 mask[^src-videomae]。

## 数据质量 vs 数据数量

SSV2 域内 42k 视频预训练（68.7%）> K400 域外 240k 视频预训练（68.5%）——**域相关性比数据规模更重要**[^src-videomae]。这是自监督预训练中第一个干净证明"质量>数量"的实验[^src-videomae]。

## 计算效率

800 epoch 仅 19.5 小时（64 V100），MoCo v3 300 epoch 需 61.7 小时——3.2× 训练加速，同时 epoch 数多 2.7×[^src-videomae]。加速根源：编码器每步仅处理 ~10% token[^src-videomae]。

## SOTA 结果（无需外部数据）

| 数据集 | 方法 | Backbone | 精度 |
|--------|------|---------|------|
| SSV2 | VideoMAE | ViT-L (32fr) | **75.4** |
| K400 | VideoMAE | ViT-H (320²) | **87.4** |
| UCF101 | VideoMAE | ViT-B | **91.3** |
| HMDB51 | VideoMAE | ViT-B | **62.6** |

VideoMAE ViT-L SSV2 75.4% 超越 BEVT（需 IN-1K+K400+DALL-E 多源预训练）的 70.6%[^src-videomae]。ViT-H K400 87.4% 超越 ViViT-H（JFT-300M 监督预训练）的 84.9%[^src-videomae]。

## 局限性

- 解码器微调时丢弃，存在结构性算力浪费[^src-videomae]。
- Linear probing 不强（38.9% SSV2 vs fine-tuning 69.6%），特征非线性程度远高于图像 MAE[^src-videomae]。
- 仅 16 帧，长视频建模受限[^src-videomae]。
- 仅 RGB 模态，未利用音频和文本[^src-videomae]。
- Tube masking 在运动摄像机场景下可能非最优（同一空间位置在不同帧对应不同物理位置）[^src-videomae]。
- 缺乏严格理论：为什么"随机 mask + 重建像素"能导出好表示？论文给直觉但无证明[^src-videomae]。

## 相关页面

- [[source-videomae]] — VideoMAE 论文摘要
- [[mae]] — MAE（图像掩码自编码器），VideoMAE 的直接方法论来源
- [[std-mae]] — STD-MAE，时空解耦掩码自编码器（时序预测应用）
- [[patchtst]] — PatchTST，masked patch autoencoder 用于时序预测

[^src-videomae]: [[source-videomae]]
