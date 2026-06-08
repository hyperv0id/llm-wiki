---
title: "Sundial — A Family of Highly Capable Time Series Foundation Models"
type: source-summary
tags:
  - time-series
  - foundation-model
  - flow-matching
  - generative-model
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 0
confidence: high
status: active
---

# Source: Sundial (ICML 2025)

**Authors**: Yong Liu\*, Guo Qin\*, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang, Jianmin Wang, Mingsheng Long (Tsinghua University, BNRist). **Venue**: ICML 2025, Vancouver. **Code**: [github.com/thuml/Sundial](https://github.com/thuml/Sundial). **Checkpoints**: [huggingface.co/thuml/sundial-base-128m](https://huggingface.co/thuml/sundial-base-128m).

## 核心贡献

Sundial 提出了首个**原生且灵活**的时间序列基础模型系列，基于流匹配 (Flow Matching) 的 TimeFlow Loss 在连续值时间序列上进行生成式预训练，无需离散 tokenization 或参数化先验分布。

### 1. TimeFlow Loss — 生成式训练目标

TimeFlow Loss 是基于流匹配的参数化训练目标，允许自回归 Transformer 在连续值域中学习每个 patch token 的预测分布，直接从中采样[^src-sundial]。与依赖参数化先验（如高斯混合）或离散 tokenization（Chronos、LLMTime）的方法不同，TimeFlow 不预设任何分布形式，从大型异构数据集中灵活学习任意复杂分布。

### 2. Sundial 模型系列

三种规模配置 (Table 5): Small (32M), Base (128M), Large (444M)。架构特性:
- **Patch tokenization**: patch size 16, 实现 patch 级预测以减少自回归步数
- **Decoder-only Transformer**: Pre-LN, RoPE, FlashAttention, KV Cache
- **Context length ≤ 2880**, 预测长度 16 (短时) 或 720 (长时)
- 支持任意长度 lookback 的动态推理

### 3. TimeBench — 万亿级预训练数据集

TimeBench 包含 **1.032 万亿时间点** (Table 4)，来源: Chronos (94B)、ECG (48B)、自收集金融/IoT/医疗数据 (16.3B)、LOTSA (230B)、ERA5 多频段气象数据 (~642B)、合成数据 (0.5B)。覆盖多频率、多长度、多变元数的时序数据。所有评估数据集被排除以进行真正的零样本预测。

## 实验结果

- **TSLib 点预测**: Sundial-Large 在 8 个获胜数中最高（表 9），平均 MSE 较 Time-MoE 降低 7.57%, MAE 降低 4.71%
- **GIFT-Eval (23 datasets)**: MASE 排名第 1，CRPS 排名第 2，超过所有监督模型和先进基础模型（表 2）
- **FEV Leaderboard (27 datasets)**: 零样本性能超过 70% 的分布内训练的统计方法和深度模型，仅作为第二好的零样本预训练模型（仅次于 Chronos），但推理速度 **35× 加速**（图 4-5）
- **GIFT-Eval CRPS**: TimeFlow 在 GIFT-Eval 上 CRPS=0.505，Diffusion=0.534，MSE=0.642

## 关键发现

1. **连续 tokenization 更有效**: 基于 patch 的连续 token 避免了 Chronos 等离散 tokenization 的 OOV 问题和粗粒度预测区间
2. **流匹配优于扩散**: TimeFlow Loss 在 CRPS 上显著优于 diffusion-based 和 MSE 训练目标（表 3, 表 7）
3. **生成式预测对抗 mode collapse**: MSE 优化会导致过平滑预测，而 TimeFlow 的生成式建模可以生成多样化预测（附录 C.1, Figure 14-15）
4. **测试时校准 (Test-Time Calibration)**: 更多采样数 + 更多采样步数 → 更优概率度量，无需重新训练
5. **可扩展性**: 更大模型 → 更低训练损失（图 6），在更大预训练数据上持续提升（表 8）
6. **模型适应**: 在 FEV 上微调进一步提升性能，超越零样本和从头训练

## 局限性

- 高频数据性能不保证（TimeBench 中中低频数据居多）
- 仅使用朴素高斯噪声初始化的采样策略，后处理有很大改进空间
- 单变量预训练无法显式利用变量间相关性
- 多步自回归可能导致过平滑预测

## 相关页面

- [[sundial]] — Sundial 模型家族实体页面
- [[timeflow-loss]] — TimeFlow Loss 技术详解
- [[timebench]] — TimeBench 万亿级预训练数据集
- [[flow-matching]] — Flow Matching 理论基础
- [[generative-time-series-forecasting]] — 生成式时间序列预测
- [[chronos]] — Chronos，离散 tokenization 的对比方法
- [[timesfm]] — TimesFM，单模态确定性预测基础模型
- [[patch-based-tokenization]] — Patch 级别的 tokenization

[^src-sundial]: (self-reference — this is the source page for all [^src-sundial] citations across the wiki)
