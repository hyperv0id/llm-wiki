---
title: "SADI"
type: entity
tags:
  - diffusion-models
  - time-series
  - data-imputation
  - self-attention
  - partial-blackout
  - aaai-2025
created: 2026-06-08
last_updated: 2026-08-29
source_count: 2
confidence: high
status: active
---

# SADI (Self-Attention Diffusion Model for Time-series Imputation)

**SADI** 是由 Oregon State University 提出的双阶段自注意力扩散模型，用于多元时间序列插补，发表于 AAAI 2025[^src-sadi]。核心创新：引入 partial blackout 这一更通用的缺失模式，并通过显式建模特征依赖（FDE）和时序相关性（GTA）及双阶段加权插补机制，在四个真实世界数据集上全面超越 [[csdi|CSDI]]、BRITS、SAITS 和 MICE[^src-sadi]。

## 核心架构

SADI 的去噪函数 $\epsilon_\theta$ 包含三个关键组件[^src-sadi]：

### 1. [[feature-dependency-encoder|FDE (Feature Dependency Encoder)]]

1-D 膨胀卷积（核 1×3，每层膨胀率 +1）+ self-attention on feature dimension。膨胀卷积提取局部模式，递增的膨胀率扩展感受野；特征维度的自注意力捕获时间感知的特征间全局依赖。$N_{FDE}$ 层堆叠。

### 2. [[gated-temporal-attention|GTA (Gated Temporal Attention)]]

受 DiffWave/WaveNet 残差块启发，用 self-attention 替代膨胀卷积建模非局部时间依赖。$N_{GTA}$ 层堆叠 + 时间维度的位置编码 + GLU 激活。所有层的 skip connections 聚合为第一阶段插补 $\epsilon_1$。

### 3. 双阶段插补与加权组合

第二阶段 GTA 块以第一阶段插补结果 + 重新引入的原始噪声数据为输入进行精炼，产生 $\epsilon_2$。最终输出[^src-sadi]：

$$\tilde{W}_L = \text{sigmoid}(\text{linear}(\text{concat}(W_L, M_0^{co})))$$

$$\epsilon_\theta = (1 - \tilde{W}_L) \odot \epsilon_1 + \tilde{W}_L \odot \epsilon_2$$

注意力权重 $W_L$ 与缺失掩码 $M_0^{co}$ 拼接后经 FFN+sigmoid 产生动态权重。

## 训练策略

两种策略[^src-sadi]：

| 策略 | 方法 | 特点 |
|------|------|------|
| **SADI-RM** | 纯随机缺失训练 | 每 epoch 随机掩码观测值作为插补目标 |
| **SADI-MPB** | [[mixed-partial-blackout-training|混合部分停电训练]] | RM 预收敛后交替引入 partial blackout 块 |

SADI-MPB 在所有数据集上超越 SADI-RM，证明显式暴露于 partial blackout 模式的训练价值[^src-sadi]。

## 性能

在 AgAID、Air Quality、Electricity、NACSE 四个数据集上，SADI 在 MSE 和 CRPS 双指标均超越所有基线（CSDI, BRITS, SAITS, MICE）[^src-sadi]。关键优势：

- **高维数据集高效**：SADI 在 Electricity (370 特征) 和 NACSE (352 特征) 上无需降低通道数，而 CSDI 因 GPU 显存限制被迫减少通道数[^src-sadi]
- **低缺失特征数优势更大**：当部分停电仅涉及少数特征时，FDE 显式建模特征间相关性的优势最为突出[^src-sadi]

## 与相关方法的关系

| 方法 | 特征依赖 | 时间依赖 | 缺失模式 | 阶段数 |
|------|---------|---------|---------|--------|
| [[csdi\|CSDI]] | 分离式 Feature Transformer | 分离式 Time Transformer | 随机缺失 | 单阶段 |
| [[ssd-ts\|SSD-TS]] | CMB (Mamba) | BAM (Mamba+Attention) | 随机缺失 | 单阶段 |
| [[lscd\|LSCD]] | 频谱条件编码器 | Transformer | 随机缺失 | 单阶段 |
| **SADI** | **FDE (卷积+注意力)** | **GTA (自注意力+GLU)** | **部分停电** | **双阶段加权** |

## 综述归类

Wang & Du 等人的 MTSI 综述将 SADI 归为生成式-扩散类插补方法，概述其为"利用自注意力机制捕获病患间相似性以插补缺失值"的相似度感知扩散模型；综述 Table 1 将其缺失机制标注为 MCAR/MAR/MNAR——在综述收录的 33 个方法中，仅 SADI 与 supnotMIWAE（标注 MNAR）覆盖 MNAR 机制[^src-mts-imputation-survey]。该机制标注属综述作者的二手归类；本页"partial blackout"口径来自原论文，两套口径分立。

## 局限性

- 双阶段架构增加计算量[^src-sadi]
- 代码未公开 (anonymous.4open.science)[^src-sadi]
- 未拓展到 forecasting 任务

## 关联页面

- [[source-sadi]] — 源文件摘要
- [[partial-blackout]] — 部分停电缺失模式
- [[feature-dependency-encoder]] — FDE 技术详解
- [[gated-temporal-attention]] — GTA 技术详解
- [[two-stage-imputation]] — 双阶段插补范式
- [[mixed-partial-blackout-training]] — MPB 训练策略
- [[csdi]] — CSDI，扩散插补的 baseline 方法
- [[ssd-ts]] — SSD-TS，Mamba backbone 替代方案
- [[cofill]] — CoFILL，双流时空扩散插补
- [[fence]] — FENCE，动态反馈引导扩散插补
- [[saits]] — SAITS，FDE/GTA 的注意力设计灵感来源
- [[mts-imputation-taxonomy]] — MTSI 综述的分类框架，SADI 归为生成式-扩散类且 Table 1 标注覆盖 MCAR/MAR/MNAR

[^src-sadi]: [[source-sadi]]
[^src-mts-imputation-survey]: [[source-mts-imputation-survey]]
