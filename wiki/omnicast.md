---
title: "OmniCast"
type: entity
tags:
  - weather-forecasting
  - diffusion
  - s2s
  - masked-generative
  - neurips-2025
  - vae
created: 2026-07-21
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# OmniCast

**OmniCast** 是 Nguyen 等人（UCLA, UCI, Argonne National Laboratory, AI2）在 NeurIPS 2025 提出的掩码潜扩散模型，用于跨时间尺度的概率性天气预测[^src-omnicast]。它统一了中期预报（≤15 天）和次季节到季节（S2S，2–6 周）预测，论文报告其在 S2S 尺度达到 SOTA，且比同类概率方法快 10–20 倍[^src-omnicast]。

## 核心架构

OmniCast 由两个组件构成：

1. **连续 VAE 编码器**：将 69 变量天气状态（温度、风、位势高度、湿度等，含 13 个气压层）压缩为 8 × 16（或 45 × 90，取决于分辨率）的连续潜变量网格，空间压缩比 16×。选择连续 VAE 而非 VQ-VAE 是因为离散化在天气数据上导致 ~3938:1 的极端压缩比，重建误差严重[^src-omnicast]。

2. **掩码生成式 Transformer**：基于 MAE 的 encoder-decoder 架构，训练时随机掩码 50–100% 的未来 token（每个 token 是潜空间中的一个连续向量），通过一个轻量 MLP 扩散头对每个 token 独立建模条件分布。推理时以迭代解码生成完整未来序列——每轮随机选择部分掩码 token 通过扩散采样解码，直至全部揭示[^src-omnicast]。

## 关键设计选择

| 选择 | OmniCast 方案 | 替代方案（被否决） |
|:-----|:-------------|:-----------------|
| 潜空间 | 连续 VAE（D=1024） | VQ-VAE（重建误差 2–3× 更高） |
| 生成范式 | 掩码生成 + 扩散头 | 纯自回归（累积误差严重） |
| 训练序列 | 44 帧 × 24h 间隔 | 短序列/短间隔（S2S 性能差） |
| 辅助损失 | 前 10 帧加权 MSE | 全部帧 MSE（S2S 尺度反效果） |
| 解码策略 | 跨时空全随机 | 逐帧自回归/随机帧（欠离散） |
| 扩散温度 | τ = 1.3 | τ < 1（欠离散）或 τ > 1.3（RMSE 退化） |

## 实验结果

- **S2S 预测（ChaosBench）**：day 10 后确定性指标与 ECMWF-ENS 并列前二，day 15 后概率指标（CRPS/SSR）超越 ECMWF-ENS；SDIV/SRES 物理一致性优于所有 DL 方法；短中期逊于 Stormer，S2S 尺度与 ClimaX 可比[^src-omnicast]。
- **中期预报（WeatherBench2）**：与 IFS-ENS 可比、略逊于 GenCast，推理快 10–20×（A100 上 29s vs GenCast TPUv5 480s）[^src-omnicast]。
- **稳定性**：可生成 100 年稳定 rollout 而无物理崩溃[^src-omnicast]。
- **效率来源**：① 45 × 90 潜网格 vs 721 × 1440 原始网格（256× 减少）；② 仅一次 Transformer 前向传播 + 轻量 MLP 扩散头（而非每扩散步都走完整网络）[^src-omnicast]。

## 与相关工作的关系

- 同一 UCLA 组的前序工作：[[source-climax|ClimaX]]（ICML 2023，首个天气基础模型）、Stormer（缩放 Transformer 中期预报）[^src-omnicast]
- 掩码生成框架来自 CV 领域的 [[mae|MAE]]（He et al., 2022）、MaskGIT（Chang et al., 2022）、MAR（Li et al., 2024）
- 概率天气预测对比：GenCast（扩散模型，更慢）、NeuralGCM（论文报告训练成本 10 天/128 TPUv5e）[^src-omnicast]
- S2S 其他方案：[[cirt|CirT]]（直接预测平均值）、Fuxi-S2S（预测日平均值，不可比）[^src-omnicast]

## 相关页面

- [[subseasonal-to-seasonal-forecasting]] — S2S 预测的问题定义与方法论
- [[source-omnicast]] — 论文摘要
- [[latent-diffusion-models]] — 潜扩散模型通用范式
- [[masked-generative-modeling]] — 掩码生成建模概念
- [[source-climax|ClimaX]] — 同组前序天气基础模型
- ChaosBench — S2S 评估基准

[^src-omnicast]: [[source-omnicast]]
