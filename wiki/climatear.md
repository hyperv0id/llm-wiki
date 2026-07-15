---
title: "ClimateAR"
type: entity
tags:
  - climate-forecasting
  - autoregressive-model
  - generative-model
  - probabilistic-forecasting
  - vector-quantization
  - visual-autoregressive
  - multi-scale
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# ClimateAR

**ClimateAR** 是首个将视觉自回归（Visual Autoregressive, VAR）范式引入概率气候预测的生成模型，由浙大、阿里达摩院、中科院大气所在 ICML 2026 提出[^src-climatear]。

## 架构

ClimateAR 由两大组件构成：

1. **对齐 VQ 分词器**：将多变量气候状态 $X_t \in \mathbb{R}^{C \times H \times W}$ 编码为 $K$ 个从粗到细的多尺度残差离散 token map $(r_1, r_2, ..., r_K)$。采用分段码本策略——将残差特征沿通道维拆分为 $N$ 段独立量化，有效码本容量从 $V$ 提升至 $V^N$。浅层使用域特定卷积层处理 CMIP6/ERA5 的分布差异，深层共享码本和网络实现跨域语义对齐[^src-climatear]。

2. **Decoder-only AR Transformer**：从可学习 [S] token 开始逐尺度预测下一尺度 token map $p(r_k \mid r_{<k}, r'_{\le K})$，条件于历史状态的多尺度 token 序列 $(r'_1, ..., r'_K)$。条件控制包含两个层级：
   - **Intra-scale mixed token**：在各尺度将自回归特征 $\tilde{f}_{k-1}$ 与该尺度条件特征 $\tilde{f}'_k$ 拼接，维持尺度内的物理一致性[^src-climatear]。
   - **Hybrid-scale prompt**：通过 cross-attention 将全部尺度条件 token 压缩为连续前缀，使各尺度均能感知跨尺度交互（如 ENSO 大尺度模态对小尺度区域过程的调制）[^src-climatear]。

## 训练流程

1. **VQ-VAE 训练**：在 CMIP6 + ERA5 + ORAS5 混合数据上训练，损失包含 VQ loss + SSIM 感知项[^src-climatear]。
2. **AR 预训练**：在 CMIP6 模拟数据上以 cross-entropy 分类目标预训练 AR Transformer[^src-climatear]。
3. **AR 微调**：在 ERA5/ORAS5 真实再分析数据上微调，采用噪声增强 teacher-forcing 缓解 exposure bias[^src-climatear]。

## 关键性能

- 12 变量 × 1–10 月 lead time，平均 ACC 提升 37.56% vs 最强 baseline[^src-climatear]
- ENSO 指数（Niño3.4）概率预测能力强，集合预报能有效捕获观测 ENSO 变率[^src-climatear]
- 零样本 CMIP6→ERA5 直接迁移已超越所有数据驱动 baseline[^src-climatear]

## 相关页面

- [[source-climatear]] — 论文源摘要
- [[mixed-scale-conditioning]] — 混合尺度条件控制机制
- [[weather-foundation-model]] — 天气基础模型
- [[generative-time-series-forecasting]] — 生成式时间序列预测
- [[subseasonal-to-seasonal-forecasting]] — S2S 预测

[^src-climatear]: [[source-climatear]]
