---
title: "SIFusion: Multi-Granularity Arctic Sea Ice Forecasting"
type: source-summary
tags:
  - sea-ice-forecasting
  - multi-granularity
  - spatio-temporal
  - transformer
  - climate
created: 2026-07-21
last_updated: 2026-07-21
source_count: 0
confidence: high
status: active
---

# Source: SIFusion (NeurIPS 2025)

**Authors**: Jingyi Xu, Shengnan Wang (equal contribution), Weidong Yang, Keyi Liu, Yeqi Luo (Fudan University), Ben Fei (CUHK), Lei Bai (Shanghai AI Lab). NeurIPS 2025.

## 核心贡献

SIFusion 是首个统一多时间粒度的北极海冰密集度（SIC）预测框架，基于 Swin Transformer V2 空间编码器 + encoder-only Transformer 粒度 variate 建模，同时预测日尺度（7 天）、周平均（8 周）和月平均（6 月）三个粒度。详见 [[sifusion]]。

## 三大创新

1. **独立空间 tokenization**：用共享 Swin Transformer V2 将各粒度 SIC 独立编码为 1D spatial token，解耦 U-Net 式 channel-wise fusion 中空间特征与序列建模的纠缠问题。引入 spatial feature skip connection 保留空间信息。详见 [[independent-spatial-tokenization]]。

2. **多粒度 variate 建模**：将各粒度的独立 spatial token 沿各自时间维拼接形成 granularity variates，通过 encoder-only Transformer 的 attention 作用在 variate token 上捕捉 inter-granularity 相关性、FFN 处理 intra-granularity 序列信息。灵感来自 [[itransformer|iTransformer]] 的 variate attention 思路。详见 [[granularity-variates]]。

3. **Sequential feature skip connection**：通过 cross-attention 在原始 sequential features 和预测特征之间建立跳跃连接，补偿深层编码造成的 intra-granularity 信息损失。

## 实验

数据集为 NSIDC G02202 v4（1978–2023，448×304 格点 × 25km），训练集 1978–2013，验证集 2014–2015，测试集 2016–2023。仅使用 SIC 数据（无大气/海洋辅助变量），在 RMSE、MAE、R²、NSE、IIEE、SIEdif 六个指标上全面超越 IceNet、SICNet、MT-IceNet、IceFormer、SICNet90、ConvLSTM、PredRNN、SimVP 等 baseline。消融实验验证 [[multi-granularity-sea-ice-forecasting|多粒度联合建模]] 带来的跨粒度一致性能提升，以及 [[granularity-variates|granularity variate 注意力机制]] 优于 vanilla Transformer 和 MLP-Mixer 的 backbone 替代方案。

## 局限

仅使用 SIC 数据，未融入气候变量（海温等），在异常年份（如 2022 年海冰反常增长）仍有边界区域偏差。未来可将气候变量和预训练大气/海洋基础模型纳入多粒度视角。

## 相关页面

- [[sifusion]] — SIFusion 模型实体
- [[multi-granularity-sea-ice-forecasting]] — 多粒度海冰预测概念
- [[granularity-variates]] — 粒度 variate 技术
- [[independent-spatial-tokenization]] — 独立空间 tokenization
- [[sea-ice-concentration-forecasting]] — 海冰密集度预测领域
- [[arctic-amplification]] — 北极放大效应气候背景
