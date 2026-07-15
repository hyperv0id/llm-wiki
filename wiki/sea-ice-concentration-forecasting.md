---
title: "Sea Ice Concentration Forecasting"
type: concept
tags:
  - sea-ice-forecasting
  - climate
  - spatio-temporal
  - deep-learning
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Sea Ice Concentration Forecasting

**海冰密集度（Sea Ice Concentration, SIC）预测** 是气候科学和深度学习交叉领域的重要任务：给定历史 SIC 记录 $Y = \{X_{T-L-1}, ..., X_T\} \in [0\%, 100\%]^{L \times H \times W}$，预测未来 $P$ 步的 SIC 值 $\hat{Y} = \{X_{T+1}, ..., X_{T+P}\}$。SIC 数据来自卫星遥感，每个像素对应 25km×25km 格点，覆盖 pan-Arctic 区域（N:89.8°, S:31.1°, E:180°, W:−180°）[^src-sifusion]。

## 方法演进

| 阶段 | 代表方法 | 特点 |
|------|---------|------|
| 数值/统计模型 | ECMWF SEAS5, 线性 Markov | 依赖高性能计算、参数化不确定 |
| CNN/U-Net | IceNet, SICNet, MT-IceNet | channel-wise fusion 隐式建模时空，单粒度预测 |
| Transformer | IceFormer | 首次将 Transformer 引入 SIC 预测，但仍是单粒度 |
| 多粒度联合 | [[sifusion\|SIFusion]] (NeurIPS 2025) | 同时建模日/周/月三粒度，显式序列建模 |

## 关键数据集

NSIDC G02202 Version 4：自 1978 年 10 月 25 日起的每日 SIC 数据，分辨率 448×304 像素（25km 格点），SIC 值域 0–100%。SIC > 15% 的区域定义为海冰范围（Sea Ice Extent, SIE）[^src-sifusion]。

## 评估指标

- **RMSE / MAE**：SIC 值预测精度
- **R² / NSE**：预测空间模式与真值的吻合度
- **IIEE**（Integrated Ice-Edge Error）：冰缘线误差，分解为高估（O）和低估（U）分量
- **SIEdif**：预测与真值海冰面积的绝对偏差（百万 km²）

## 气候背景

北极海冰受 [[arctic-amplification|北极放大效应]] 驱动加速消融——1979–2021 年间北极升温速度是全球平均的 2–4 倍。SIC 预测对极地生态、沿海社区、航运经济和全球气候系统均有重要影响[^src-sifusion]。

## 相关页面

- [[sifusion]] — SIFusion 多粒度预测模型
- [[multi-granularity-sea-ice-forecasting]] — 多粒度预测范式
- [[independent-spatial-tokenization]] — 空间 tokenization 技术
- [[granularity-variates]] — 粒度 variate 建模
- [[subseasonal-to-seasonal-forecasting]] — S2S 预测

[^src-sifusion]: [[source-sifusion]]
