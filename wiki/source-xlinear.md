---
title: "XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs"
type: source-summary
tags:
  - time-series-forecasting
  - exogenous
  - linear-models
  - mlp
  - aaai-2026
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs

**来源**：Xinyang Chen、Huidong Jin、Yu Huang、Zaiwen Feng（华中农业大学信息学院；CSIRO）；AAAI 2026[^src-xlinear]。

[[source-timexer|TimeXer]] 用分块自注意力与变量级交叉注意力融合外生变量，但注意力矩阵带来二次计算与显存开销。XLinear 把跨变量信息压缩进一个可学习的全局摘要向量：先让内生序列带着该向量沿时间轴做一次门控滤波，再让该向量沿变量轴吸收外生序列信息，最后直接拼接两者通过线性层输出预测，全程不引入注意力，将复杂度压至线性级[^src-xlinear]。

## 去黑话算子配方与路线差异

模型全程由常规归一化、线性投影与两层门控 MLP 组成，未引入改变数学结构的新底层算子[^src-xlinear]：

- **归一化与线性嵌入**：用 `RevIN` 归一化内生序列 $X \in \mathbb{R}^{T\times M}$ 与外生序列 $E \in \mathbb{R}^{T\times C}$，经共享线性层映射至隐维 $d$，整体保持 $O(Td)$ 线性复杂度[^src-xlinear]。
- **时间与变量双门控交互**：时间门控模块（TGM）将内生特征与可学习全局 token $X_{\text{glob}}$ 拼接，经 $\sigma(W_2\phi(W_1 X_{\text{endo\_tok}}))\odot X_{\text{endo\_tok}}$ 逐时间步门控滤波；变量门控模块（VGM）再将全局 token 沿变量维与外生特征拼接并做同构的 `MLP + Sigmoid` 门控聚合，最后把吸收外生信息的全局 token 与内生特征沿时间维拼接，接单层线性头（FC）直连预测[^src-xlinear]。
- **有向关系对照**：与 [[source-timexer|TimeXer]] 属**同槽位互斥**——同样以全局 token 桥接内生与外生，但把 patch 自注意力与交叉注意力全部替换为双 MLP 门控以换取线性开销；与 [[crosslinear|CrossLinear]] 同处 $O(T)$ 线性外生阵营并属**同槽位互斥**（1D 卷积跨相关残差嵌入 vs 全局 token 双门控）；相对 [[itransformer|iTransformer]] 则将 $N\times N$ 变量注意力塌缩为单全局 token 通道，并与 [[tide|TiDE]]、[[channel-independence|Channel Independence]] 共同构成外生轻量化对照[^src-xlinear]。

## 证据

1. **定位与主结论**：7 公开与 5 农业/水文集多数最优或次优，Weather 快 $\ge 39.3\%$、显存低约 $43.7\%$[^src-xlinear]。
2. **真实涨点归因**：去全局 token（GT）或外生通路（ES）均掉点（Weather-96：$0.149\to$ GT $0.153$/ES $0.175$），主要涨点来自外生通路；未做 TGM/VGM 顺序、门控激活或参数共享消融[^src-xlinear]。
3. **证据缺口与负结果**：862 维 Traffic Many-to-Many 落后 TimeXer（$0.432$ vs $0.428$）与 iTransformer，单 token 门控去噪容量不足，高维扩展性待提升[^src-xlinear]。

## 相关页面

- [[source-timexer]] — 注意力外生融合桥梁（同槽位互斥）
- [[crosslinear]] — 1D 卷积跨相关线性外生预测（同槽位互斥）
- [[itransformer]] — 变量维 $N\times N$ 自注意力（复现与高维容量对照）
- [[tide]] — 通道独立残差 MLP 协变量预测（复现与机制对照）
- [[channel-independence]] — 通道独立 vs 单全局 token 外生混合折中对照

[^src-xlinear]: [[source-xlinear]]
