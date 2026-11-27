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
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs

真实预测里常见不对等因果：便宜的外生变量（气象、水质）单向驱动内生目标（湖温、作物产量），而主流模型要么对称地全对全建模，要么用注意力融合外生信息并付出二次开销，逐通道独立的线性模型则丢掉跨变量信息。XLinear 给每个内生变量配一个可学习全局 token：内生序列先带它沿时间轴门控滤波，token 再沿变量轴吸收外生信息，最后拼接送入线性预测头，全程无注意力。作者报告其在 7 个公开基准与 5 个真实外生数据集上多数最优或次优，训练开销显著更低[^src-xlinear]。

## 核心做法

全流程只由 RevIN 归一化、线性嵌入与「两层 MLP + Sigmoid 逐元素门控」一种标准算子组合构成，未引入新底层算子；论文把同一门控沿时间轴与变量轴各部署一次，称时间门控模块（TGM）与变量门控模块（VGM）[^src-xlinear]：

- **时间门控**：全局 token 与内生嵌入沿时间维拼接，门控筛选关键时序特征并写入 token[^src-xlinear]。
- **变量门控**：token 再与外生嵌入沿变量维拼接，同构门控把外生信息压进 token[^src-xlinear]。
- **预测头**：单个全连接层把「增强后内生序列 + 全局 token」的拼接直接映射为多步预测，各通道独立输出[^src-xlinear]。

## 证据

1. **定位与主结论**：7 公开基准加 5 农业水文外生数据集多数最优或次优，训练开销显著低于注意力类基线（作者报告至少快 39.3%）[^src-xlinear]。
2. **真实涨点归因**：消融中仅留纯时序通路（ES）掉点最狠，仅留全局 token（GT）反优于纯时序，涨点主要来自外生信息通路；未消融门控激活与模块顺序[^src-xlinear]。
3. **证据缺口与负结果**：862 变量 Traffic 落后 iTransformer 与 TimeXer，高维下单 token 容量不足；基线对比集中于 lookback-96 口径[^src-xlinear]。

## 局限

论文自述：高维场景使 VGM 输入维数膨胀，作者将压缩 VGM 输入维数列为后续工作[^src-xlinear]。

## 相关页面

- **同槽位互斥** → [[source-timexer|TimeXer]]：同用全局 token 桥接内生与外生，但对方靠 patch 自注意力 + 变量级交叉注意力，XLinear 以双 MLP 门控换线性开销。
- **同槽位互斥** → [[crosslinear|CrossLinear]]：同属线性复杂度外生预测，但对方以 1D 卷积跨相关残差嵌入建模变量关系，无全局 token 通道。
- **同槽位互斥** → [[itransformer|iTransformer]]：对方在变量维开 $N\times N$ 注意力，在 862 变量的 Traffic 上优于 XLinear，是其容量上界对照。
- **同构/特例** → [[tide|TiDE]]、[[channel-independence|Channel Independence]]：同为无注意力轻量路线，TiDE 用残差 MLP 编解码器吸收协变量；XLinear 预测头逐通道独立输出，与 Channel Independence 仅在输出端一致。

出处：Chen et al., *XLinear*（AAAI 2026），代码公开于 GitHub[^src-xlinear]。

[^src-xlinear]: [[source-xlinear]]
