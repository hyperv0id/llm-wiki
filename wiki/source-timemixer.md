---
title: "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - multiscale
  - mlp
  - ICLR-2024
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: high
status: active
---

# TimeMixer 论文摘要

## 核心论点

TimeMixer 提出从*多尺度混合*（multiscale-mixing）的视角解决时序预测中复杂时间变异的挑战。核心观察：时间序列在不同采样尺度上呈现不同模式——细粒度尺度反映微观变化（如小时级交通流），粗粒度尺度反映宏观趋势（如年度经济模式）。基于这一观察，TimeMixer 是一个全 MLP 架构，通过 Past-Decomposable-Mixing（PDM）和 Future-Multipredictor-Mixing（FMM）两大模块，分别在历史信息提取和未来预测阶段利用不同尺度序列的互补能力[^src-timemixer]。

## 主要贡献

1. **多尺度混合范式**：超越传统的分解和多周期分析方法，首次将多尺度混合视角引入时序预测——同时利用多尺度序列的*解耦变异*和*互补预测能力*[^src-timemixer]。

2. **PDM（Past-Decomposable-Mixing）**：在多尺度序列上分别对季节性分量和趋势分量进行混合——季节性采用自下而上（fine-to-coarse）聚合微观细节，趋势采用自上而下（coarse-to-fine）注入宏观先验。两组分量独立混合，各取所需[^src-timemixer]。

3. **FMM（Future-Multipredictor-Mixing）**：为每个尺度配备独立预测器，将多个尺度的预测结果求和集成为最终输出，利用不同尺度对细节和趋势的互补预测能力[^src-timemixer]。

4. **全 MLP 架构**：不使用任何注意力机制或卷积，仅依赖线性层和 GELU 激活函数，在效率上显著优于 Transformer 类模型[^src-timemixer]。

## 实验结果

- **长期预测**（8 个 benchmark）：在 Weather 上比 PatchTST 降低 9.4% MSE，Solar-Energy 降低 24.7% MSE，18 个 benchmark 全部取得一致 SOTA[^src-timemixer]。
- **短期预测**：在 PEMS（4 个交通数据集）和 M4（6 个子集，覆盖小时级到年级频率）上均取得最优[^src-timemixer]。
- **效率**：GPU 内存和运行时间均优于 PatchTST，在输入长度 192–3072 范围内一致高效[^src-timemixer]。
- **消融实验**：移除 FMM 会导致 PEMS04 MAE 从 19.21 升至 21.67（+12.8%），同时移除分解和过去混合（消融⑩）则 MAE 升至 24.87（+29.5%），证明每个组件均不可少[^src-timemixer]。
- **混合方向可解释性**：可视化显示季节性混合权重呈现周期性变化，而趋势混合权重以局部分聚合为主，验证了分别使用 bottom-up 和 top-down 设计的必要性[^src-timemixer]。

## 技术细节

- 输入序列通过平均池化下采样为 M 个尺度：$x_m \in \mathbb{R}^{\lfloor P/2^m\rfloor \times C}$。
- PDM 先对每个尺度做 SeriesDecomp（来自 Autoformer），分离 seasonal 和 trend 分量。
- Seasonal Mixing：从 fine 到 coarse（bottom-up），每个尺度的 $s_m$ 通过残差连接接收 $s_{m-1}$ 的信息。
- Trend Mixing：从 coarse 到 fine（top-down），每个尺度的 $t_m$ 从 $t_{m+1}$ 获取宏观指导。
- 所有混合操作均为时间维度的两层线性变换 + GELU。
- FMM 的 L2 loss 基于集成结果计算：$\|x - \sum_m \hat{x}_m\|$。

## 局限性

- 线性混合层在超长输入时参数量增大，不利于移动端部署[^src-timemixer]。
- 仅在时间维度混合，未涉及变量维度跨通道交互[^src-timemixer]。
- 缺少对多尺度混合最优性的理论分析[^src-timemixer]。

[^src-timemixer]: [[source-timemixer]]
