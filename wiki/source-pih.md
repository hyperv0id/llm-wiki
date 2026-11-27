---
title: "PIH: Enhancing the Maximum Effective Window for Long-Term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - long-term-forecasting
  - context-length
  - information-bottleneck
  - mamba
  - transformer
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# PIH: Enhancing the Maximum Effective Window for Long-Term Time Series Forecasting

直觉上回看窗越长可用的历史越充分，但 Transformer 时序模型拉长窗口后误差常不降反升：长历史夹带的噪声冗余会摊薄注意力并诱发过拟合。论文将「再加长窗口也不再改善误差的临界长度」定义为最大有效窗口（MEW），并用两个可插拔模块从信息滤波与架构分工两端推长它。挂到 [[patchtst|PatchTST]] 后的完整模型，作者报告取得 SOTA 且算力开销反而更低[^src-pih]。

## 核心做法

PIH（Patch-IBF-HTM）在 PatchTST 的 patch 化流程中串入两个与骨干解耦、可移植的模块[^src-pih]。变分信息瓶颈滤波（IBF）以 `MLP + Sigmoid` 给各 patch 打重要度分，经 `Gumbel-Sigmoid` 采出可微的伯努利掩码，对判冗余的 patch 段注入同均值方差的高斯噪声以切断信息流动；训练目标为预测误差叠加一项压缩惩罚的变分上界，绕开指数级子序列枚举[^src-pih]。Transformer-Mamba 混合切分（HTM）先以 [[mamba|Mamba]] 做长程选择性滤波，再按 interval（交错分组，保全局趋势）或 block（连续分段，保局部形态）把滤波后序列缩成多组短段，交 Transformer 处理局部依赖[^src-pih]。

## 证据

1. **主结论**：$L=1024$ 下在 7 数据集整体优于 PatchTST、DLinear、S-Mamba 等，开销约为其 $1/2$–$1/3$；PatchTST 延窗至 1024 误差回升，PIH 单调改善[^src-pih]。
2. **归因**：消融仅给图无数值表——两模块各自有效、组合最优，单组件掉点无法量化；MEW 在 5/7 数据集推长、余者触顶，$\beta$ 与采样温度敏感[^src-pih]。
3. **缺口**：短窗 $L=96$ 反落后 PatchTST；MEW 受离散粗网格分辨率截断；缺 LLM 类与 Exchange 基线[^src-pih]。

## 相关页面

- [[patchtst|PatchTST]] — 瓶颈接力 $\Delta$：PIH 以其为宿主压降算力并温和推长 MEW，但短窗 $L=96$ 反而落后宿主。
- [[mamba|Mamba]] — 跨槽位正交：Mamba 负责长程选择性滤波，Transformer 负责短段局部注意力，两者混合优于任一单独使用。
- [[information-bottleneck-principle|信息瓶颈原理]] — 同构实现：把压缩—充分性的变分上界落成序列级可微噪声掩码。
- [[lookback-horizon-scaling]] 与 [[source-2606-27282-ltsf-linear-models|SearchCast]] — 同槽位互斥：同在「回看窗长度如何定」槽位，PIH 以统一架构推长 MEW，SearchCast/LHS 按序列与步长搜索异质最优窗（幂律斜率可负），两视角互为张力。

出处：Jiahui Zhang 等，中国科学技术大学；NeurIPS 2025 Poster。

[^src-pih]: [[source-pih]]
