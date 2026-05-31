---
title: "S-Mamba: Is Mamba Effective for Time Series Forecasting?"
type: source-summary
tags:
  - time-series
  - mamba
  - state-space-model
  - forecasting
  - multivariate
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# Source: S-Mamba

Zihan Wang et al., "Is Mamba Effective for Time Series Forecasting?", Neurocomputing 2024 (arXiv: 2403.11144).

## 核心贡献

S-Mamba 是首个将 [[mamba|Mamba]] 选择��状态空间模型引入多变量时间序列预测（MTSF）的 baseline 框架。它将变量间相关性（VC）编码从 Transformer 的 self-attention 转移到双向 Mamba block，同时用 FFN 处理时间依赖（TD）[^src-s-mamba]。

## 架构设计

S-Mamba 采用两阶段编码策略[^src-s-mamba]：

- **Mamba VC Encoding Layer**：双向 Mamba block 替代 Transformer attention，实现轻量、低开销的全局变量间相关性提取。各变量被视作独立通道，Mamba 的 RNN-like 的顺序处理能力在此层捕获跨变量依赖模式。
- **FFN TD Encoding Layer**：Feed-Forward Network 提取每个变量内部的时间依赖关系，保持与 [[itransformer|iTransformer]] 类似的 FFN-on-time 设计。

## 关键发现

**主实验**：在 13 个公开数据集（PEMS 系列、Traffic、Electricity、Weather、Solar-Energy、ETT 系列、Exchange）上对比 [[patchtst|PatchTST]]、[[itransformer|iTransformer]]、Crossformer、DLinear、FEDformer、Autoformer 等 9 个 SOTA 模型。S-Mamba 在多数数据集上取得最优或次优 MSE，同时 GPU 内存占用和训练时间低于 Transformer 基线[^src-s-mamba]。

**消融实验揭示的核心发现**[^src-s-mamba]：

| 组件替换 | 结论 |
|----------|------|
| VC Encoding: Bi-Mamba → Attention | MSE 显著上升 |
| VC Encoding: Bi-Mamba → Uni-Mamba | 性能退化（丢失一半方向信息）|
| TD Encoding: FFN → Bi-Mamba | 性能显著下降 |
| TD Encoding: FFN → Attention | 性能下降 |
| 移除 VC Encoding | 性能崩溃 |
| 移除 FFN TD Encoding | 性能严重退化 |

核心结论：**Mamba 在 VC 编码上优于 Transformer，而 FFN 在 TD 编码上保持统治地位**[^src-s-mamba]。

此外[^src-s-mamba]：

- **变量顺序不敏感**：S-Mamba 不受 variate reordering 影响（与 Hippo 矩阵初始化偏差预期相反），表明训练后 Mamba 能有效学习全局跨变量相关性。
- **泛化能力**：仅用 40% 变量训练后预测全部变量的实验证明 Mamba 具备与 Transformer 相当的泛化能力。
- **增窗下性能提升**：S-Mamba 和 iTransformer 随回看窗口增大性能持续提升，且 S-Mamba 始终优于 iTransformer。
- **Mamba 可提升现有 Transformer**：在 Reformer/Informer/Transformer 的 Encoder-Decoder 间插入 Mamba block 即可获得性能增益，甚至可整体替换高级 Transformer（Autoformer、Flashformer、Flowformer）的 Encoder 为 uni-Mamba。

## 局限性

- 在 variate 数量少（Exchange, ETT 系列）且多为非周期变量的数据集上优势有限，甚至 VC Encoding 可能引入噪声[^src-s-mamba]。
- 未探索 Mamba 作为预训练 backbone 的潜力（列为 Future Work）。
- 受 Hippo 矩阵初始化影响，Mamba 天然带有"邻近变量优先"的偏差，虽训练后缓解，但超长变量序列场景仍需关注。

[^src-s-mamba]: [[s-mamba]]
