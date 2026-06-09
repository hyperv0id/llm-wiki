---
title: "RATD: Retrieval-Augmented Diffusion Models for Time Series Forecasting (NeurIPS 2024)"
type: source-summary
tags:
  - diffusion-models
  - time-series-forecasting
  - retrieval-augmented-generation
  - probabilistic-forecasting
  - neurips-2024
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# RATD: Retrieval-Augmented Diffusion Models for Time Series Forecasting

**来源**：Jingwei Liu, Ling Yang, Hongyan Li, Shenda Hong. *Retrieval-Augmented Diffusion Models for Time Series Forecasting*. NeurIPS 2024. (arXiv:2410.18712)

## 核心论点

[[ratd|RATD]] 提出**首个检索增强的时间序列扩散模型**，针对现有时序扩散模型（如 [[csdi|CSDI]]、[[timegrad|TimeGrad]]）性能不稳定的问题[^src-ratd]。作者指出两个限制因素：(1) 时间序列普遍**缺乏语义/标签引导**，去噪过程没有像图像扩散那样的文本或标签引导；(2) 时序数据集**规模不足且严重不平衡**，导致模型倾向生成常见预测、难以处理罕见复杂样本[^src-ratd]。

## 方法

RATD 包含两部分[^src-ratd]：

1. **嵌入式检索（embedding-based retrieval）**：用冻结的预训练编码器 $E_\phi$（默认 TCN）将历史序列与数据库样本嵌入，按 L2 距离检索 k 个最近邻，取其未来段作为「参照（references）」$x^R$。数据库有两种构建方式——规模不足数据集直接用整个训练集，类别不平衡数据集（如 MIMIC）用含全部类别的子集。检索索引被预处理存入字典以降低训练成本。
2. **参照引导的扩散模型（reference-guided diffusion）**：基于 DiffWave + 2D Transformer 架构（与 CSDI 同源）。关键模块 **Reference Modulated Attention (RMA)** 通过矩阵点积融合三种特征——当前时序特征、侧信息 $I_s$、参照特征——把参照作为条件注入去噪过程。RMA 置于每个残差块开头效果最佳。网络采用 **$x_0$-预测**（[[x-prediction]]）而非 $\epsilon$-预测，消融显示前者更优。

## 主要贡献

- 首次将检索增强（RAG）范式引入时间序列扩散预测，强调对不足数据的最大化利用[^src-ratd]；
- 设计 RMA 模块以低成本提供合理的参照引导[^src-ratd]；
- 在 5 个真实数据集上用 MSE/MAE/CRPS 多指标系统评测[^src-ratd]。

## 结果

在 Exchange、Wind、Electricity、Weather 四个数据集上，RATD 超越扩散基线（CSDI、TimeDiff、mr-Diff、D³VAE），并在 4 个数据集中 3 个超越全部基线；在缺乏短期周期性的 **Wind** 数据集上优势最显著[^src-ratd]。在 **MIMIC-IV-ECG** 上，完整测试集与 iTransformer 相近，但在罕见病例子集（占比 < 2%）显著领先，证明检索增强对复杂/罕见任务的针对性价值[^src-ratd]。尽管多了检索模块，因非自回归框架，采样效率甚至略优于 TimeGrad/MG-TSD/SSSD[^src-ratd]。

## 局限性

- Transformer 框架在变量数过多时计算开销大[^src-ratd]；
- 训练前的检索预处理增加约十小时训练时间[^src-ratd]。

## 关联

RATD 是检索增强时序/时空生成研究线的奠基工作之一，早于 [[craft|CRAFT]]（NeurIPS 2025）、[[rast|RAST]]（AAAI 2026）、[[middir|MiDDiR]]（ICLR 2026）。其 RMA 以「条件特征输入」方式利用检索，区别于 [[retrieval-guidance|MiDDiR 的分析性得分倾斜]]。

[^src-ratd]: [[source-ratd]]
