---
title: "STAMImputer: Spatio-Temporal Attention MoE for Traffic Data Imputation"
type: source-summary
tags:
  - spatio-temporal
  - traffic
  - data-imputation
  - mixture-of-experts
  - graph-attention
  - low-rank
  - block-missing
  - arxiv-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# STAMImputer: Spatio-Temporal Attention MoE for Traffic Data Imputation

**STAMImputer** (arXiv 2506.08054, Wang et al., Beihang University et al.; extended version of an IJCAI 2025 paper) is a Mixture-of-Experts spatio-temporal imputation model for traffic data, claimed to be the first application of the MoE framework to traffic data imputation[^src-stamimputer].

## 核心问题

论文针对现有交通数据填补的两个痛点[^src-stamimputer]：

1. **时序-空间顺序学习的失效**：主流的 time-to-space 顺序框架（先时间模块、后空间模块）在块状缺失（[[partial-blackout|block missing]]）场景下会提取并传播无效特征，尤其当时空块同时缺失（如传感器长时间断电）时更为严重[^src-stamimputer]。
2. **静态图的局限**：静态图方法难以捕获超出局部社区的全局空间依赖；而纯全局空间注意力方法在高度稀疏数据下缺乏先验引导，难以捕获正确的空间相关性[^src-stamimputer]。

作者主张时空填补应根据实时缺失情况动态调整，并结合静态图与全局注意力两者之长[^src-stamimputer]。

## 方法

STAMImputer 用一个**外层 MoE 框架**协调三类专家，分三个阶段运行——注意力阶段、观测阶段、读出阶段[^src-stamimputer]：

- **时间专家 (T-Expert)**：经典多头自注意力 Transformer 编码器 (MSAT)，擅长恢复稳定低频序列趋势[^src-stamimputer]。
- **空间专家 (S-Expert)**：新颖的 [[lrsgat|Low-rank guided Sampling Graph ATtention (LrSGAT)]] 机制，通过混合采样平衡局部与全局空间相关性，并生成半自适应动态图[^src-stamimputer]。
- **观测专家 (O-Expert)**：前馈网络，扮演"仲裁者"，基于原始（含缺失）数据与稀疏度特征对注意力专家的输出打置信分，动态加权时空注意力（[[mixture-of-experts|MoE]] 路由）[^src-stamimputer]。

与传统将专家置于 FFN 内层的 MoE 不同，STAMImputer 将专家网络上移到框架外层，以"受控解耦"方式分离时空维度特征[^src-stamimputer]。输入前先用离散小波变换 (DWT) 分解低/高频分量，并拼接时间、时空位置编码与稀疏度特征作为嵌入[^src-stamimputer]。

## 结果

在 PemsD8、SZ-Taxi、DiDi-SZ、NYC-Taxi 四个交通基准上，分点缺失 (25%/60% 缺失率) 与块缺失 (0.2%/1% 故障概率) 两种模式评测，以 MAE 衡量，STAMImputer 总体优于 [[imputeformer|ImputeFormer]]、SPIN、SAITS 等 SOTA，仅个别场景排名第二[^src-stamimputer]。消融实验显示注意力专家最关键，块缺失下将 LrSGAT 换成 MLP 性能急剧退化；MoE 在两种缺失模式下均一致有益[^src-stamimputer]。下游实验中，将其学到的动态图 (DGSL) 接入 Graph-WaveNet，可在高稀疏/块缺失场景显著提升预测[^src-stamimputer]。

## 局限

- 仅在交通流/速度数据 (C=1 单通道) 上验证，未涉及多变量泛化[^src-stamimputer]。
- 动态图构建被作者列为待改进方向（追求更高效构建方法）[^src-stamimputer]。
- 总复杂度 $O(T^2D + N\log N\,D + NTD)$，时间专家的 $O(T^2D)$ 在长序列上仍是瓶颈[^src-stamimputer]。

[^src-stamimputer]: [[source-stamimputer]]
