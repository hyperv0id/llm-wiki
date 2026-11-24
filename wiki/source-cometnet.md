---
title: "CometNet: Contextual Motif-guided Long-term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - motif-retrieval
  - mixture-of-experts
  - periodicity
  - aaai-2026
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# CometNet: Contextual Motif-guided Long-term Time Series Forecasting

**来源**：Weixu Wang、Xiaobo Zhou、Xin Qiao、Lei Wang、Tie Qiu（天津大学计算机学院）；AAAI 2026（扩展版 arXiv:2511.08049）[^src-cometnet]。

常规长时时间序列预测受限于固定长度的回看窗口，且滑窗训练的梯度无法跨窗口传播，难以捕捉跨度数千步的重现规律。CometNet 在训练前先从整条训练序列离线抽出一组跨尺度的代表性上下文母题，在线预测时让短回看窗以专家混合内容门控匹配相关母题，并叠加当前窗口在母题生命周期中的相对位置信号，从而突破短回看窗的视野约束[^src-cometnet]。

## 去黑话算子配方与路线差异

模型未引入改变数学结构的新底层算子，核心由离线标准信号聚类与在线常规门控线性模块拼装而成[^src-cometnet]：

- **离线母题库抽取**：对训练集做 `MovingAvg` 去趋势与 `FFT` 提取高幅主导周期，按各尺度下采样切分候选子序列；再以 `DTW` 相似度构图取连通分量做跨尺度去重，每簇选中心原型并按覆盖收益筛选 $K$ 个主导母题（高维场景依赖未在正文展开的频域哈希分组并行加速）[^src-cometnet]。
- **在线内容门控预测**：将局部回看窗口线性编码为 query，对 $K$ 个母题专家做 `softmax` 加权（Dense MoE），融合「当前窗口处于母题生命周期的相对相位」信号，经多尺度滤波与 `Linear` 头输出预测[^src-cometnet]。
- **有向关系对照**：与 [[gtr|GTR]] 属**同槽位互斥**（可学习绝对相位周期矩阵检索 vs 离线 `DTW` 离散母题内容门控检索）；与 [[contextual-pattern-bank|Contextual Pattern Bank]] 属**同构对照**（参数化增量模式库 vs 非参数聚类母题库）；相对 [[patchtst|PatchTST]] 构成**瓶颈接力 $\Delta$**（窗内局部 patch 编码 vs 挂载窗外离线母题外推）；属 [[mixture-of-experts|Mixture of Experts]] 的**架构特例**（以离线母题作稠密专家并融合生命周期相位）；并与 [[lookback-horizon-scaling|Lookback–Horizon Scaling]] 形成对照，绕开「长预测步长需更长回看窗口」的约束[^src-cometnet]。

## 证据

1. **定位与主结论**：固定 $L=96$ 的 8 个基准中 7 项平均 MSE/MAE 最低，长步长（如 ETTh2 $H=720$）优势更显著[^src-cometnet]。
2. **真实涨点归因**：消融仅覆盖 4 个低维 ETT 的 $H=720$；去母题引导掉点最重（ETTh1 MSE $0.391\to 0.544$），去 MoE 路由（$0.476$）与多尺度发现（$0.447$）次之，ETTh2 上生命周期相位重要性高于多尺度发现[^src-cometnet]。
3. **证据缺口与负结果**：862 维 Traffic 平均 MSE/MAE 逊于 TimeMixer++ 排第二；高维依赖未在正文展开的分组加速，且缺跨数据集迁移与建库成本评测[^src-cometnet]。

## 相关页面

- [[gtr]] — 可学习绝对相位周期矩阵 vs 离线离散母题内容门控（同槽位互斥）
- [[contextual-pattern-bank]] — 参数化模式库 vs 非参数聚类母题库（同构对照）
- [[patchtst]] — 窗内局部编码 vs 窗外母题外推（瓶颈接力 $\Delta$）
- [[mixture-of-experts]] — 离线母题原型作稠密专家融合生命周期相位（架构特例）
- [[lookback-horizon-scaling]] — 绕开长预测步长需更长回看窗口的约束（路线对照）

[^src-cometnet]: [[source-cometnet]]
