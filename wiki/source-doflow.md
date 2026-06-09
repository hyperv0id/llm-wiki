---
title: "DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting on Time Series (ICLR 2026)"
type: source-summary
tags:
  - time-series
  - causal-inference
  - continuous-normalizing-flow
  - flow-matching
  - counterfactual
  - generative-model
  - iclr
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# DoFlow（ICLR 2026）

**DoFlow** (Wu, Xie & Qiu, Georgia Tech + Northwestern–Argonne, ICLR 2026, arXiv:2511.02137) 是一个定义在因果有向无环图（DAG）上的流式生成模型，统一了时间序列的**观测（observational）**、**干预（interventional）** 和**反事实（counterfactual）** 预测[^src-doflow]。

## 核心问题

大多数现代预测器是纯观测式的：它们从历史中学习相关性并外推，但无法回答因果"what-if"问题[^src-doflow]。论文聚焦两类因果查询：(1) 干预查询——"在某个计划性的变量修改下预测会如何变化？"；(2) 反事实查询——"如果当初施加了不同的干预，这条已观测到的特定轨迹会变成什么样？"[^src-doflow]。作者指出，在其撰文时尚无针对反事实时间序列预测的通用框架[^src-doflow]。

## 方法

DoFlow 为 DAG 中每个节点 $i$ 学习一个**时间条件化的连续归一化流（CNF）**，自回归地预测 $X_{i,t}$，每步以节点自身与父节点的历史为条件[^src-doflow]。历史由 RNN（LSTM 或 GRU）编码器汇总为隐藏状态 $H_{i,t-1}=\text{concat}(h_{i,t-1}, h_{\text{pa}(i),t-1})$[^src-doflow]。每个 CNF 通过 Neural ODE（ODE 时间 $s\in[0,1]$，与时序索引 $t$ 不同）连接数据分布（$s=0$）与基分布 $\mathcal{N}(0,1)$（$s=1$），用 Conditional Flow Matching 损失训练[^src-doflow]。

- **观测/干预预测**：按拓扑序逐节点解码；被干预节点固定为 $\gamma_{i,t}$，其余节点从 $z\sim\mathcal{N}(0,1)$ 反向积分生成，隐藏状态自回归更新使干预沿 DAG 传播[^src-doflow]。
- **反事实预测**：遵循 abduction–action–prediction 三步——先将事实观测编码为潜变量（abduction），再施加干预（action），最后从该潜变量解码出反事实轨迹（prediction）[^src-doflow]。

## 理论与附加能力

论文给出**反事实恢复**结果（Corollary 4.5）：在单调 SCM 与三条假设（A1 外生噪声独立、A2 结构方程关于噪声严格单调连续、A3 编码潜变量与条件独立）下，编码-解码过程几乎必然恢复真实反事实[^src-doflow]。该结果与 Bijective Generation Mechanisms (BGM) 相关但不要求分布匹配，提供模型特定的逐点恢复[^src-doflow]。此外，CNF 通过变量变换公式赋予预测轨迹显式对数似然，可用于**基于似然的异常检测**[^src-doflow]。

## 结果

在 Tree、Diamond、FC-Layer、Chain 四种合成 DAG（含加性与非线性非加性 SCM）上，用 RMSE、MMD、CRPS 评估；DoFlow 在观测与干预预测上一致优于改造后的 GRU/TFT/TiDE/TSMixer/DeepVAR/MQF2 基线，并独有反事实预测能力（基线在干预任务给出 NA）[^src-doflow]。真实应用：Argonne 水电系统的干预预测与提前 10–20 分钟的停机异常检测；以及 Bica et al. (2020a) 癌症治疗数据集上的因果治疗效应估计，归一化 RMSE 显著优于 CRN、RMSN、MSM[^src-doflow]。

## 局限

DoFlow 假设**因果 DAG 已知且正确设定**，并满足因果充分性（无隐藏混杂）[^src-doflow]。反事实在 DoFlow 中产生单一确定性轨迹，故 MMD/CRPS 只对观测与干预计算[^src-doflow]。每个节点训练一个独立的流，但网络较浅、总规模适中[^src-doflow]。

## 链接

- [[doflow]] — 方法/模型主页
- [[causal-counterfactual-recovery]] — abduction–action–prediction 与反事实恢复理论
- [[continuous-normalizing-flow]] — CNF 基础
- [[flow-matching]] — CFM 训练目标
- [[neural-ordinary-differential-equation]] — Neural ODE
- [[e2-cstp]] — 另一种因果时空预测方法（混杂消除路线）

[^src-doflow]: [[source-doflow]]
