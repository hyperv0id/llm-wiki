---
title: "Learning Weekly Urban Evolution: Delay-Coupled Neural Dynamics for Extremely Long-Horizon Spatiotemporal Forecasting"
type: source-summary
tags:
  - traffic-forecasting
  - long-horizon-forecasting
  - delayed-interaction
  - graph-dynamics
  - residual-decomposition
  - iclr-2027
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# Learning Weekly Urban Evolution: Delay-Coupled Neural Dynamics for Extremely Long-Horizon Spatiotemporal Forecasting

匿名投稿，ICLR 2027（under review，double-blind）。四数据集（PEMS04 / PEMS08 / SD / Seattle）、四 horizon（8 h / 1 d / 3 d / 7 d）共 16 个 setting。

## 核心论断

真实交通数据给出三项观测：邻接节点偏差起始时刻错开并随时间重叠累积；偏差跨午夜延续、节点间滞后相关随日内阶段变化；不同日期共享日 profile 但局部峰谷幅度不同（Fig. 1）。由此提出三项建模要求与 DeCoDy——自述为「graph dynamics framework with time delays」（§1、§3）[^src-decody-iclr2027]。

## 机制

- 用训练集统计构造[[shrinkage-periodic-background|收缩的周周期背景]]，历史与未来都以残差表示（式 1–2）；
- 由历史表示、已知未来日历与背景形状生成节点持续性 $d_i$、时变驱动 $s_i[k]$ 与逐时步逐延迟的耦合 $A_{ij\ell}[k]$，耦合经尺度化归一使行增益低于 $\rho$（式 5–7）；
- 残差递推含自状态持续、局部驱动、五个延迟档（1/2/3/6/12 步，即 5–60 min）的邻居影响三项（[[multi-delay-coupled-accumulation|式 9–10]]）；
- 因 $d_i$ 在 horizon 内固定，自回归项折叠为指数核卷积，其余跨节点延迟耦合构成线性系统 $(I-\mathcal{K})r=r^{base}$，由[[constrained-full-horizon-response-solver|并行不动点迭代]]同时求解全部未来位置（式 11–14）。

## 证据

Table 1：MAE 在 16 个 setting 中最低 14 个（落后 SD 的 1 d 与 7 d），7 天在四个数据集上 RMSE 均最低。正文写「lowest RMSE in 9 settings」，按 Table 1 计数为 10 个，两处不一致[^src-decody-iclr2027]。Table 2 的五个消融变体在 12 个 setting 上均劣于完整模型，多数差距 0.1–1%。求解器在 PEMS08 上 $H=2016$ 时并行 32.54 ms 对串行 1136.20 ms（Table 5）[^src-decody-iclr2027]。

## 局限

PEMS04/PEMS08 用节点子集，作者声明不能与全节点版本已发表结果比数；DeCoDy 额外输入天气与事件，主表无法分离架构与外部信息的贡献；单种子且无显著性检验；无参数量、训练时间与端到端时延；延迟档数、回看长度与外推精度均无敏感性实验；未与 PDFormer 等延迟建模方法对比（§A.1–A.4、附录 E）[^src-decody-iclr2027]。

详见 [[decody]]。

[^src-decody-iclr2027]: [[source-decody-iclr2027]]
