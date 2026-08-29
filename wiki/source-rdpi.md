---
title: "RDPI: A Refine Diffusion Probability Generation Method for Spatiotemporal Data Imputation"
type: source-summary
tags:
  - diffusion-models
  - spatio-temporal-imputation
  - conditional-diffusion
  - residual-learning
  - aaaai-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: low
status: active
---

# Source: RDPI — A Refine Diffusion Probability Generation Method for Spatiotemporal Data Imputation

**RDPI** 由 Zijin Liu、Xiang Zhao、You Song（北京航空航天大学，You Song 为通讯作者）发表。**raw:** `raw/liu-rdpi-refine-diffusion-imputation-arxiv-2024.pdf`，为 arXiv v1（arXiv:2412.12642，2024-12-17，双栏 AAAI 模板排版）：首页含 AAAI 2025 版权声明（Copyright © 2025, AAAI），但无会议论文集页眉；AAAI 2025 会议著录来自用户，与版权声明一致，未在 PDF 内以 proceedings 形式核实。代码开源于 github.com/liuzjin/RDPI[^src-rdpi]。

## 核心论点

论文将时空插补方法分为确定性、概率与扩散三类，认为自回归确定性方法缺少不确定性建模、易误差累积，而以 CSDI 为代表的条件扩散模型只在反向去噪训练中使用观测条件，前向与插补过程忽略条件（论文表述）[^src-rdpi]。RDPI 提出两阶段框架：初始阶段用确定性插补模型（实验用 GRIN）生成粗插值；精炼阶段以粗插值与真值间的残差为扩散目标训练条件扩散模型，并把观测值纳入前向过程、据此推导新 ELBO（论文自述贡献）；摘要另称显著降低采样计算成本（正文无耗时对比数据）[^src-rdpi]。

## 实验结果（作者报告）

PEMS-BAY、METR-LA、AQI、AQI36 四个数据集，各实验跑 5 次。Table 3（空气质量）与 Table 4（交通）中 RDPI 的 MAE/MSE 均为最优，如 AQI36 In-sample MAE 7.98±0.24（MIDM 9.41、CSDI 9.60）；作者称 In-sample MSE 相对最近基线在 AQI36 降低超过 34%、在 AQI 降低超过 50%（Results 节）。MRE 非全面最优：AQI In-sample 上 MIDM 16.87 优于 RDPI 17.17，作者归因于概率采样对残差的平滑作用（Results 节）[^src-rdpi]。

## 范围与局限

论文自述：MRE 不能充分反映模型表现（作者解释）；时空解耦与图结构/多维时间序列的条件扩散留作未来工作（Conclusion）。三点文本问题：正文数据集段落与 Table 1 的 PEMS-BAY/METR-LA 节点数互换（207/325），该段并称 "three datasets" 而表列四个；predicting xθ 变体在 AQI36 的 MSE（153.66）低于完整 RDPI（238.25），与正文 "x 预测不是最优选择" 的总结不完全一致；基线清单不含 PriSTI，而划分与缺失协议沿用 GRIN 与 PriSTI（Settings 节）[^src-rdpi]。

## 相关页面

[[rdpi]] · [[forward-process-conditioning]] · [[csdi]] · [[grin]] · [[pristi]]

[^src-rdpi]: [[source-rdpi]]
