---
title: "MTSCI — A Conditional Diffusion Model for Multivariate Time Series Consistent Imputation"
type: source-summary
tags:
  - time-series
  - data-imputation
  - diffusion-models
  - contrastive-learning
  - cikm-2024
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

**源文件**：`raw/zhou-mtsci-arxiv-2024.pdf`。**版本核实**：PDF 首行为 arXiv:2408.05740v1（2024-08-11）水印，但版式为 ACM CIKM'24 会议排版——页眉 "CIKM '24, October 21–25, 2024, Boise, ID, USA"、ACM 版权块、ISBN 979-8-4007-0436-9/24/10、DOI 10.1145/3627673.3679532。**CIKM 2024 著录可在 PDF 内核实**。论文全题 "MTSCI: A Conditional Diffusion Model for Multivariate Time Series Consistent Imputation"；作者 Jianping Zhou、Junhao Li、Guanjie Zheng（通讯作者）、Xinbing Wang（上海交通大学）与 Chenghu Zhou（中国科学院）[^src-mtsci]。

**核心论点**：论文提出插补一致性（imputation consistency）概念，分为 intra-consistency（插补值与观测值在窗口内互相可重构）与 inter-consistency（完整样本与相邻窗口保持时序一致），并认为既有插补方法只依赖插补目标的归纳偏置、未处理一致性（Sec. 1，作者观点/自述）。机制：前向加噪以 contrastive complementary mask 生成互补双视图、以 InfoNCE 式 intra contrastive loss 约束两视图表示（Sec. 4.2、4.3.1）；去噪阶段以 mixup 机制把 context 窗口条件与当前窗口观测混合，推理时退化为单窗口条件（Sec. 4.3.2）[^src-mtsci]。去噪网络为 vanilla + inverted transformer 双块编码器（作者称后者受 iTransformer 启发）；作者报告 ε-预测优于 $x_0$-预测（Table 4）[^src-mtsci]。

**实验**（作者报告）：ETT / Weather / METR-LA 三数据集、13 个基线、point（20% 随机）与 block 缺失两种模式，指标 MAE/RMSE/MAPE、5 次运行平均。作者报告 Table 2/3 各 9 列（3 数据集 × 3 指标）数值均由 MTSCI 取得最低，平均提升 17.88% MAE / 15.09% RMSE / 13.64% MAPE；CRPS 六组设置低于 CSDI（Table 5）；缺失率 10–70% 敏感性见 Table 6（point 10–50% 档 CSDI 的 RMSE 低于 MTSCI，论文"优于基线"为总结性表述），跨缺失模式泛化见 Table 7[^src-mtsci]。

**局限与注意**：论文无独立局限性章节，结论自述未来扩展到更复杂缺失场景（Sec. 6）；Table 7 非全指标占优（如 ETT Block→Point 上 CSDI 三项指标更低，论文表述为 "relatively better performance"）；context 窗口仅训练期可用；对比损失权重 $\lambda$ 需小范围调节（Sec. 5.6）。注意：LOFT/FENCE/PRDIM 等后续论文表格中的 MTSCI 数字为其各自复现/适配口径，与本页原文数字分立[^src-mtsci]。

[^src-mtsci]: [[source-mtsci]]
