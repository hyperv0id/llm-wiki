---
title: "A Survey and Benchmarking of Spatial-Temporal Traffic Data Imputation Models (Guo et al., arXiv:2412.04733v2)"
type: source-summary
tags:
  - spatiotemporal-imputation
  - benchmark
  - traffic
  - survey
  - experimental-evaluation
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

**源文件**：`raw/guo-imputation-evaluation-st-traffic-arxiv-2024.pdf`。**版本核实**：PDF 水印为 arXiv:2412.04733**v2** [cs.LG] 17 Oct 2025，题名 "A Survey and Benchmarking of Spatial-Temporal Traffic Data Imputation Models"（v2 改题后的题名，PDF 内核实成立）。v1 题名依 [[fence|FENCE]]（AAAI 2026）参考文献著录为 "An Experimental Evaluation of Imputation Models for Spatial-Temporal Traffic Data"（2024，raw PDF 核实）；本仓库无 v1 PDF。作者 Shengnan Guo、Tonglong Wei（共同一作）、Yiheng Huang、Zekai Shen、Yujuan Dong、Junliang Lin、Youfang Lin、Huaiyu Wan（北京交通大学）与 Yan Lin（Aalborg University）[^src-guo-imputation-evaluation]。

**核心论点**：论文将交通插补研究归纳为三个缺口——缺模型分类法、缺统一可复现的评测管线、缺跨有效性/效率/鲁棒性的深入比较，据此给出两项贡献：practice-oriented 双分类（缺失模式四类 SRTR/SRTC/SCTR/SCTC，见 [[traffic-missing-patterns]]；模型按时空建模技术 RNN/Attention/GNN/TC × 损失设计（predictive / 生成式 GAN-VAE-Diffusion，训练策略 masked/reconstruction SSL）），以及统一评测管线：11 个模型 × 4 个数据集（PEMS04、PEMS08、TW、Seattle）× 20 个缺失场景（4 模式 × 5 缺失率），评测代码公开[^src-guo-imputation-evaluation]。

**主要发现（该评测复现口径）**：缺失率低于 0.5 时各模型误差变化相对稳定，达到 0.7 与 0.9 时所有模型误差显著上升；PEMS04/08 上 ImputeFormer 在空间随机（SR-）模式最好、LATC 在空间连续（SC-）模式最好；Seattle 速度数据上 BRITS 一致最优；TW 上 IGNNK 在多数模式最好但 SCTR 下表现差。按进入 top-3 的次数计，BRITS、GCASTN、PriSTI、ImputeFormer、LATC 最稳健，评测者将共同点归因于先验知识引入（时延机制或低秩假设）。效率上 LATC 内存最小（98MB），PriSTI 推理最慢（8553.77s，PEMS04 SR-TR 0.5 设置）。详见 [[st-traffic-imputation-benchmark]][^src-guo-imputation-evaluation]。

**局限与不一致（如实记录）**：摘要与贡献均写 11 个模型、正文 Sec. IV 与 V.B 三处写 "10 models/baselines"（"select the following 10 models"、"10 recently proposed sequence imputation and prediction models"、"10 baselines"），实际列出 A–K 共 11 个（另有 LAST 朴素基线）——正文与摘要的 10/11 计数不一致。无独立局限性章节；"提取额外先验知识可能是方向"为作者自述推测（"we guess"）。正文称 LATC 训练时间最小，但其 Table IV 中 E2GAN 训练 312.42s 低于 LATC 829.00s[^src-guo-imputation-evaluation]。

**关联**：[[fence|FENCE]] 引用本评测 v1（raw 核实）；[[loft|LOFT]]（KDD 2026）为同组后续插补工作、基线清单与本评测重叠（IGNNK/GCASTN/ImputeFormer/PriSTI 等），其 raw PDF 不在仓库，对本文的引用关系未在 raw 核实。

[^src-guo-imputation-evaluation]: [[source-guo-imputation-evaluation]]
