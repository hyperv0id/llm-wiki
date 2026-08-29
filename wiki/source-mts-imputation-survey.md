---
title: "Deep Learning for Multivariate Time Series Imputation: A Survey"
type: source-summary
tags:
  - survey
  - data-imputation
  - time-series
  - taxonomy
  - arxiv
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# Source: Deep Learning for Multivariate Time Series Imputation: A Survey

**作者**: Jun Wang, Wenjie Du（共同一作）, Yiyuan Yang, Linglong Qian, Wei Cao, Keli Zhang, Wenjia Wang, Yuxuan Liang, Qingsong Wen（通讯）（PyPOTS Research / HKUST / Oxford / KCL / Huawei Noah's Ark Lab / HKUST-GZ / Squirrel Ai Learning）
**版本**: arXiv:2402.04059v3 [cs.LG]，2025-05-20，9 页（LaTeX/pdfTeX，创建于 2025-05-21）
**raw**: `raw/wang-mts-imputation-survey-arxiv-2024.pdf`
**著录备注**: 用户著录为 IJCAI 2025 接收；PDF 内仅 arXiv v3 水印、无 IJCAI 标识，wiki 未在 PDF 内核实。

## 核心论点

综述提出深度多元时间序列插补（MTSI）双视角分类：按**插补不确定性**分预测式（输出确定值，重构式损失）与生成式（从学习分布采样以量化不确定性，对数似然目标）；按**网络架构**分 RNN/CNN/GNN/Attention（预测式）与 VAE/GAN/Diffusion（生成式），并把大模型（PFM/LLM）单列为第三类（应用策略与一般神经网络不同）[^src-mts-imputation-survey]。作者自称首个深度 MTSI 系统性综述[^src-mts-imputation-survey]。

缺失机制按 Rubin 1976 分 MCAR/MAR/MNAR：MCAR/MAR 可忽略（只学 $p(X^o)$），MNAR 不可忽略（需联合学 $p(X^o, M)$）[^src-mts-imputation-survey]。Table 1 对 33 个方法标注类别、架构与缺失机制——绝大多数标注 MCAR，仅 supnotMIWAE 标注 MNAR、SADI 标注 MCAR/MAR/MNAR[^src-mts-imputation-survey]。

综述第二项贡献是工具箱梳理：imputeTS（R，单变量）、mice、GluonTS、Sktime、ImputeBench 等，重点介绍自研 [[pypots|PyPOTS]] 生态（写作时含 37 个插补模型）与 TSI-Bench 基准（172 个公开数据集、28 个算法、34,804 组实验）[^src-mts-imputation-survey]。

## 未来方向

1. **缺失机制**：现有方法多以 MCAR/MAR 运作，真实数据 MNAR 常见且不可忽略，会引入分布偏移[^src-mts-imputation-survey]。
2. **下游性能**："impute and predict" 两阶段范式为主流，"encode-and-predict" 端到端范式在缺失模式携带下游信息时更有前景[^src-mts-imputation-survey]。
3. **可扩展性**：深度插补计算成本高于统计/机器学习方法，需并行与分布式方案[^src-mts-imputation-survey]。
4. **大模型**：领域时间约束与缺失先验纳入预训练、不规则模式架构创新、多模态利用、不确定性量化评估[^src-mts-imputation-survey]。

## 范围与局限

- 覆盖截至约 2024 年的方法（Table 1 最晚为 NeurIPS 2024），未涉及其后的流匹配插补路线。
- 部分方法的转述较浅：如 ImputeFormer 仅被描述为"利用自注意力与时间上下文建模的 Transformer 框架"，未提其低秩归纳偏置核心（见 [[imputeformer]]）。
- Table 1 机制标注为综述作者二手归类，与各原论文自述可能不一致，引用时须保持归因。

## 相关页面

[[mts-imputation-taxonomy]] · [[pypots]] · [[missing-not-at-random]] · [[csdi]] · [[pristi]] · [[grin]]

[^src-mts-imputation-survey]: [[source-mts-imputation-survey]]
