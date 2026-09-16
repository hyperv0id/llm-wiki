---
title: "When Do Contrastive Learning Signals Help Spatio-Temporal Graph Forecasting? (STGCL)"
type: source-summary
tags:
  - spatio-temporal-graph
  - contrastive-learning
  - traffic-forecasting
  - self-supervised
  - arxiv
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: high
status: active
---

# Source: STGCL — When Do Contrastive Learning Signals Help Spatio-Temporal Graph Forecasting?

## 著录

- **作者**：Xu Liu、Yuxuan Liang、Bryan Hooi、Roger Zimmermann（NUS）；Chao Huang（HKU）；Yu Zheng（JD iCity）。Liu 与 Liang 共同一作。
- **发表**：ACM SIGSPATIAL 2022，DOI [10.1145/3557915.3560939](https://doi.org/10.1145/3557915.3560939)，正文 12 页。
- **版本**：arXiv:2108.11873，v1 为 2021-08，本次读 v2 定稿（2022-11-03）。
- **原文**：[arXiv](https://arxiv.org/abs/2108.11873v2) · [HTML](https://arxiv.org/html/2108.11873v2) · [PDF](https://arxiv.org/pdf/2108.11873v2.pdf) · [代码](https://github.com/liuxu77/STGCL)
- **本地**：`downloads/stgcl.pdf`（SHA256 `5f6a27a9…b85cf`）、`raw/stgcl.txt`（SHA256 `84f51b70…daa5`；2026-09-15 由 downloads/ 迁入 raw/）。
- **阅读范围**：正文 §1–§7、References、Appendix A。表格数据经 PDF 文本抽取核对，未做版面级校验。
- **性质**：实证探索 / 方法框架论文，不是 survey。

## 摘要

出发点：交通时空图基准 PEMS-04 与 PEMS-08 各只有约 17,000 个实例，模型容易过拟合；对比学习被当成补充自监督信号的手段，全文用四个问题组织对照实验。[^src-stgcl]

四条结论。第一，端到端联合学习有效，两阶段对比预训练掉点——PEMS-04 上 GWN 的 MAE 从 19.33 变成 20.22（节点级）/ 20.67（图级），原因是对比学习优化的 uniformity 利于分类、未必保留回归需要的结构。第二，联合学习下图级对比优于节点级；节点级完整时空对比复杂度 $O(M^2N^2)$，按空间与时间维度因子化后降到 $O(M^2+N^2)$。第三，四种增强（边掩码、输入掩码、时域平移、频域 DCT 平滑）调优后效果接近，模型对增强语义不敏感，输入掩码 1% 是最稳的默认值。第四，用 time-of-day 差值阈值 $r_f$ 过滤时间上最接近的假硬负样本，30~60 分钟最佳，120 分钟因负样本过少反而变差。[^src-stgcl]

在 GWN、MTGNN、DCRNN、AGCRN 上一致改善，t 检验 0.05 水平显著比例为图级 83%、节点级 54%，方差普遍小于基线，长预测跨度增益更明显。[^src-stgcl]

## 局限

- 「增强语义不敏感」只覆盖这四种方法在 PEMS-04 / PEMS-08 上的调参结果。
- 预训练微调的负结果限于「基线超参 + 微调学习率 10%」这一设置，未试其他微调策略。
- 论文列出的后续方向：从掩码建模取信号、单一 dropout 当增强、自适应增强、把 $r_f$ 换成动态加权。
- 时间定位：2021–2022 年的早期实证工作；之后的路线转向掩码预训练、时空基础模型与生成式时空建模。本文另有 [[stgcl]] 机制页。

[^src-stgcl]: [[source-stgcl]]
