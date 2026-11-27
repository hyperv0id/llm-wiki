---
title: "Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series"
type: source-summary
tags:
  - multimodal-time-series
  - irregular-time-series
  - benchmark
  - dataset
  - time-series-forecasting
  - neurips-2025
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series

真实世界的多变量时间序列往往采样疏密不均、通道间时间戳互不同步，还夹杂零散出现的文本日志，而既有基准大多假设规则网格与已对齐的模态，模型一落地就水土不服[^src-time-imm]。论文的关键观察是不规则性有成因：事件触发的记录、受条件限制的采样、系统故障留下的残缺，三类机制产生性质不同的不规则模式，混作一团评测会掩盖模型的真实短板[^src-time-imm]。为此论文构建 Time-IMM 数据集，按成因组织九个亚类的异步多模态序列，并配套 IMM-TSF 评测框架，把「数值模型如何接收异步文本」拆成时间融合与模态融合两个可替换槽位逐一检验[^src-time-imm]。主结论：接入文本平均使 MSE 下降约 6.7%，文本信息量大的场景降幅接近 38%[^src-time-imm]。

## 核心做法

- **成因驱动数据集**：九个亚类分属触发（trigger-based）、约束（constraint-based）、故障（artifact-based）三类机制；文本日志统一经大语言模型提炼为五句摘要以压长度去冗余[^src-time-imm]。
- **预对齐**（pre-alignment）：对需要规则输入的常规数值模型，按全局时间网格与观测掩码对齐，保留时间戳与缺失掩码，而非插值抹平不规则性[^src-time-imm]。
- **时间戳到文本融合**（Timestamp-to-Text Fusion，TTF）：对比按时间差衰减的高斯加权平均（Recency-Weighted Averaging）与 Time2Vec 时间编码加交叉注意力（Time2Vec-Augmented Cross-Attention）[^src-time-imm]。
- **模态融合**（Multimodality Fusion，MMF）：对比门控循环单元（GRU）门控残差相加（GRU-Gated Residual Addition）与固定混合权重的交叉注意力相加（Cross-Attention Addition）；文本编码器一律冻结[^src-time-imm]。

## 证据

1. **定位与主结论**：接入文本是普适涨点，平均降 MSE 约 6.7%，文本信息量大的数据集近 38%；不规则骨干以 t-PatchGNN 最强[^src-time-imm]。
2. **真实涨点归因**：GRU 门控残差相加一致优于固定权重交叉注意力相加，门控对噪声文本的抑制是关键；时间融合两路径与文本编码器规模均无显著差异，涨点来自接入文本本身而非融合算子；摘要流水线与编码器微调未消融[^src-time-imm]。
3. **证据缺口与负结果**：增益与文本的预测价值绑定而非与不规则类型绑定（FNSPID 数值端已够规则，文本增益有限）；数据集规模偏小，作者自认或压低大编码器优势[^src-time-imm]。

## 局限

论文自述的边界：预测假设未来查询时刻已知；仅覆盖预测任务，异常检测、分类、检索未纳入；次模态仅文本，图像、音频等未入基准；每个数据集只含单一不规则成因，复合不规则留作后续工作[^src-time-imm]。

## 相关页面

- [[time-mmd|Time-MMD]] — 瓶颈接力 $\Delta$：前序基准补齐多领域数值–文本对但停留规则网格，本文把基准推进到非均匀采样与通道异步。
- [[timesx|TimesX]] — 同槽位互斥：同属补齐 Time-MMD 的基准分支，TimesX 聚焦规则日/周频的真实时钟上下文与截止时间隔离，本文专攻异步文本日志维度。
- [[constrained-text-fusion|受控文本融合]] — 复现对齐：门控受控注入优于无差别注入的实证在两个独立来源中相互印证。
- [[multimodal-time-series-forecasting|多模态时间序列预测]] — 特例：本文是该概念谱系下面向不规则异步场景的基准实例。

出版信息：NeurIPS 2025 Datasets and Benchmarks Track；Ching Chang 等（UCLA、NYCU）。

[^src-time-imm]: [[source-time-imm]]
