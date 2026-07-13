# Ingest 报告：ProbTS

**Source PDF (immutable)**:  
`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/2310.07446.pdf`  
（仓库内镜像：`raw/2310.07446.pdf`，仅只读）

**Slug**: `src-probts`  
**Date**: 2026-07-13  
**Venue**: NeurIPS 2024 Datasets and Benchmarks | arXiv:2310.07446v5

## 创建

- `wiki/source-probts.md` — WHY：source-summary，记录统一点/分布、短/长 horizon 基准动机、数据特征、方法轴与主要发现。
- `wiki/probts.md` — WHY：基准工具实体，便于与模型/指标/解码概念交叉链接。
- `wiki/ar-vs-nar-decoding.md` — WHY：论文核心方法轴之一，贯穿经典模型与 TSFM 分析。
- `wiki/non-gaussianity.md` — WHY：论文提出的窗口分布复杂度量化指标，解释点 vs 概率方法场景偏好。

## 修改

- `wiki/index.md` — WHY：登记新建 source / entity / concept 条目。
- `wiki/log.md` — WHY：追加 2026-07-13 ingest 记录。
- `wiki/instance-normalization.md` — WHY：补充 ProbTS 对 RevIN vs 均值缩放的跨场景结论。
- `wiki/timegrad.md` — WHY：作为 AR 概率代表在 ProbTS 长程误差累积/强季节发现中被引用。
- `wiki/csdi.md` — WHY：作为 NAR 概率代表与长程显存/效率、复杂分布基线角色。
- `wiki/patchtst.md` — WHY：作为长程 NAR 点预测代表，短程/强季节对比。
- `wiki/timesfm.md` — WHY：AR 基础模型在 ProbTS 零样本 horizon 分析中的代表。
- `wiki/chronos.md` — WHY：量化分布头基础模型在复杂分布上的局限。
- `wiki/generative-time-series-forecasting.md` — WHY：补统一基准与长程概率预测开放问题。

## 新建交叉链接

- [[source-probts]] ↔ [[probts]] ↔ [[ar-vs-nar-decoding]] ↔ [[non-gaussianity]]
- [[probts]] ↔ [[timegrad]] / [[csdi]] / [[patchtst]] / [[timesfm]] / [[chronos]]
- [[instance-normalization]] ↔ [[source-probts]]（RevIN vs mean scaling）
- [[generative-time-series-forecasting]] ↔ [[probts]]

## 未创建（有意收窄）

- 未为 Lag-Llama / Timer / MOIRAI / UniTS / N-HiTS 等各自新建 entity（已有 TimesFM/Chronos/PatchTST 等代表即可）。
- 未为每个数据集建页。
- 未修改 `raw/`。
- 未新建独立 analysis 页（任务为标准 ingest，非 query 归档）。
