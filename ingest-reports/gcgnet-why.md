# Ingest 报告：GCGNet

**Source PDF (immutable, external path; not copied into raw/)**:  
`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/GCGNet_Li_2026_ICLR.pdf`

**Slug**: `src-gcgnet`  
**Date**: 2026-07-13  
**Venue**: ICLR 2026 | arXiv:2603.08032v2

## 创建

- `wiki/source-gcgnet.md` — WHY：source-summary，记录 GCGNet 问题设定、三模块架构、12 数据集实验与局限。
- `wiki/gcgnet.md` — WHY：方法实体页，便于与 TimeXer / KITE / DAG 外生谱系交叉链接。
- `wiki/joint-temporal-channel-correlation.md` — WHY：论文核心概念——联合时间–通道相关 vs 两步串行建模。
- `wiki/graph-structure-aligner.md` — WHY：关键技术，patch 图 VAE + L1 结构对齐损失。
- `wiki/graph-refiner.md` — WHY：关键技术，top-k 稀疏 + GCN 精炼并防 Graph VAE 退化。
- `wiki/variational-generator-exogenous.md` — WHY：关键技术，VAE 粗预测与可选未来外生替换。

## 修改

- `wiki/index.md` — WHY：登记新建 source / entity / concept / technique 条目。
- `wiki/log.md` — WHY：追加 2026-07-13 ingest 记录。
- `wiki/source-timexer.md` — WHY：反向链接（论文将 TimeXer 归为 temporal→channel 两步策略）。
- `wiki/source-kite.md`、`wiki/kite.md` — WHY：同 ECNU 外生谱系对照链接。
- `wiki/dag.md` — WHY：同为确定性外生预测的邻近工作对照。

## 新建交叉链接

- [[source-gcgnet]] ↔ [[gcgnet]]
- [[gcgnet]] ↔ [[joint-temporal-channel-correlation]]
- [[gcgnet]] ↔ [[graph-structure-aligner]] ↔ [[graph-refiner]] ↔ [[variational-generator-exogenous]]
- [[source-gcgnet]] ↔ [[source-timexer]] / [[source-exotst]] / [[source-exost]] / [[source-kite]] / [[source-dag]]
- [[kite]] / [[dag]] ↔ [[gcgnet]]

## 未创建（有意收窄）

- 未为 12 个数据集各自建 entity（仅在 source-summary 中汇总）。
- 未新建独立 baseline 页（TimeXer/PatchTST 等已存在）。
- 未修改 `raw/`（外部 PDF 只读）。

## 维护记录

- **2026-07-18** — 修复 source-gcgnet.md 交叉引用：TFT wikilink 从 `[[source-timexer|TFT]]` 修正为 `[[source-tft|TFT]]`（不同模型），CrossLinear 从纯文本改为 `[[source-crosslinear|CrossLinear]]`。
