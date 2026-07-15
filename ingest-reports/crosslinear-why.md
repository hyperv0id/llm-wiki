# Ingest 报告：CrossLinear

**Source PDF (now in raw/)**:  
`raw/crosslinear-kdd-2025.pdf`

**Slug**: `src-crosslinear`  
**Date**: 2026-07-13  
**Venue**: KDD 2025 | arXiv:2505.23116v1

## 创建

- `wiki/source-crosslinear.md` — WHY：source-summary，记录 many-to-one 外生范式、CI/CD 过拟合论断、架构、12 数据集实验与局限。
- `wiki/crosslinear.md` — WHY：方法实体页，便于与 TimeXer / ExoST / ExoLLM / ExoTST / GCGNet 外生谱系交叉链接。
- `wiki/cross-correlation-embedding.md` — WHY：核心即插即用技术（1D conv + \(\alpha\) residual mix），可被其他 CI 骨干复用。

## 修改

- `wiki/index.md` — WHY：登记新建 source / entity / technique 条目。
- `wiki/log.md` — WHY：追加 2026-07-13 ingest 记录。
- `wiki/source-timexer.md` — WHY：反向链接（同协议 many-to-one 外生基准与 Linear 轻量对照）。
- `wiki/source-exost.md` — WHY：外生谱系交叉链接。
- `wiki/source-exollm.md` — WHY：外生谱系交叉链接。
- `wiki/source-exotst.md` — WHY：外生谱系交叉链接。
- `wiki/source-gcgnet.md` — WHY：GCGNet 已将 CrossLinear 列为 baseline，补正式 wikilink。
- `wiki/gcgnet.md` — WHY：Related 列表补 CrossLinear。
- `wiki/channel-independence.md` — WHY：论文核心是 CI 骨干 + 轻量 CD 注入，补充该折中路径。

## 新建交叉链接

- [[source-crosslinear]] ↔ [[crosslinear]] ↔ [[cross-correlation-embedding]]
- [[source-crosslinear]] ↔ [[source-timexer]] / [[source-exost]] / [[source-exollm]] / [[source-exotst]] / [[source-gcgnet]]
- [[cross-correlation-embedding]] ↔ [[channel-independence]] / [[patch-based-tokenization]]

## 未创建（有意收窄）

- 未为 12 个数据集各自建 entity。
- 未新建独立 baseline 页（TimeXer/SparseTSF/RLinear 等已有或可复用）。
- 未修改 `raw/`（外部 PDF 只读）。
- 未扩展 `multimodal-exogenous-guided-long-term-st-forecasting` 分析页（任务仅要求 ingest + 与既有外生源交叉链接）。

## 2026-07-16 补充

- PDF 已拷贝至 `raw/crosslinear-kdd-2025.pdf`（之前仅为外部路径，未入 raw/）。
- 更新 frontmatter `last_updated` 至 2026-07-16。
- 追加 log 记录 PDF 入库。
