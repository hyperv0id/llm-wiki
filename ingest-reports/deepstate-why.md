# Ingest 报告：DeepState (Rangapuram et al., NeurIPS 2018)

**Intended path (mislabeled on disk)**:  
`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/deepstate-rangapuram-2018.pdf`

**Content verification**: The file at the above path (and `raw/deepstate-rangapuram-2018.pdf`) is **not** DeepState. `pdftotext` shows it is *The module embedding theorem via towers of algebras* (Coles, Huston, Penneys, Srinivas; arXiv:1810.07049, math.OA, 40 pages). Per `raw/` immutability, that file was **not** modified or deleted.

**Authoritative PDF used for this ingest**:  
NeurIPS 2018 proceedings  
`https://proceedings.neurips.cc/paper_files/paper/2018/file/5cf68969fb67aa6082363a6d4e6468e2-Paper.pdf`  
Title: *Deep State Space Models for Time Series Forecasting*  
Authors: Syama Sundar Rangapuram, Matthias Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, Tim Januschowski  
Local extract cache: `/tmp/deepstate-ingest/deepstate-rangapuram-2018-correct.pdf` (10 pages)

**Slug**: `src-deepstate`  
**Date**: 2026-07-13  
**Venue**: NeurIPS 2018

## 创建

- `wiki/source-deepstate.md` — WHY：source-summary；记录 RNN→线性 SSM 参数、Kalman 似然、相对 DeepAR 的非自回归输入设计、实验与局限。
- `wiki/deepstate.md` — WHY：方法实体，便于与 Kalman / K²VAE / 概率预测谱系交叉链接。
- `wiki/deep-state-space-model.md` — WHY：抽象“深度状态空间模型”概念，区分经典 SSM、DeepState 式参数化与后续神经 Kalman。

## 修改

- `wiki/index.md` — WHY：登记 source / entity / concept。
- `wiki/log.md` — WHY：追加 2026-07-13 ingest 记录（含 PDF 错标说明）。
- `wiki/kalman-filter.md` — WHY：正文已点名 DeepState，补正式 wikilink 与 `src-deepstate` 引用。
- `wiki/generative-time-series-forecasting.md` — WHY：补 SSM/DeepState 分支，避免概率预测谱系只有扩散/流。

## 新建交叉链接

- [[source-deepstate]] ↔ [[deepstate]] ↔ [[deep-state-space-model]]
- [[source-deepstate]] ↔ [[kalman-filter]] / [[generative-time-series-forecasting]] / [[k2vae]] / [[timegrad]]

## 未创建（有意收窄）

- 未新建 DeepAR source/entity（本任务源文件为 DeepState；DeepAR 仅作对照提及）。
- 未为 electricity/traffic/M4/tourism/parts 数据集建页。
- 未修改 `raw/deepstate-rangapuram-2018.pdf`（不可变；且内容错误）。
- 未扩展 `multimodal-exogenous-guided-long-term-st-forecasting` 分析页（DeepState 为数值协变量+SSM，非多模态主线）。
