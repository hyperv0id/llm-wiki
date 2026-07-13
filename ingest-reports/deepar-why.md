# Ingest 报告：DeepAR (Salinas et al., arXiv:1704.04110)

**PDF**:  
`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/1704.04110.pdf`  
（`raw/1704.04110.pdf` 已存在且与外部 PDF 字节一致；按不可变策略未改 raw/）

**Slug**: `src-deepar`  
**Date**: 2026-07-13  
**Title**: DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks  
**Authors**: David Salinas, Valentin Flunkert, Jan Gasthaus (Amazon Research)  
**Version used**: arXiv:1704.04110v3 [cs.AI] 22 Feb 2019

## 创建

- `wiki/source-deepar.md` — WHY：source-summary；记录全局 AR-RNN、高斯/负二项似然、幂律尺度处理、MC 轨迹分位数、实验与局限。
- `wiki/deepar.md` — WHY：方法实体，便于与 DeepState / TimeGrad / ProbTS / TFT 等交叉链接。

## 修改

- `wiki/index.md` — WHY：登记 source 与 entity。
- `wiki/log.md` — WHY：追加 2026-07-13 ingest 记录。
- `wiki/generative-time-series-forecasting.md` — WHY：在概率谱系中补 AR-RNN/DeepAR 分支（此前仅有扩散/流/SSM 等）。
- `wiki/deep-state-space-model.md` — WHY：对照表中 DeepAR 改为正式 wikilink。
- `wiki/deepstate.md` — WHY：定位表与关联中链到 [[deepar]]。
- `wiki/source-deepstate.md` — WHY：正文已对照 DeepAR，补 wikilink。
- `wiki/timegrad.md` — WHY：“继承自 DeepAR”改为正式 wikilink。
- `wiki/ar-vs-nar-decoding.md` — WHY：短程概率 AR 代表链到 [[deepar]]，并增加脚注引用。

## 新建交叉链接

- [[source-deepar]] ↔ [[deepar]]
- [[deepar]] ↔ [[generative-time-series-forecasting]] / [[ar-vs-nar-decoding]] / [[deepstate]] / [[timegrad]] / [[probts]]
- [[source-deepar]] ↔ [[deep-state-space-model]]（经对照表）

## 未创建（有意收窄）

- 未新建独立 “negative-binomial-likelihood” / “velocity-weighted-sampling” 技术页（信息收敛在 source/entity）。
- 未为 parts/electricity/traffic/ec 数据集建页。
- 未扩展 multimodal-exogenous 分析主线（DeepAR 为数值协变量 + 单变量相关序列集合）。
- 未修改 `raw/1704.04110.pdf`。
