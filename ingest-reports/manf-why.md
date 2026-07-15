# Ingest 报告：MANF — Multi-scale Attention Flow for Probabilistic Time Series Forecasting

## 状态
**补全已部分 ingest 的条目**（wiki 页面已于 2026-07-13 创建）。本次补充：PDF 入库 + ingest 报告。

## 创建
- `raw/manf-multi-scale-attention-flow.pdf` — WHY：论文 PDF 此前未拷贝到 raw/，导致 source count 不一致；现补全

## 已有（2026-07-13 已创建，本次验证通过）
- `wiki/source-maf.md` — WHY：源摘要，Feng et al. arXiv:2205.07493，MANF = 多尺度注意力（动态相对位置）编码器 + 条件 RealNVP NAR 生成
- `wiki/manf.md` — WHY：模型实体，NAR 概率预测架构，六数据集 CRPS-sum/MSE SOTA
- `wiki/multi-scale-attention.md` — WHY：核心技术创新，尺度随层加深、动态相对位置编码的注意力机制
- `wiki/normalizing-flow.md` — WHY：更新以包含 MANF 中的条件 RealNVP 用法与 NAR vs AR 流对比

## 修改
- `wiki/log.md` — WHY：追加本次补全记录
- `wiki/manf.md`、`wiki/source-maf.md`、`wiki/generative-time-series-forecasting.md` — WHY：lint 修复，“后续 TimeGrad”事实错误（TimeGrad ICML 2021 早于 MANF arXiv 2022），改“后续”为“相对”/“同期”/“并行”
- ~~无 wiki 页面内容修改~~ — 2026-07-19 lint 修复了 3 处"后续 TimeGrad"的时间顺序错误（见上方）

## 已有交叉链接（2026-07-13）
- [[manf]] ↔ [[multi-scale-attention]]
- [[manf]] ↔ [[normalizing-flow]]
- [[manf]] ↔ [[ar-vs-nar-decoding]]
- [[manf]] → [[generative-time-series-forecasting]]
- [[manf]] → [[timegrad]]
- [[multi-scale-attention]] → [[generative-time-series-forecasting]]

## 未创建（不需创建）
- RealNVP（已在 normalizing-flow 中充分覆盖）
- CRPS-sum（评估指标，不足以独立成页）
- NKF / HMGT（baseline 模型，非本文贡献）
