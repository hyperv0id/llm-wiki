# Ingest 报告：FrèqFlow/SpectFlow (arXiv 2511.16426, NeurIPS 2025)

## 创建
- **raw/2511.16426v1.md** — 源文件（pdftotext 全文）
- **wiki/source-2511-16426.md** — 源摘要页面。WHY：记录 FrèqFlow 的核心论点、方法论、实验结果和局限性
- **wiki/freqflow-ts.md** — 实体页面。WHY：FrèqFlow（别名 SpectFlow）是一个新的时间序列预测模型实体，与已有 FreqFlow（图像生成）完全不同。使用 freqflow-ts 作为 slug 以避免冲突

## 修改
- **wiki/traffic-forecasting.md** — 新增 Flow Matching / Frequency-Domain 小节，列出 FrèqFlow 的核心方法和实验结果。source_count: 15→16
- **wiki/flow-matching.md** — 在相关页面中添加 [[freqflow-ts]] 链接。source_count: 2→3
- **wiki/generative-time-series-forecasting.md** — 在流匹配方法小节中添加 FrèqFlow，在对比表格中增加一行。source_count: 3→4
- **wiki/index.md** — 在 Sources 和 Entities 类别中各添加一条

## 新建交叉链接
- [[source-2511-16426]] ↔ [[freqflow-ts]]
- [[freqflow-ts]] ↔ [[traffic-forecasting]]
- [[freqflow-ts]] ↔ [[flow-matching]]
- [[freqflow-ts]] ↔ [[generative-time-series-forecasting]]
- [[freqflow-ts]] ↔ [[freqflow]]（对比链接，两个同名但完全不同的模型）

## 注意
- 本文有两个名字：arXiv 标题为 FrèqFlow，pipeline 图注为 SpectFlow。两个名字均在页面中记录
- 本文已被 NeurIPS 2025 接收（39th Conference on Neural Information Processing Systems）