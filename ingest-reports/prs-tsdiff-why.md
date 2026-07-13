# Ingest 报告：Predict, Refine, Synthesize / TSDiff (2307.11494)

## 创建
- wiki/source-prs.md — WHY：source-summary，记录 NeurIPS 2023 无条件时序扩散 + self-guidance 论文核心论点、方法、实验与局限；slug `src-prs`
- wiki/tsdiff.md — WHY：实体页面，TSDiff 模型（Predict / Refine / Synthesize）
- wiki/observation-self-guidance.md — WHY：技术页面，推理期观测自引导（MS / Quantile），对照 classifier guidance 与 CFG
- wiki/linear-predictive-score.md — WHY：技术页面，LPS 合成样本预测质量指标

## 修改
- wiki/index.md — WHY：Sources / Entities / Techniques 登记新页面
- wiki/log.md — WHY：追加 ingest 日志
- wiki/generative-time-series-forecasting.md — WHY：在扩散方法谱系中插入 TSDiff，表格与相关链接；source_count 14→15
- wiki/source-timegrad.md — WHY：后续影响加入 TSDiff 对照
- wiki/timegrad.md — WHY：关联页面与后续推动链接 TSDiff；source_count 7→8
- wiki/source-csdi.md — WHY：贡献/后续中注明 TSDiff 直接对比 CSDI
- wiki/csdi.md — WHY：后续影响与关联页面加入 TSDiff / self-guidance；source_count 9→10
- wiki/source-tsflow.md — WHY：标明 TSFlow 在 TSDiff 无条件→条件路线之后；链接 TSDiff；source_count 1→2
- wiki/tsflow.md — WHY：将纯文本 TSDiff 改为 wikilink；补充 observation-self-guidance / LPS；source_count 1→2
- wiki/classifier-guidance.md — WHY：后续发展补充 self-guidance 第三条路线；source_count 3→4
- wiki/classifier-free-guidance.md — WHY：区分 CFG 与 observation self-guidance；source_count 6→7

## 新建交叉链接
- [[source-prs]] ↔ [[tsdiff]] ↔ [[observation-self-guidance]] ↔ [[linear-predictive-score]]
- [[tsdiff]] ↔ [[source-timegrad]] / [[timegrad]]
- [[tsdiff]] ↔ [[source-csdi]] / [[csdi]]
- [[tsdiff]] ↔ [[source-tsflow]] / [[tsflow]]
- [[observation-self-guidance]] ↔ [[classifier-guidance]] / [[classifier-free-guidance]]
- [[tsdiff]] ↔ [[generative-time-series-forecasting]]

## 源文件
- 仓库内：`raw/2307.11494.pdf`（不可变，已存在）
- 外部任务路径：`/run/media/jcheng/WD-Data/yjs/INSIS/多模态外生信息引导的长期时空预测/2307.11494.pdf`
