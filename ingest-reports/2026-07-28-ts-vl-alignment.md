# Ingest 报告：time-series-vision-language-exploring-the-limits-of-alignment

## 创建
- wiki/source-ts-vl-alignment.md — WHY：arXiv:2602.19367 源摘要（近正交、后验 InfoNCE、尺度/ID/不对称/中介、数据与局限）
- wiki/ts-vl-alignment.md — WHY：实体页（框架、结果表意、对外生多模态 ST 含义、与 Time-VLM/Time-MMD 对照）

## 修改
- wiki/time-vlm.md — WHY：谱系表增加对齐几何诊断对照；相关页 + 脚注；source_count 1→2
- wiki/multimodal-time-series-forecasting.md — WHY：新专节「对齐极限」+ 相关概念/脚注；source_count 14→15
- wiki/contrastive-learning.md — WHY：Trimodal limits 小节 + Applications/脚注；source_count 4→5
- wiki/index.md — WHY：登记 source-ts-vl-alignment / ts-vl-alignment（主 Sources/Entities + continued）
- wiki/log.md — WHY：记录 ingest

## 新建交叉链接
- [[ts-vl-alignment]] ↔ [[source-ts-vl-alignment]]
- [[ts-vl-alignment]] ↔ [[multimodal-time-series-forecasting]] / [[contrastive-learning]] / [[time-vlm]]
- [[source-ts-vl-alignment]] → [[time-vlm]] / [[source-time-vlm]] / [[time-mmd]] / [[chronos]] / [[timesfm]]

## 源文件
- raw/time-series-vision-language-exploring-the-limits-of-alignment.pdf（只读；md5 59ac82debd897249a503ca323caf940b）

## 自检
- source-summary 覆盖 abstract–§5 Discussion + App A 局限 + 数据/指标；type source-summary；自引脚注 [^src-ts-vl-alignment] → [[source-ts-vl-alignment]]
- 核心 claim：近正交无耦合；后验投影有限/不对称；ID 饱和；图像中介；外生多模态需显式耦合
- 新 wikilink 目标均已创建；更新页 last_updated/source_count 已调
- 未 git commit / 未改 raw
