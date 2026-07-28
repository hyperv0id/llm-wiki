# Ingest 报告：rethinking-multimodal-fusion-for-time-series-text-modalities-need-constrained-fusion

## 创建
- wiki/source-constrained-text-fusion.md — WHY：arXiv:2603.22372 源摘要（naive vs constrained、CFA 公式、20K 设定、win rate/效率/toy 瓶颈）
- wiki/constrained-text-fusion.md — WHY：实体页（方法族表、与 TaTS/TiMi/VoT/Time-MMD/Time-VLM/TS–VL 谱系）

## 修改
- wiki/multimodal-time-series-forecasting.md — WHY：专节「Constrained Text Fusion」+ 相关/脚注；source_count 15→16
- wiki/time-mmd.md — WHY：下游表与相关页挂 CFA；source_count 1→2
- wiki/non-fusion-guidance.md — WHY：vs CFA 互补反-naive；source_count 1→2
- wiki/timi.md — WHY：与 CFA 诊断相近/机制正交；脚注与相关页
- wiki/vot.md — WHY：融合哲学对照；source_count 3→4
- wiki/time-vlm.md — WHY：谱系表 + 相关；source_count 2→3
- wiki/ts-vl-alignment.md — WHY：外生文本行补 CFA；脚注/相关
- wiki/index.md — WHY：登记 source-constrained-text-fusion / constrained-text-fusion
- wiki/log.md — WHY：记录 ingest

## 新建交叉链接
- [[constrained-text-fusion]] ↔ [[source-constrained-text-fusion]]
- [[constrained-text-fusion]] ↔ [[time-mmd]] / [[multimodal-time-series-forecasting]] / [[non-fusion-guidance]] / [[timi]] / [[vot]] / [[time-vlm]] / [[ts-vl-alignment]]

## 源文件
- raw/rethinking-multimodal-fusion-for-time-series-text-modalities-need-constrained-fusion.pdf（只读）

## 自检
- 脚注 `[^src-constrained-text-fusion]` → `[[source-constrained-text-fusion]]`；无 `\|` 逃逸 wikilink
- 核心 claim：naive 常 < unimodal；constrained 更稳；CFA 低秩 plug-in；Time-MMD 9 域；>20K 实验
- 未 git / 未改 raw
