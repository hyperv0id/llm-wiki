# Ingest 报告：jmlr-25-1243 (FusionBench)

## 创建
- wiki/source-jmlr-25-1243.md — WHY：JMLR 26 FusionBench 源摘要（三分法、Table 1 方法族、CLIP/ResNet/GPT-2/Flan-T5/Mistral 设定、主表数字、泛化/鲁棒/成本）
- wiki/fusionbench.md — WHY：实体页（三模块架构、方法覆盖、关键 AVG、与 MergeKit/训时 MoE 谱系）
- wiki/deep-model-fusion.md — WHY：概念页（ensemble/merging/mixing 分类学 + learn-from-model）
- wiki/model-merging.md — WHY：技术页（合并机制族、B/32 数字、预设与风险）
- wiki/task-arithmetic.md — WHY：技术页（任务向量定义、正交证据、与 Ties/RegMean/Ada 对照）
- notes/20260731T204831--paper-fusionbench__paper.org — WHY：ljg-paper 中文精读（八个 CLIP 专才锚点）
- notes/images/20260731T204831--paper-fusionbench-overview.png — WHY：Figure 1 框架图

## 修改
- wiki/mixture-of-experts.md — WHY：专节「事后模型混合中的 MoE」对照 WE-MoE/SMILE vs 训时 MoE；source_count 8→9；相关技术链 FusionBench
- wiki/index.md — WHY：登记 source-jmlr-25-1243 / fusionbench / deep-model-fusion / model-merging / task-arithmetic；last_updated
- wiki/log.md — WHY：记录 ingest

## 新建交叉链接
- [[fusionbench]] ↔ [[source-jmlr-25-1243]]
- [[fusionbench]] ↔ [[deep-model-fusion]] / [[model-merging]] / [[task-arithmetic]] / [[mixture-of-experts]]
- [[deep-model-fusion]] ↔ [[model-merging]] / [[task-arithmetic]] / [[fusionbench]]
- [[model-merging]] ↔ [[task-arithmetic]] / [[deep-model-fusion]]
- [[mixture-of-experts]] → [[fusionbench]] / [[deep-model-fusion]] / [[model-merging]] / [[source-jmlr-25-1243]]

## 源文件
- raw/jmlr-25-1243.pdf（只读；md5 3e7e1b849a730cb755e9ad041828430e；未改名/未修改）

## 自检
- source-summary：source_count 0、confidence low（仓库惯例）；无自引脚注循环
- 实体/概念/技术：source_count 1、confidence medium；脚注 `[^src-jmlr-25-1243]` → `[[source-jmlr-25-1243]]`
- 关键 claim 对齐 PDF Appendix E：B/32 AVG Soup 66.5 / TA 68.0 / Ties 72.2 / RegMean++ 84.4 / layer Ada 82.6 / WEMoE 89.2 / SMILE 89.3 / MTL 88.6 / STL 90.3
- 无 `|` 逃逸 wikilink；未 git commit；未改 raw
- 影响面：wiki 内原先无 model-merging / task-arithmetic / deep-model-fusion 页——本轮新建主题域，仅反向钩到 mixture-of-experts
