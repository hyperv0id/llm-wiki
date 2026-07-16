# Ingest 报告：time-moe

Date: 2026-07-25

## 创建
- **wiki/source-time-moe.md** — WHY：Time-MoE 论文的 source-summary 页面，摘要 decoder-only + Sparse MoE 时序基础模型架构、Time-300B 数据集、2.4B 参数缩放、零样本/全样本实验结果
- **wiki/time-moe.md** — WHY：Time-MoE 模型系列 entity 页面，详细记录三档模型配置（base/large/ultra）、架构设计选择、训练细节、与 Moirai-MoE 的对比
- **wiki/time-300b.md** — WHY：Time-300B 数据集 entity 页面，记录 309B 时间点、9+ 领域组成、数据清洗管线、与同类数据集规模对比

## 修改
- **wiki/mixture-of-experts.md** — WHY：修正"首个 Sparse MoE 时序基础模型"的历史归属——Time-MoE (ICLR 2025) 早于 Moirai-MoE (ICML 2025)；新增 Time-MoE 专节（架构/门控/token 化/多分辨率预测/缩放定律）；添加演进 note callout
- **wiki/moirai-moe.md** — WHY：更新"与 Time-MoE 的关系"章节——从"同期工作"改为"先于 Moirai-MoE 的工作"，添加双向对比表，引用 Time-MoE 自源
- **wiki/source-moirai-moe.md** — WHY：更新 frontmatter（source_count: 1, confidence: high）
- **wiki/token-level-specialization.md** — WHY：新增"与 Time-MoE 的对比"章节，区分两种专业化哲学（隐式数据驱动 vs 显式结构驱动），引用 Time-MoE 源
- **wiki/index.md** — WHY：添加 [[source-time-moe]] 到 Sources、[[time-300b]] 和 [[time-moe]] 到 Entities
- **wiki/log.md** — WHY：记录本次 ingest 操作

## 新建交叉链接
- [[time-moe]] ↔ [[mixture-of-experts]]
- [[time-moe]] ↔ [[moirai-moe]]
- [[time-moe]] ↔ [[time-300b]]
- [[time-moe]] ↔ [[token-level-specialization]]
- [[time-300b]] ↔ [[timebench]]
- [[time-moe]] ↔ [[source-time-moe]]
