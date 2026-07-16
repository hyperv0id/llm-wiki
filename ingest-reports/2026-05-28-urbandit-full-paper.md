# Ingest 报告：UrbanDiT — 完整论文（2026-05-28 重写）

基于完整论文 PDF（arXiv:2411.12164v2）对 2026-05-12 摘要级 ingest 进行全面重写。

## 修改（重写）
- wiki/source-urbandit.md — WHY：从基于摘要+README 的浅层描述 → 基于完整论文的 500+ 词深度 source-summary。新增：数据统一化机制（3D CNN / GCN）、掩码策略公式、统一提示学习三 memory pool 完整实现、rectified flow 训练策略、消融实验、三种模型规模配置、6 grid + 3 graph 数据集统计表、20+ 基线全面对比、关键性能数据（11.3% / 30.4% 提升）、扩展性分析
- wiki/urbandit.md — WHY：从 37 行扩至 100+ 行。新增：架构概览图、掩码公式、三种规模对比表、6 个 grid 数据集统计表、关键性能表、与 UniST/UrbanGPT/MoST/CSDI 的对比分析。修复了未定义的 `[^src-github]` 引用。confidence: medium → high

## 更新
- wiki/spatio-temporal-foundation-model.md — WHY：UrbanDiT 描述从一行扩展为包含 rectified flow 加速、5 任务覆盖、graph+grid 统一的关键细节
- wiki/index.md — WHY：更新 UrbanDiT 条目描述、source 条目标注重写、添加 unified-prompt-learning 到 Techniques 分类

## 创建
- wiki/unified-prompt-learning.md — WHY：UrbanDiT 最核心的技术创新。三 memory pool（时域/频域/空域）+ task mask prompt 的统一提示学习框架，是理解 UrbanDiT 如何实现多任务多数据源统一的关键

## 新建交叉链接
- [[urbandit]] ↔ [[unified-prompt-learning]]
- [[unified-prompt-learning]] ↔ [[source-urbandit]]
- [[urbandit]] → 新增 [[rectified-flow]] 链接

## 置信度变更
- [[source-urbandit]]: confidence 保留 medium（source_count=1，不满足 high 的 ≥2 来源要求；完整论文提升了内容深度但未增加独立来源数量）
- [[urbandit]]: confidence 保留 medium（同上，source_count=1）

## 时间线
1. 2026-05-12：基于摘要 + GitHub README 的初步 ingest（source_count=1, confidence=medium）
2. 2026-05-28：收到完整论文 PDF，全面重写（source_count=1 不变，因为只有一篇源文件，但内容从摘要级升为论文级）