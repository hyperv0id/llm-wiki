# Ingest 报告：alphaflow-understanding-and-improving-meanflow-models-arxiv25

源文件：`raw/alphaflow-understanding-and-improving-meanflow-models-arxiv25.pdf`（arXiv:2510.20771 v1，2025-10-23，Snap Research / University of Michigan）

## 创建
- wiki/source-alphaflow.md — WHY：源摘要页，概括分解定理、梯度冲突实证、α 目标族与课程退火，及论文自述局限（300-500 字口径）。
- wiki/alphaflow.md — WHY：α-Flow 技术页，覆盖 MeanFlow 分解、L_α 目标、定理 1 统一视角（含 Shortcut/MeanFlow/CT 特例）、课程调度、作者报告实验与局限。
- wiki/meanflow.md — WHY：ingest AlphaFlow 时 MeanFlow 原文尚未单独收录，按 α-Flow 论文的转述建立占位技术页（带明确「来源限制」callout，声明机制描述转述自 α-Flow）；raw/ 中 MeanFlow PDF 已于同日补齐，待单独 ingest 后升级为直接引用。

## 修改
- wiki/shortcut-models.md — WHY：新增「与 α-Flow 的统一关系」节：α=1/2 时 L_SC = ½L_α，Shortcut 训练是 α-Flow 目标的特例（定理 1 口径）。
- wiki/average-velocity-modeling.md — WHY：新增「与 MeanFlow 的关系」节：CoGenCast 的 JVP 修正目标与 MeanFlow 训练目标同构，α-Flow 将其分解并报告梯度强负相关。

## 新建交叉链接
- [[alphaflow]] ↔ [[meanflow]]
- [[alphaflow]] ↔ [[shortcut-models]]
- [[alphaflow]] ↔ [[loft]]（LOFT 将 α-Flow 作为轨迹矫正基线，冲突处理方式对比）
- [[alphaflow]] ↔ [[average-velocity-modeling]]（平均速度建模同构目标）
- [[meanflow]] ↔ [[consistency-models]]、[[shortcut-models]]、[[flow-matching]]

## 备注
- 本次 ingest 与 GiFlow、Consistency-FM 同日进行；上一会话在 AlphaFlow 收尾（log/index/报告）前中断，本报告为收尾补记。
- MeanFlow（raw/geng-meanflow-one-step-generative-arxiv-2025.pdf）计划单独 ingest，届时 [[meanflow]] 页应从转述升级为原文直接引用。
