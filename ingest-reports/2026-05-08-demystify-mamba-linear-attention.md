# Ingest 报告：Demystify Mamba in Vision: A Linear Attention Perspective

## 创建
- wiki/source-demystify-mamba-linear-attention-2024.md — WHY：源文件摘要页面，记录论文核心贡献（统一框架+6差异+MILA）
- wiki/mamba.md — WHY：Mamba 是论文的核心研究对象，作为新实体引入 wiki
- wiki/mila.md — WHY：MILA 是论文提出的架构，作为独立实体记录
- wiki/linear-attention-unified-framework.md — WHY：论文的核心理论贡献，统一框架值得独立概念页面
- wiki/forget-gate-in-sequential-models.md — WHY：遗忘门是论文识别的关键机制，且可被位置编码替代的发现对交通预测有启示
- wiki/mamba-block-design.md — WHY：块设计是论文识别的最有影响力设计（+1.8%），值得独立技术页面

## 修改
- wiki/linear-attention-bias.md — WHY：ALiBi 线性偏置与遗忘门的衰减功能相似，添加对比关系
- wiki/generalized-positional-encoding-framework.md — WHY：GPE 框架的 f(n) 直接解释遗忘门为何可被位置编码替代，添加理论联系
- wiki/traffic-forecasting.md — WHY：遗忘门的衰减模式对应交通中邻近传感器的强关联，添加交通特化 PE 的启示
- wiki/glu-gated-linear-unit.md — WHY：Mamba 输入门本质是 GLU 变体，添加技术联系
- wiki/index.md — WHY：添加 6 个新页面的索引条目
- wiki/log.md — WHY：记录 ingest 活动

## 新建交叉链接
- [[mamba]] ↔ [[mila]]
- [[mamba]] ↔ [[linear-attention-unified-framework]]
- [[mamba]] ↔ [[forget-gate-in-sequential-models]]
- [[mamba]] ↔ [[mamba-block-design]]
- [[forget-gate-in-sequential-models]] ↔ [[linear-attention-bias]]
- [[forget-gate-in-sequential-models]] ↔ [[generalized-positional-encoding-framework]]
- [[linear-attention-unified-framework]] ↔ [[linear-attention-bias]]
- [[glu-gated-linear-unit]] ↔ [[mamba]]
