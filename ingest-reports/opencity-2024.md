# Ingest 报告：OpenCity (2024)

## 创建
- wiki/source-opencity.md — WHY：新 source-summary，覆盖 OpenCity 论文的核心贡献（instance norm, patch embedding, TimeShift Transformer）、实验结果（6 数据集零样本，4/6 超越全量基线）、局限性和关键架构公式
- ingest-reports/opencity-2024.md — WHY：WHY 报告，记录本次 ingest 的所有变更决策

## 修改
- wiki/opencity.md — WHY：从 stub（28 行, type=entity, 仅引用 [[source-most]]）重写为完整技术页面（~130 行, type=technique, 引用 [[source-opencity]]），覆盖架构（嵌入层、上下文编码、TimeShift Transformer、GCN 聚合、输出）、预训练设置、关键结果表格、消融洞察、部署效率、局限性和血脉谱系
- wiki/spatio-temporal-foundation-model.md — WHY：增强 OpenCity 条目从一句话到详细描述（TimeShift Transformer, Laplacian eigenvectors, instance norm），添加 [^src-opencity] 引用，source_count 3→4
- wiki/traffic-forecasting.md — WHY：在 Foundation Model 小节新增 OpenCity 条目，添加 [^src-opencity] 引用，source_count 19→20
- wiki/most.md — WHY：last_updated 更新为 2026-05-31
- wiki/index.md — WHY：在 Sources 添加 [[source-opencity]]，更新 Entities 中 [[opencity]] 摘要
- wiki/log.md — WHY：追加 ingest 条目

## 新建交叉链接
- [[source-opencity]] ↔ [[opencity]]
- [[opencity]] → [[gpt-st]], [[spatio-temporal-foundation-model]], [[most]], [[urbandit]], [[traffic-forecasting]], [[urbandit-paper-river]], [[source-gpt-st]], [[patchtst]]
- [[spatio-temporal-foundation-model]] → [[source-opencity]]
- [[traffic-forecasting]] → [[source-opencity]]
