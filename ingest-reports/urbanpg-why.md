# Ingest 报告：UrbanPG (AAAI 2026)

## 创建
- wiki/source-urbanpg.md — WHY：新源文件首次摄入，为标准 source-summary 页面
- wiki/urbanpg.md — WHY：UrbanPG 是全新的时空预测架构（提示-骨干解耦 + 线性注意力 + 三范式统一），在所有现有页面中无等价物，需创建为 technique 页面

## 修改
- wiki/index.md — WHY：在 Sources、Techniques、Entities 三个分组中添加 UrbanPG 条目（含类型双归属，提升图谱连通性）
- wiki/log.md — WHY：按 wiki 规范记录摄取活动
- wiki/spatio-temporal-foundation-model.md — WHY：UrbanPG 是时空基础模型的重要新成员，解耦个性化/通用的设计思想为该概念引入新维度，需在 Existing Models 列表和 Related Pages 中添加引用
- wiki/traffic-forecasting.md — WHY：UrbanPG 在大规模交通预测上取得 SOTA（8600 节点、48-72% 效率提升），填补了 Foundation Model 小节中关于提示-骨干解耦方案的空白
- wiki/urbanfm.md — WHY：UrbanPG 是 UrbanFM 的关键对比——UrbanPG 不支持多任务训练是 UrbanFM 设计动机的证据，双向引用强化论证链
- wiki/urbangpt.md — WHY：UrbanPG 在 Comparison 表中作为新行加入，补充 LLM 范式之外的高效轻量方案，丰富图谱对比维度
- wiki/opencity.md — WHY：UrbanPG 与 OpenCity 共享"轻量化+可扩展"的设计哲学，但实现路径不同（OpenCity 用 Transformer+GNN，UrbanPG 用 prompt-backbone 解耦）
- wiki/bigcity.md — WHY：BIGCity 的多模态统一与 UrbanPG 的范式解耦构成了对比——一个横向扩展模态，一个纵向扩展学习范式
- wiki/linear-attention-unified-framework.md — WHY：UrbanPG 的 STCA 模块直接使用了 Performers 的随机特征映射线性注意力，是该统一框架在 STGNN 领域的最新实例，需在 Related 中添加引用

## 新建交叉链接
- [[urbanpg]] ↔ [[spatio-temporal-foundation-model]]
- [[urbanpg]] ↔ [[traffic-forecasting]]
- [[urbanpg]] ↔ [[urbanfm]]
- [[urbanpg]] ↔ [[urbangpt]]
- [[urbanpg]] ↔ [[opencity]]
- [[urbanpg]] ↔ [[bigcity]]
- [[urbanpg]] ↔ [[linear-attention-unified-framework]]
- [[urbanpg]] ↔ [[uniflow]]
- [[urbanpg]] ↔ [[unist]]
- [[urbanpg]] ↔ [[urbandit]]
- [[urbanpg]] ↔ [[std-mae]]
- [[urbanpg]] ↔ [[gwnet]]
- [[urbanpg]] ↔ [[stid]]
