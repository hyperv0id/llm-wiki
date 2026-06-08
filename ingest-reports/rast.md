# Ingest 报告：RAST (2508.16623)

## 创建
- wiki/source-rast.md — WHY：源文件摘要页面，记录 RAST 论文的核心贡献、架构设计和关键结果
- wiki/rast.md — WHY：RAST 是首个 RAG-for-STF 框架，引入检索增强的全新范式，在 6 个数据集上 SOTA
- wiki/retrieval-augmented-spatio-temporal-forecasting.md — WHY：RAG-for-STF 是一个全新的跨领域概念（NLP RAG → 时空预测），有明确的理论边界和设计空间，值得独立归档
- wiki/spatio-temporal-retrieval-store.md — WHY：双维度 FAISS 记忆库 + 动量管理是 RAST 的核心工程创新，具有可复用的技术价值
- wiki/dual-dimension-feature-disentanglement.md — WHY：时间/空间双流解耦编码是 RAST 实现双维度检索的前提设计，低秩分解理论基础扎实

## 修改
- wiki/gtr.md — WHY：GTR 和 RAST 同为检索增强的时序预测方法，但检索维度不同（仅时间 vs 双维度），需要双向交叉引用
- wiki/ragc.md — WHY：RAGC 和 RAST 在 LargeST 上均有评估，两者从不同角度解决大规模问题（正则化 vs 检索增强）
- wiki/index.md — WHY：新增 5 个页面的索引条目
- wiki/log.md — WHY：记录本次 ingest 操作

## 新建交叉链接
- [[rast]] ↔ [[gtr]]（同为检索增强预测，维度不同）
- [[rast]] ↔ [[ragc]]（同为大规模路网方法，策略不同）
- [[rast]] ↔ [[retrieval-augmented-spatio-temporal-forecasting]]
- [[retrieval-augmented-spatio-temporal-forecasting]] ↔ [[gtr]]
- [[spatio-temporal-retrieval-store]] ↔ [[dual-dimension-feature-disentanglement]]
