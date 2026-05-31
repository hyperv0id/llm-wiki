# Ingest Report: LSTNet (Lai et al., SIGIR 2018)

## Created
- **wiki/source-lstnet.md** — 源文件摘要：首个跨维度 MTS 深度学习框架，CNN+RNN+Skip-RNN+AR 四组件架构
- **wiki/lstnet.md** — 实体页面：LSTNet 完整架构解析，跨维度依赖建模路线的起点，与 CI/CD 路线的对比

## Modified
- **wiki/index.md** — 添加 source-lstnet 和 lstnet 条目
- **wiki/log.md** — 记录 ingest 操作

## New Cross-Links
- [[lstnet]] ↔ [[cross-dimension-dependency]] — LSTNet 是 CNN 跨维度路线的开山之作
- [[lstnet]] ↔ [[mtgnn]] — MTGNN 延续跨维度路线，转向 GNN
- [[lstnet]] ↔ [[crossformer]] — Crossformer 延续跨维度路线，转向 Transformer
- [[lstnet]] ↔ [[channel-independence]] — CD vs CI 的早期实证对比

## Key Judgment
LSTNet 是 MTS 深度学习领域的奠基性工作。即使技术被后续模型超越，其设计原则——并行线性+非线性、显式周期建模、尺度敏感性——至今仍是核心教训。confidence: high。
