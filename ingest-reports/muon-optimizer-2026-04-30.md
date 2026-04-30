# Ingest Report: Muon Optimizer

## Source Files
- `raw/muon-optimizer.md` — 完整博客文章 (source: https://kellerjordan.github.io/posts/muon/)
- `raw/kellerjordan-muon-blog.md` — URL 记录

## Created Pages

| Page | Type | Reason |
|------|------|--------|
| [[source-muon-optimizer]] | source-summary | 源文件摘要 |
| [[source-kellerjordan-muon-blog]] | source-summary | URL 记录 |
| [[muon-optimizer]] | entity | 优化器实体 |
| [[newton-schulz-iteration]] | technique | 核心算法技术 |
| [[gradient-orthogonalization]] | concept | 优化概念 |

## Updated Pages

- [[index]] — 添加新页面到 Sources、Entities、Techniques、Concepts
- [[log]] — 记录 ingest 活动
- [[muon-optimizer]] — 添加第二源引用
- [[source-muon-optimizer]] — 添加第二源引用，source_count: 2

## New Cross-Links

- [[muon-optimizer]] ↔ [[newton-schulz-iteration]]
- [[muon-optimizer]] ↔ [[gradient-orthogonalization]]

## Key Insights

1. **Muon 优化器**使用 Newton-Schulz 迭代对 SGD-动量更新进行正交化
2. **关键设计**：在正交化之前应用动量（与之前方法相反）
3. **实验验证**：CIFAR-10、NanoGPT 速度纪录
4. **FLOP 开销**：< 1% 典型 LM 训练场景
5. **创新提出**：竞争性任务框架解决基线调优不足问题