# Ingest Report: S-Mamba

**Source**: arXiv 2403.11144 (Neurocomputing 2024)
**Title**: "Is Mamba Effective for Time Series Forecasting?"

## Created

- [[s-mamba]] — WHY: 首个 Mamba-based 多变量时序预测 baseline，双向 Mamba 编码变量间相关性 + FFN 编码时间依赖，13 数据集 × 9 SOTA 对比
- [[source-s-mamba]] — WHY: 论文核心论证记录：Mamba 在 VC 编码上优于 Transformer，FFN 在 TD 编码上保持统治

## Modified

- None (new ingest, no existing pages to update)

## Cross-links Established

- [[s-mamba]] ↔ [[mamba]] — Mamba SSM 是 S-Mamba 的基础构建块
- [[s-mamba]] ↔ [[mila]] — Mamba 启发架构的兄弟工作
- [[s-mamba]] ↔ [[channel-independence]] — 默认 CI 策略 + Mamba 层捕获跨变量相关性
- [[s-mamba]] ↔ [[patchtst]] — 同为 CI backbone 模型，S-Mamba 关注 VC 编码
- [[s-mamba]] ↔ [[itransformer]] — 主要对比基线，FFN-on-time TD 编码设计
- [[s-mamba]] ↔ [[lstf]] — 长序列时序预测评测范式
