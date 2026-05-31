# Ingest 报告：TimeCAP

## 创建
- `wiki/source-timecap.md` — 源文件摘要：TimeCAP 双 LLM agent + 多模态编码器的时间序列事件预测框架
- `wiki/timecap.md` — 实体页面：TimeCAP 完整架构、性能、与相关方法的对比

## 修改
- `wiki/index.md` — 在 Sources、Entities 分类中添加新条目
- `wiki/log.md` — 记录本次 ingest 操作

## 新建交叉链接
- [[timecap]] ↔ [[timesfm]] — TimesFM 作为相关 TS 基础模型
- [[timecap]] ↔ [[chronos]] — Chronos 作为相关 TS 基础模型
- [[timecap]] ↔ [[event-driven-reasoning]] — 两者均利用 LLM 理解时序上下文，互补关系
- [[timecap]] ↔ [[autoformer]] — Autoformer 为 TimeCAP 的 baseline
- [[timecap]] ↔ [[patchtst]] — PatchTST 为 TimeCAP 的 baseline 及 in-context sampler 对照
- [[timecap]] ↔ [[itransformer]] — iTransformer 为 TimeCAP 的 baseline
