# Ingest 报告：STBP (ICLR 2026)

## 创建
- `wiki/source-stbp.md` — WHY：每个 raw/ 源文件须有一个 source-summary 页面，总结 STBP 核心贡献、架构、实验结果和局限性
- `wiki/stbp.md` — WHY：STBP 是新提出的实体/框架，引入 fixed-backbone + expandable-pattern-bank 范式，需独立实体页面记录架构、工作流、性能、与 foundation model 的关系
- `wiki/contextual-pattern-bank.md` — WHY：Contextual Pattern Bank 是 STBP 的核心技术创新（三组件参数记忆 + 纯增量扩展 + prompt-based guidance），有足够技术深度独立成页
- `wiki/continual-spatio-temporal-forecasting.md` — WHY：CSTF 是 STBP 所属的问题范式，wiki 中尚未有此概念页面。该页面涵盖 CSTF 问题定义、关键挑战、方法谱系（TrafficStream→STKEC→PECPM→STRAP→EAC→STBP）、数据集和与 foundation model 的关系

## 修改
- `wiki/traffic-forecasting.md` — WHY：新增 "Continual Spatio-Temporal Learning" 章节（含 TrafficStream/STKEC/PECPM/STRAP/EAC/STBP 方法列表），source_count 从 34→35，添加 [^src-stbp] 引用
- `wiki/spatio-temporal-foundation-model.md` — WHY：新增 "Continuous Learning / Continual Adaptation" 子章节（含 STBP 条目），source_count 从 16→17，添加 [[stbp]][[continual-spatio-temporal-forecasting]] 链接和 [^src-stbp] 引用
- `wiki/index.md` — WHY：添加 [[source-stbp]]（Sources）、[[stbp]]（Entities）、[[continual-spatio-temporal-forecasting]]（Concepts）、[[contextual-pattern-bank]]（Techniques）四个条目
- `wiki/log.md` — WHY：记录本次 ingest 操作，说明创建的页面、更新的页面、核心贡献和实验结果

## 新建交叉链接
- [[stbp]] ↔ [[contextual-pattern-bank]] — 实体与核心技术的双向链接
- [[stbp]] ↔ [[continual-spatio-temporal-forecasting]] — 实体与问题范式的双向链接
- [[continual-spatio-temporal-forecasting]] ↔ [[traffic-forecasting]] — CSTF 概念页面链接到领域总页
- [[stbp]] ↔ [[spatio-temporal-foundation-model]] — STBP 自述为 ST 基础模型的中间步骤
- [[stbp]] ↔ [[eac]] — STBP 在实体页面引用 EAC 为最强 baseline
- [[contextual-pattern-bank]] ↔ [[eac]] — 对比两种 prompt 机制的差异
- [[stbp]] ↔ [[pecpm]] — 对比两种 pattern bank 方法的差异
