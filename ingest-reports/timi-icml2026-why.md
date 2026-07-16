# Ingest 报告：TiMi (ICML 2026)

## 创建
- wiki/source-timi.md — WHY：论文来源摘要页，记录 TiMi 框架的核心论点、贡献与实验
- wiki/timi.md — WHY：TiMi 实体页，描述架构、Non-Fusion Guidance 范式定位及与其他多模态方法的差异
- wiki/mmoe.md — WHY：MMoE 技术页，TMoE+SMoE 双路门控的即插即用模块，跨模态 MoE 路由的创新应用
- wiki/non-fusion-guidance.md — WHY：Non-Fusion Guidance 概念页，第三种多模态预测范式，区别于 Early/Late Fusion

## 修改
- wiki/multimodal-time-series-forecasting.md — WHY：新增 TiMi 节 + 更新对比表（加入 TiMi 列），source_count 11→12
- wiki/mixture-of-experts.md — WHY：新增"在多模态时序预测中的 Non-Fusion Guidance"节，source_count 6→7
- wiki/event-driven-reasoning.md — WHY：对比表中加入 TiMi，新增 TiMi vs VoT 区分段落，source_count 1→2
- wiki/index.md — WHY：在 Sources/Entities/Concepts/Techniques 四类各加入新页面条目
- wiki/log.md — WHY：记录 ingest 操作

## 新建交叉链接
- [[timi]] ↔ [[mmoe]] ↔ [[non-fusion-guidance]]
- [[timi]] ↔ [[multimodal-time-series-forecasting]]
- [[timi]] ↔ [[mixture-of-experts]]
- [[timi]] ↔ [[event-driven-reasoning]]
- [[timi]] ↔ [[vot]]（Non-Fusion vs Late Fusion）
- [[timi]] ↔ [[tats]]（Non-Fusion vs 即插即用拼接）
- [[mmoe]] ↔ [[mixture-of-experts]]
- [[non-fusion-guidance]] ↔ [[event-driven-reasoning]]

## Lint 修复 (2026-07-25)
- [[source-timi]] confidence: high→medium — source_count=1 不满足 high 要求
- [[timi]] confidence: high→medium — source_count=1 不满足 high 要求
- [[mixture-of-experts]] 相关技术列表缺少 [[mmoe]]/[[timi]] 回链 → 补充
- [[event-driven-reasoning]] Related Pages 缺少 [[timi]]/[[non-fusion-guidance]]/[[mmoe]] → 补充
- 幻觉检查：PDF 原文与 wiki 所有事实性论断（作者/方法名/指标/数据集/LLM/公式）一致，无捏造
