# Ingest 报告：xCPD — Routing Channel-Patch Dependencies

## 创建
- **wiki/source-xcpd.md** — WHY：新源文件 xCPD (Li et al., ICLR 2026) 的源文件摘要；核心贡献是图频谱分解驱动的 patch 级通道依赖路由
- **wiki/xcpd.md** — WHY：xCPD 是 ICLR 2026 的新实体方法，代表从 channel 级建模→channel-patch 频谱域建模的重要范式转变，与 CI/CD/CP 策略、MoE、patch tokenization 等 wiki 现有概念高度关联

## 修改
- 无（纯新增）

## 新建交叉链接
- [[xcpd]] ↔ [[channel-independence]] — xCPD 明确目标为弥补 CI/CD 策略在 patch 级频率解耦上的不足
- [[xcpd]] ↔ [[cross-dimension-dependency]] — CD 策略 vs 频谱域 CPD 的对比
- [[xcpd]] ↔ [[crossformer]] — xCPD 对比分析表中列举 Crossformer（同为 patch 粒度但时间域 vs 频谱域）
- [[xcpd]] ↔ [[patchtst]] — xCPD 使用的四个 CI backbone 之一，沿用其 CI + patching 设计但引入跨通道依赖
- [[xcpd]] ↔ [[itransformer]] — xCPD 在附加 backbone 实验中使用 iTransformer，证明模型无关性
- [[xcpd]] ↔ [[patch-based-tokenization]] — xCPD 将 patch tokenization 的概念推广至 channel-patch 图节点
- [[xcpd]] ↔ [[mixture-of-experts]] — xCPD 的 DyMoE 路由是对标准 Top-K MoE 的改进（动态选择 1-3 expert）

## 未修改的现有页面（需后续更新）
- [[channel-independence]] — 需添加 xCPD 作为 CP 策略的 plugin 案例
- [[cross-dimension-dependency]] — 需添加 xCPD 作为频谱域 CD 建模的新范式
- [[mixture-of-experts]] — 需添加 DyMoE 作为 MoE 变体的参考
