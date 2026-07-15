# Ingest Report：GMF — Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion

## 创建
- `wiki/source-gmf.md` — 源摘要页（type: source-summary, 300-500 字），覆盖论文五大核心贡献 + 实验 + 局限
- `wiki/gmf.md` — 实体页（type: entity），详细记录 GMF 框架架构、理论保证、实验表现、方法论定位
- `wiki/circular-dependency-in-multimodal-fusion.md` — 概念页（type: concept），形式化"循环依赖"问题及其经验证据与解决方向
- `wiki/geometric-barrier-principle.md` — 技术页（type: technique），Theorem 4.5 及其 Corollary 4.6 的完整形式化与经验验证
- `wiki/transport-based-reliability-assessment.md` — 技术页（type: technique），传输代价作为可靠性信号的方法论框架

## 修改
- `wiki/schrodinger-bridge.md` — 新增"GMF：多模态融合的几何可靠性"应用案例章节 + `[^src-gmf]` 引用，source_count: 1→2
- `wiki/building-schrodinger-bridges.md` — 引言段列入 GMF 单步 RF 近似 DSB 的应用，source_count: 2→3
- `wiki/rectified-flow.md` — 新增"GMF：多模态融合可靠性评估"应用案例章节，source_count: 3→4
- `wiki/index.md` — 添加 5 个新页面条目至 Sources/Entities/Concepts/Techniques 各部分
- `wiki/log.md` — 追加 ingest 记录条目

## 新建交叉链接
- [[schrodinger-bridge]] ↔ [[gmf]]（SB 应用 → GMF 框架）
- [[rectified-flow]] ↔ [[gmf]]（RF 应用 → GMF）
- [[building-schrodinger-bridges]] ↔ [[gmf]]（DSB 近似 → GMF）
- [[circular-dependency-in-multimodal-fusion]] ↔ [[gmf]]（问题概念 ↔ 解决方案）
- [[geometric-barrier-principle]] ↔ [[gmf]]（核心理论 ↔ 框架）
- [[transport-based-reliability-assessment]] ↔ [[gmf]]（方法论 ↔ 框架）
- [[geometric-barrier-principle]] ↔ [[circular-dependency-in-multimodal-fusion]]（解决机制 ↔ 问题定义）
- [[transport-based-reliability-assessment]] ↔ [[rectified-flow]]（方法论 ↔ 底层工具）
- [[transport-based-reliability-assessment]] ↔ [[schrodinger-bridge]]（方法论 ↔ 理论基础）

## 源文件
`raw/gmf-geometry-based-schrodinger-bridges-multimodal-fusion.pdf`（603 KB，不可变）
