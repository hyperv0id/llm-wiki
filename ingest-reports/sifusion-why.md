# Ingest 报告：SIFusion — Multi-Granularity Arctic Sea Ice Forecasting

## 创建
- wiki/source-sifusion.md — WHY：论文源摘要，Arctic SIC 预测领域首个多粒度统一框架 (NeurIPS 2025)
- wiki/sifusion.md — WHY：模型实体页面，Swin V2 空间编码 + granularity variate attention 架构详解
- wiki/multi-granularity-sea-ice-forecasting.md — WHY：概念页面，跨粒度 SIC 联合建模新范式，与 multi-scale attention 区分
- wiki/granularity-variates.md — WHY：技术页面，iTransformer variate 思路跨粒度迁移的核心机制
- wiki/independent-spatial-tokenization.md — WHY：技术页面，解耦 U-Net channel-wise fusion 的空间编码策略
- wiki/sea-ice-concentration-forecasting.md — WHY：领域概览，SIC 预测从数值模型到深度学习的演进
- wiki/arctic-amplification.md — WHY：气候背景概念，北极放大效应作为海冰预测动机

## 修改
- wiki/subseasonal-to-seasonal-forecasting.md — WHY：SIFusion 覆盖 S2S 尺度且仅用 SIC 数据，填补 S2S 方法中海冰预测空白
- wiki/itransformer.md — WHY：granularity variates 直接受 iTransformer variate attention 启发，需要双向交叉引用
- wiki/multi-scale-attention.md — WHY：多粒度海冰预测与多尺度注意力同属多粒度建模但机制不同，需区分
- wiki/index.md — WHY：所有新页面加入索引的 Sources/Entities/Concepts/Techniques 类别
- wiki/log.md — WHY：按 Ingest 工作流记录操作

## 新建交叉链接
- [[sifusion]] ↔ [[multi-granularity-sea-ice-forecasting]] ↔ [[granularity-variates]] ↔ [[independent-spatial-tokenization]] ↔ [[sea-ice-concentration-forecasting]]
- [[sifusion]] → [[subseasonal-to-seasonal-forecasting]]（S2S 方法）
- [[granularity-variates]] → [[itransformer]]（variate attention 灵感来源）
- [[multi-granularity-sea-ice-forecasting]] → [[multi-scale-attention]]（多粒度机制对比）
- [[sea-ice-concentration-forecasting]] → [[arctic-amplification]]（气候背景）

## Raw
- raw/sifusion-multi-granularity-arctic-sea-ice-forecasting.pdf（新拷贝，未修改 raw/ 任何已有文件）

## Lint 修复（2026-07-21）

### 问题
- source-sifusion.md：7 处自引用 `[^src-sifusion]` + 脚注自引用 → 违反 source-summary 不自引用规则（qdf 先例）
- sifusion.md：缺少到 arctic-amplification 的交叉引用

### 修复
- source-sifusion.md：移除全部自引用 + 新增到 sifusion/itransformer/independent-spatial-tokenization/granularity-variates/multi-granularity-sea-ice-forecasting/sea-ice-concentration-forecasting/arctic-amplification 的 wikilink
- sifusion.md：新增 arctic-amplification 相关页面链接

### 幻觉检查（通过）
对照 pdftotext 验证：作者、NeurIPS 2025、Swin V2、三粒度、2×2 patch、32 spatial channels、NSIDC G02202 v4 参数、数据划分、指标、Table 1/2/3 数字全部一致——无捏造或错引
