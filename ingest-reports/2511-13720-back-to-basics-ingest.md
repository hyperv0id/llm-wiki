# Ingest 报告：Back to Basics: Let Denoising Generative Models Denoise

**Source**: arXiv 2511.13720 (Li & He, MIT, 2025)
**Ingest Date**: 2026-05-13

## 创建

- `wiki/sources/source-back-to-basics-let-denoising-generative-models-denoise.md` — WHY：完整 source-summary 页面，覆盖核心论点（流形假设、9 种组合、JiT 设计、实验对比）
- `wiki/entities/jit.md` — WHY：JiT (Just image Transformers) 是论文提出的像素空间扩散模型，作为独立实体页面记录其设计、变体、实验结果和意义
- `wiki/concepts/x-prediction.md` — WHY：x-prediction 是论文的核心概念，与 ε-/v-prediction 形成对比，需要独立概念页面系统梳理三种预测目标的历史、理论差异和实践建议

## 修改

- `wiki/diffusion-model.md` — WHY：在"三种等价训练目标"后添加 x-prediction 与 ε-/v-prediction 在流形假设下的根本差异说明，在关键实现中添加 JiT，在相关概念中添加 [[x-prediction]] 和 [[jit]]。source_count: 14→15
- `wiki/edm-design-space.md` — WHY：新增 "EDM 预处理器与 x-prediction 的矛盾" 小节，说明 EDM pre-conditioner 因 cskip ≠ 0 偏离 x-prediction 导致高维失效。source_count: 2→3
- `wiki/index.md` — WHY：添加三个新页面的索引条目
- `wiki/log.md` — WHY：追加本次 ingest 记录

## 新建交叉链接

- [[x-prediction]] ↔ [[diffusion-model]] — x-prediction 是扩散模型的一种参数化方式
- [[x-prediction]] ↔ [[edm-design-space]] — EDM pre-conditioner 与 x-prediction 存在内在矛盾
- [[x-prediction]] ↔ [[jit]] — JiT 是 x-prediction 的系统实践
- [[x-prediction]] ↔ [[elf]] — ELF 也在语言领域中用 x-prediction 实现共享权重
- [[x-prediction]] ↔ [[flow-matching]] — v-prediction 的数学框架与 x-prediction 的关系
- [[jit]] ↔ [[diffusion-model]] — JiT 是一种像素空间扩散模型
- [[jit]] ↔ [[elf]] — 同实验室（MIT, Kaiming He）的两个工作，都使用 x-prediction
