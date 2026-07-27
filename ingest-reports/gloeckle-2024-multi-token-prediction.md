# Ingest 报告：Gloeckle et al. Multi-token Prediction (ICML 2024)

## 创建
- wiki/source-gloeckle-2024-multi-token-prediction.md — WHY：raw PDF 对应 source-summary；覆盖问题、架构/显存技巧、代码与 NL 实验、机制推测与局限
- wiki/multi-token-prediction.md — WHY：核心训练目标值得独立 technique 页，便于与 next-token、STF、scheduled sampling 对照
- wiki/self-speculative-decoding.md — WHY：论文第二大可落地收益（推理 3×/6×），且依赖 multi-token 预训练 head 质量，单独成页避免与训练目标混写

## 修改
- wiki/sparse-teacher-forcing.md — WHY：同属缓解 teacher-forcing ↔ 自回归分布错配；补 LLM multi-token 对照段与交叉链接；source_count 1→2
- wiki/dcrnn.md — WHY：Related 中 scheduled sampling 条目指向 multi-token 作为离散 LM 上对 scheduled sampling 的替代叙事；last_updated 刷新
- wiki/index.md — WHY：Sources / Techniques 登记新页；last_updated → 2026-07-27
- wiki/log.md — WHY：按 ingest 工作流追加时间线

## 未新建（有意）
- scheduled-sampling 独立页 — index/dcrnn 已有 wikilink 但本库可能仍为悬空概念；本次不凭单篇 LLM 论文补全时序 scheduled sampling 专页
- next-token-prediction 专页 — 全库大量隐含前提，单源不足以立 concept 而不膨胀

## 新建交叉链接
- [[multi-token-prediction]] ↔ [[source-gloeckle-2024-multi-token-prediction]]
- [[multi-token-prediction]] ↔ [[self-speculative-decoding]]
- [[multi-token-prediction]] ↔ [[sparse-teacher-forcing]]
- [[self-speculative-decoding]] ↔ [[source-gloeckle-2024-multi-token-prediction]]
- [[dcrnn]] → [[multi-token-prediction]]（Related 旁注）
- [[sparse-teacher-forcing]] → [[dynamix]] / [[multi-token-prediction]] / [[self-speculative-decoding]]

## raw
- raw/gloeckle-2024-multi-token-prediction.pdf ← 拷贝自 Zotero `72B9W2DI/Gloeckle 等 - Better & Faster Large Language Models via Multi-token Prediction.pdf`
