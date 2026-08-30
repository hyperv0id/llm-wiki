# Ingest 报告：TriAttention (arXiv 2604.04921, ICML 2026)

源文件：`raw/triattention.pdf`（arXiv:2604.04921v1, 2026-04-06，19 页，pdftotext 全文阅读）；`raw/triattention-openreview.md`（OpenReview forum 0tgzJK50Jz，ICML 2026 Submission 10803，用户提供的页面存档，含 Decision: Accept (regular) 与 4 份评审/rebuttal）。

## 创建

- `wiki/source-triattention.md` — WHY：raw PDF 的 source-summary，每个 raw 文件对应一个；记录问题、Q/K 集中观察、三角级数机制、主结果与论文自述局限。
- `wiki/source-triattention-openreview.md` — WHY：venue/decision 与评审动态（评分、rebuttal 补充实验、仅见于 rebuttal 的论断）是 PDF 之外的信息源，按「每个 raw 文件一个 source-summary」独立成页。
- `wiki/triattention.md` — WHY：方法实体页，承载打分函数（S_trig/S_norm/未来偏移/GQA 归一聚合/窗口剪枝）、完整实验数字、评审动态与局限；特别记录 Table 1/3（AIME24 42.1）与附录消融（45.8）的内部数字不一致，两组数字分别归因。
- `wiki/qk-concentration.md` — WHY：论文的核心经验现象，独立于具体方法、可被后续 RoPE/注意力几何类工作引用；含 MRL 定义、跨域/跨架构测量与发现方式注记。
- `wiki/attention-distance-preference.md` — WHY：论文的核心理论结果（RoPE logit → 三角级数 → 距离偏好可预测），与经验现象、方法实现分立成页，各自一条知识链。
- `wiki/kv-cache-compression.md` — WHY：问题域页，此前 wiki 无任何 KV cache 页面；记录 TriAttention 口径的三类 post-RoPE 方法谱系与观察窗口限制（已注明为二手转述口径）。
- `wiki/attention-sink.md` — WHY：高频复现概念，wiki 首次出现；confidence: low 并加 callout 注明原始出处（StreamingLLM/VATP）未 ingest。

## 修改

- `wiki/rope.md` — WHY：「后续发展」补 pre-RoPE 空间几何条目（Q/K 集中与三角级数），「相关页面」补 3 条链接；新事实句带 [^src-triattention]，source_count 3→4。
- `wiki/context-window-extension.md` — WHY：新增「相关页面」链接 [[kv-cache-compression]]（长上下文两条互补路线），结构性链接、无新论断，source_count 不变。
- `wiki/index.md` — WHY：Sources +2、Entities +1、Concepts +4 条目。
- `wiki/log.md` — WHY：追加 ingest 条目。

## 新建交叉链接

- [[rope]] ↔ [[qk-concentration]] / [[attention-distance-preference]]（RoPE 的 pre-RoPE 几何）
- [[qk-concentration]] ↔ [[attention-distance-preference]]（现象 → 推论）
- [[triattention]] ↔ [[qk-concentration]] / [[attention-distance-preference]] / [[kv-cache-compression]]（方法 → 前提与问题域）
- [[attention-sink]] ↔ [[kv-cache-compression]] / [[attention-distance-preference]]
- [[context-window-extension]] → [[kv-cache-compression]]（长上下文两条路线）

## 备注

- 命名辨析：论文 related work 引用的 QUEST（Tang et al. 2024, query-aware sparsity）与 wiki 既有 [[quest-attention]]（Query-modulated Spherical Attention, ICLR 2026）是同名不同文；为避免混淆，本次全部新页面均未链接 quest 相关页面。
- 矛盾核对：grep 全 wiki 无既有 KV cache 压缩/Q/K 集中论断，无矛盾触发；[[rope]] 的「长期衰减」为 RoFormer 原文的内积上界论证，与 TriAttention 报告的「部分头在远距离成峰」分属不同主张（上界 vs 经验偏好），不构成矛盾，未做并置处理。
- 论文内部数字不一致（AIME24 同配置 42.1 vs 45.8）在 [[triattention]] 与 log 中如实记录，未择优采用。
