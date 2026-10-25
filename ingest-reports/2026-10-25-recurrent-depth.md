# Ingest 报告：2502.05171-recurrent-depth + tosea-looped-transformer

## 创建
- wiki/source-2502-05171-recurrent-depth.md — WHY：主源（arXiv:2502.05171v2）的摘要页，覆盖问题（串行算力必须落成 token）、机制（prelude/core/coda + 随机循环 + 截断反传）、证据（Table 1/2/4）与自述局限。
- wiki/source-tosea-looped-transformer.md — WHY：用户指定的背景源；单独成页以便把「二手报道 / 已确认未测量 / 推测」分级，并记录它与论文三处数字偏差，避免二手数字流入正文页。
- wiki/recurrent-depth.md — WHY：该源唯一的架构机制页；含 Stage 1–6 分阶段维度块（前向、训练路径、推理路径），避免另建张量表。
- wiki/latent-reasoning.md — WHY：论文的核心概念主张（隐空间推理 vs verbalized CoT）与可监督性代价需要独立落点，跨源引用论文与博客。

## 修改
- wiki/mixture-of-experts.md — WHY：补「与循环深度的互补：参数重 vs 计算重」小节，引论文 §9 的 "We posit" 原文（source_count 14→15）。
- wiki/index.md — WHY：按 type 五段插入 4 条新条目（Sources ×2、Concepts ×1、Techniques ×1），段内字母序。

## 新建交叉链接
- [[recurrent-depth]] ↔ [[latent-reasoning]]
- [[recurrent-depth]] ↔ [[mixture-of-experts]]（compute-heavy / parameter-heavy 镜像，论文 §9）
- [[latent-reasoning]] ↔ [[source-tosea-looped-transformer]]（可监督性报道链）
- [[recurrent-depth]] ↔ [[source-2502-05171-recurrent-depth]]、[[source-tosea-looped-transformer]]（双源引用：机制与谱系表分别挂到主源与二手源）

## 数值备注
- 学习率两处冲突：§4.1 写 η=5e-4，§4.3 写最终 run 峰值 4e-5，同页并列，未择一。
- 参数量 3.5B（正文/Table 1）与 Fig 1 横轴 3.6B 并存，指同一 checkpoint（Huginn-0125）。
- "等效 50B" 是 FLOP 口径（§5.1「FLOP budgets equivalent to a standard 50B parameter fixed-depth transformer」），摘要写作「computation load equivalent to 50 billion parameters」；Fig 1 的 Materialized Parameters 轴是展开层数折算（r=64 时 103B），不是精度等效。博客把两者混为「roughly 50B materialized parameters」，已在博客页并列说明。
- 博客与论文的三处数字偏差：r=1 的 SciQ 47.19 vs 47.10；r=1 的 GSM8K CoT 0.00/0.15 vs 0.00/0.00（0.00/0.15 属 180B 早期 checkpoint 行）；50B 口径见上。
- Table 2 的 flexible/strict 列序不确定：caption 说 "flexible and strict"，但 Ours w/ sys prompt 行 24.87/38.13 与 strict 通常更低的规律相反，本 wiki 引用时按 caption 顺序并保留该风险，未择一重排。
- 附录 A.2 写 "212–262 exaFLOP/s"，按 4096 GPU × 52–64 TFLOP/s 推得应为 petaFLOP/s，属单位笔误，未在正文引用该单位。
- 47000 steps × 16M token ≈ 752–788B，与「800B/795B」不完全吻合，论文未解释。
- pdftotext 缺陷：σ=1/2 被拆成 "σ = 12"，初始化方差 2/(5h)、1/(5hl)、2/5 被拉平为 "5h 2"、"5hl 1"、"25"，按数学语义还原；Fig 10/13 类别与数值配对有 1 格偏移风险，正文只引 MMLU 前 50 题的四类均值。
- 训练成本口径：论文只给 21 个 ≤12h segment 与 47000 步，未给总 GPU·小时或总 wall-clock，wiki 未补。
