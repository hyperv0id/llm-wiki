# Ingest 报告：gated-delta-networks-2412-06464

## 创建
- wiki/source-yang-2024-gated-delta.md — WHY：每个 raw/ 源文件一个 source-summary；本页承载论文口径（出处、版本、核心论点、证据、边界）供其它页面引用。
- wiki/gated-delta-rule.md — WHY：论文的机制贡献（gated delta rule 与 chunkwise 并行算法）需要一个 technique 页落公式与逐表数字；按 AGENTS.md 数据流规范用分阶段 aligned 块。

## 修改
- wiki/trace-as-state-reproduction.md — WHY：§2 的「GDN + GA」架构归因原为二手叙述，补一手出处引用（source_count 4→5）。
- wiki/mamba.md — WHY：Mamba2 门控更新是 gated delta rule 去掉擦除项的退化形式，相关页面加链接。
- wiki/selective-ssm.md — WHY：门控遗忘谱系（选择机制 → 现代 forget gate → gated delta rule）补后续页链接。
- wiki/linear-attention-rnn-form.md — WHY：本页 Hebbian 累积是 delta rule 的特例，「在本 wiki 中的位置」补谱系句。
- wiki/index.md — WHY：Sources / Techniques 段各加一条。
- wiki/log.md — WHY：追加一条 ingest 记录。

## 新建交叉链接
- [[gated-delta-rule]] ↔ [[mamba]]（Mamba2 门控更新为退化形式）
- [[gated-delta-rule]] ↔ [[selective-ssm]]（数据依赖遗忘谱系）
- [[gated-delta-rule]] ↔ [[linear-attention-rnn-form]]（Hebbian 累积为 delta rule 特例）
- [[gated-delta-rule]] ↔ [[trace-as-state-reproduction]]（Qwen 3.7 Max 架构归因 GDN 的一手出处）

## 数值备注
- 任务简报与论文不符处（已核对 arXiv v1/v2/v3 三版全文，均不存在）：无「2.7B/380B 训练」实验（全部实验在 0.4B–1.3B、100B token 内）；无「Gated Delta Slot」一词；无「GATED DELTA NIAH」或「每块 4 token」的表述（合成召回仅 RULER 的 S-NIAH-1/2/3，Table 2）；无「GTA 与 VDeltaD」的差距分析（§1 的差距分析以 Mamba2 门控更新 vs DeltaNet delta rule 表述）。页面一律以论文实际内容为准，未补写。
- 正文与表格不一致一处：§4 正文称 Table 3 覆盖「models with 400M and 1.3B parameters」，表中仅 1.3B 行；400M 主对比数值在 v3 正文/附录均未见表载，gated-delta-rule.md 边界节已并列照录。
- Fig 1–3 为示意图/曲线/柱状图，图内数值不可读，相关结论（长度外推、吞吐）取自正文句子，未引具体数字。
- 脚注 4：α 沿用 Mamba2 参数化但论文「为简洁省略」，页面如实标注不可复核。
- pdftotext -layout 全文可读；chunkwise 公式（Eq. 8–9、§3.3）的上下标经 300–400 dpi 页面图渲染逐项核对后转录。
