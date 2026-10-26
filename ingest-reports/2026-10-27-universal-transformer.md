# Ingest 报告：universal-transformers-1807-03819

## 创建
- wiki/source-dehghani-2019-universal-transformers.md — WHY：每个 `raw/` 源一份 source-summary；问题/方法/证据/局限四段，单源 confidence medium
- wiki/universal-transformer.md — WHY：论文核心机制落成 technique 页；fixed-depth Transformer 对照表、五阶段数据流（固定步 Stages 1–4 + ACT 停机 Stage 5）、六组实验逐表数字、长度泛化论证与论文自述边界

## 修改
- wiki/recurrent-depth.md — WHY：谱系段「Dehghani et al. 用循环深度追求图灵完备」由二手口径改挂一手源，并补一手核实的三点差别（全层共享 vs 块级共享、ACT 逐位置停机 vs 无停机、每步重加位置-时间编码 vs 每轮重注入输入嵌入）；时间线句标注 UT 条已有一手源页面；博客引用保留在时间线一句作辅助。source_count 2→3，last_updated 2026-10-27
- wiki/index.md — WHY：Sources 段插入 source-dehghani-2019-universal-transformers（source-deepstate 与 source-demystify-mamba-linear-attention-2024 之间），Techniques 段插入 universal-transformer（unist 与 urbangpt 之间），段内字母序

## 新建交叉链接
- [[universal-transformer]] ↔ [[recurrent-depth]]（谱系互指：一手源页与后续循环深度路线）
- [[universal-transformer]] → [[transformer-architecture]]（fixed-depth 基线结构对照）
- [[universal-transformer]] → [[sinusoidal-positional-encoding]]、[[position-extrapolation]]（坐标编码机制与长度外推问题）
- [[universal-transformer]] → [[source-attention-is-all-you-need-1706-03762]]（基模型一手源页）
- [[ouro-looped-lm]] 类页尚未创建，本次不预置链接（用户指定将来互链）

## 数值备注
- 任务提示口径差（页面按论文执行）：bAbI 的对照是 QRNet / Sparse DNC / GA+MAGE / MemN2N / Transformer（Table 1），论文未在 bAbI 上对比 LSTM 或 Neural Programmer-Interpreter（LSTM 出现在 Tables 2–6）；论文没有 MLM 实验，语言建模 PPL 来自 LAMBADA（Table 3）。
- ACT 公式：论文正文不写数学式，只在附录 C 给 Listing 1–2 伪代码；页内 Stage 5 公式为该伪代码的逐行转写并已标注。Graves (2016) 原始 ACT 的 ponder-cost 损失与平均状态更新式不在本论文中（正文与实现均无损失项，「ponder time」只作为测量量出现），页面只写本源可查证内容、不转引 Graves 原式。
- 式 (1) 的 $\sqrt{d}$：$d$ 按式内定义是 Q 的列数（多头下即每头 $d/k$），与式 (3) 的投影维符号复用；页面按原式照录加注，未改写。
- 附录 E 双口径：best seed（与 Table 1 动态停机行一致：0.21/0.29/4.55/7.78）与 10 seed 平均（$0.73\pm0.89$ / $1.12\pm1.62$ / $6.34\pm3.32$ / $11.21\pm6.65$），页面两套都给，未替论文挑一套。
- WMT 的 +0.9 / +0.5 BLEU 归属：§3.6 明确是全连接 transition 且**不用 ACT** 的 UT base；摘要的「0.9 BLEU improvement」同口径，页面已写明。
- PDF：arXiv v3（2019-03-05），ICLR 2019 会议版，23 页，`pdftotext -layout` 转换；正文各表无缺列；Fig 3、5–7 为注意力热图，图内数值不可读（正文引用的 ponder 时间数字来自 §3.1 文字，不受影响）。

## 对 recurrent-depth 谱系段的更正
原句「Dehghani et al. 用循环深度追求图灵完备」出自 Huginn 相关工作的二手口径（引 [^src-2502-05171-recurrent-depth]），时间线部分为博客口径。本次通读一手源核实：UT 确以循环深度追求计算通用（§4 把 Neural GPU 归约为特例、附录 B 论证），但机制细节与 Huginn 不同——UT 全层共享（自注意力与 transition 都绑，脚注 5）、ACT 逐位置停机、每步重加正弦位置-时间编码；Huginn 为块级（4 层块）共享、无停机、每轮拼接重注入输入嵌入。谱系段已改挂 [^src-dehghani-2019-universal-transformers] 并把核实句指向 [[universal-transformer]]；博客引用 [^src-tosea-looped-transformer] 保留在时间线一句作辅助。
