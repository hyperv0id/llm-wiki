# Ingest 报告：MoBA 优化推导（derivation round）

源文件：`raw/moba-mixture-of-block-attention.md`（arXiv:2502.13189v1，2025-02-18）。本轮性质：对照原文核验既有内容并补数据流与推导，非重新 ingest。arXiv abs 页未见 v1 之后的更新版本，无版本漂移。

## 创建
- ingest-reports/2026-10-27-moba-derivation.md — 本报告

## 修改
- wiki/moba-attention.md — WHY: 新增「数据流与维度变换」（Stage 1 分块 mean-pool → Stage 2 块级 top-k 门控 → Stage 3 当前块 causal/历史块 non-causal 两段 attention → Stage 4 online softmax 合并，含逐头门控与「掩码先于 softmax」口径）与「门控归一化与 full attention 的关系」（淘汰块概率严格为 0 的成立条件、$Z_{\mathrm{MoBA}}\leq Z_{\mathrm{full}}$ 差异、退化确切口径 $k\geq\lceil\mathrm{pos}(\bm{q})/B\rceil$）；删去论文未出现的「无需自定义 CUDA kernel」（`raw/` 全文 grep `CUDA|kernel|custom` 无匹配）
- wiki/source-moba-mixture-of-block-attention.md — WHY: 作者行修正（原文列 Mingxing Zhang 为 Tsinghua University、Jiezhong Qiu 为 Zhejiang Lab/Zhejiang University，故「全部署名 Moonshot AI」不成立，改为 23+2）；贡献 1 同步删 CUDA kernel 表述并补「当前块 causal 与历史块 non-causal 分开计算」；last_updated
- wiki/log.md — maintenance 条目一条

## 核验结论（既有论断逐项对上原文）
- §2 机制：式 (3)–(6) 分块与门控、未来块 $s_i=-\infty$、当前块强制选中 + 块内 causal mask（shared expert 类比）、fine-grained 分块、与 full attention 的参数等价与动态切换（§2.2）——页面「机制」节全部一致。
- SWA 与 attention sink 作为特例的表述与原文 §2.2 段落逐句对应（最近块门控 / 初始块+最近块门控），无需改动。
- §2.3 五步实现与 Algorithm 1 一致；两段分开计算（Line 13 causal=True / Line 14 causal=False）与页面表述一致。
- §3.1 缩放定律：5 模型 568M–2.1B、块 512、top-3、稀疏度至多 81.25%（$1-\frac{512\times3}{8192}$）、LM loss 差 1e-3 内、trailing 拟合 $1.546\times C^{-0.108}$ vs $1.464\times C^{-0.097}$、消融 8/16/32/64/128 块选 2/4/8/16/32 固定 75%、最粗差 1e-2——数字全部一致。
- §3.2 混合训练：1.5B、30B token、32K、块 2048、top-3、90/10 两阶段、切换无 loss spike、SFT loss masking 稀疏梯度为作者推测（页面已标「原因未直接验证」）——一致。
- §3.3 下游：Llama 3.1 8B → 1M（256K 阶段起点位置插值）、100B token、块 4096、top-12、稀疏度至多 95.31%、末 3 层 full、评测仅 prefill、RULER 128K 62.5% 稀疏度 0.7818 vs 0.7849——一致。
- §3.4 效率：attention 层前向、1M prefill 至多 6.5×、10M 时 16×（Figure 2b 口径为固定 64 块、top-k=3、按比例增大块大小维持 95.31%，配 query head 级张量并行广播 K/V）——一致。
- 附录 A.1：按位置分段消除数据偏短偏差、trailing token 贡献差距大部分——一致。
- 与 [[native-sparse-attention]] 的对照段：三支路 + sigmoid 门控（式 3–5）、重映射 KV 替换原始 KV 且无「选满退化」开关（§3.2）、块重要性分取自压缩支路（式 8）、27B/3B 规模与 prefill/decode/backward 三档加速、稀疏基线 H2O/InfLLM/Quest/Exact-Top + Full Attention（§4.2）——与 NSA 页及其源摘要口径一致，无需矛盾处理。
- source 页 frontmatter `confidence: medium`（单源）符合规则，维持。

## 数值备注
- 论文内部口径不一（v1 即如此，非转换缺陷）：
  1. §3.1 正文先按 $1-\frac{512\times3}{8192}$ 算出稀疏度 81.25%，同段末尾又写 "sparsity up to 75%"。页面取公式值 81.25%。
  2. Figure 3 图题写 trailing LM loss 取 last 1K tokens，图内表格写 last 2K；页面取表格口径（$1.546\times C^{-0.108}$ / $1.464\times C^{-0.097}$），矛盾已在页面注明。
- 「15 页」经 arXiv v1 PDF（pdfinfo）复核为 15 页，维持。
- 式 (1)(2) 未写 $1/\sqrt{d_k}$ 温度项；新增数据流按原文口径省略，与页面既有 $s_i$ 公式一致。
- Algorithm 1 的 $\mathbf{S}\in\mathbb{R}^{N\times h\times n}$ 表明门控逐头独立，数据流节已按此标注。
