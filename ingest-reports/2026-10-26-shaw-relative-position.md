# Ingest 报告：Self-Attention with Relative Position Representations

源：arXiv:1803.02155v1（Peter Shaw / Jakob Uszkoreit / Ashish Vaswani，Google / Google Brain，2018-03-06，5 页）。PDF 下载后存 `<worktree>/.tmp/shaw2018.pdf`，`pdftotext -layout` 转出 302 行文本，通读范围：摘要、§1 Introduction、§2.1 Transformer、§2.2 Self-Attention、§3.1 Relation-aware Self-Attention、§3.2 Relative Position Representations、§3.3 Efficient Implementation、§4.1 Experimental Setup、§4.2 Machine Translation、§4.3 Model Variations、§5 Conclusions、Figure 1、Table 1–3、参考文献头尾核对。该文无附录（5 页即正文 + 参考文献）。按任务要求 `raw/` 未改动，PDF 只留在 `.tmp/`（已被 .gitignore 忽略），因此三个新页的脚注以 arXiv 号溯源，不指向 raw 路径。

清单信息与官方源一致：标题 "Self-Attention with Relative Position Representations"，三位作者与年份（2018）相符。v1 未见 venue 著录，页面只写 arXiv 编号，不补会议名。

## 创建

- `wiki/source-shaw-2018-relative-position.md` — WHY：仓库此前零覆盖（全库无 1803.02155 与作者名引用），源需要的摘要页缺位；按「问题 / 方法 / 证据 / 边界」四段写，数字逐条回 Table 1–3 核对。
- `wiki/relative-position-representation.md` — WHY：本论文最可迁移的部分是位置表示本身（$2k+1$ 个裁剪标签向量 + 内容项/位置项拆分），与翻译任务无关，单独成 technique 页；含 Stage 1–4 分阶段 `aligned` 维度块（投影 → 每头 gather 边矩阵并与 key 相加 → softmax → value 侧加边向量 / 拼接 / 投影），以及三条推导（为何只依赖 $j-i$、裁剪后每 query 只有 $2k+1$ 个相对位置、softmax 的内容项 × 位置项分解及其前提）。
- `wiki/relation-aware-self-attention.md` — WHY：论文把方法定位为「自注意力推广到标号有向图」的接口（§3.1 标题与摘要都这么写），这部分既不属于序列位置编码也不属于翻译任务；单列概念页承载两条注入路径、边的两种取值（通用图 $n^2$ 条边 / 序列 $2k+1$ 标签）、空间代价与两条路径的消融。

## 修改

- `wiki/rope.md` — WHY：页面把相对位置偏置写成「XLNet 式相对位置偏置」但无出处，而这条血脉的起点就是本源；补来源句、对比表下方的加法/旋转分工说明、相关页面链接与脚注，`source_count` 4→5。
- `wiki/alibi.md` — WHY：「与其他位置编码的关系」只列了 Sinusoidal / Rotary / T5 Bias，缺相对位置表示这一支，导致 ALiBi 的「固定斜率 + 只改 logit」失去对照；补一条并加脚注，`source_count` 4→5。
- `wiki/generalized-positional-encoding-framework.md` — WHY：GPE 的 $f(n)\,q^\top W(n)k+b(n)$ 只覆盖乘性变换与标量偏置，Shaw 的位置项（依赖 query 内容、不经过 $x_j$ 投影）落在两者之外；补一小节说明覆盖边界，标 `[INFERENCE]`，`source_count` 2→3。
- `wiki/scaling-factor-sqrt-dk.md` — WHY：该页只讨论内容项的点积方差膨胀，而式 (5) 里 $1/\sqrt{d_z}$ 是乘在「内容项 + 位置项」之和外面的，位置向量没有独立斜率参数；补一节并说明只缩放内容项的后果，标 `[INFERENCE]`，`source_count` 1→2。
- `wiki/index.md` — WHY：新增三个条目（Sources / Concepts / Techniques 各一，段内字母序、每页仅一条），并刷新 `last_updated`。
- `wiki/log.md` — WHY：一条 ingest 记录（问题 / 机制 / 证据 / 创建与更新 / 报告 / 数值备注）。

## 新建交叉链接

- [[relative-position-representation]] ↔ [[source-shaw-2018-relative-position]]
- [[relation-aware-self-attention]] ↔ [[relative-position-representation]]（通用图形式与序列特化）
- [[rope]] ↔ [[relative-position-representation]]（加法型 vs 旋转型）
- [[alibi]] ↔ [[relative-position-representation]]（学习向量 + 双路注入 vs 固定标量斜率、只改 logit）
- [[generalized-positional-encoding-framework]] ↔ [[relative-position-representation]]（$W(n)$/$b(n)$ 覆盖不到边向量位置项）
- [[scaling-factor-sqrt-dk]] ↔ [[relative-position-representation]]（共用 $1/\sqrt{d_z}$）
- [[transformer-architecture]]、[[sinusoidal-positional-encoding]]、[[context-window-extension]] — 按名链接，本 worktree 无此页（属兄弟分支），未新建同名页

## 数值备注

- **摘要与表格的口径差**：摘要写 "improvements of 1.3 BLEU and 0.3 BLEU"，对应 Table 1 里 big 模型的两个差值（EN-DE 27.9→29.2、EN-FR 41.2→41.5）；base 模型的差值是 0.3 与 0.5（§4.2 正文同时给出两组）。页面按模型分别引用，不把摘要数字当成两任务的平均或唯一口径。
- **Table 1 的基线是同库复现值**：§4.2 说明基线在 tensor2tensor 里重跑以隔离相对位置的影响；页面只引 Table 1 的数字，不引入其他来源的英语-德语基线值。
- **「叠加绝对位置编码无进一步收益」缺表格**：§4.2 有此陈述但 Table 1 只有纯绝对、纯相对两行，没有组合配置的行；页面按「无对应表格」记录，不当成有表格支撑的结论。
- **裁剪与未见长度的关系是动机陈述**：§3.2 写 "Clipping the maximum distance also enables the model to generalize to sequence lengths not seen during training"，全文没有外推实验（$k$ 消融都在训练长度内）；页面注明为动机。
- **数值一致性**：Table 2 的 $k=0$ 行 12.5 与 Table 3 的 $a^V$、$a^K$ 双关行 12.5 相同，论文未解释；本页给出一条代数说明（$k=0$ 时位置项在行内为常数、softmax 平移不变）并标 `[INFERENCE]`，不改动原文数字。
- **抽取缺陷**：`pdftotext -layout` 把双栏正文与表格交错，Table 2/3 的数值列与页边注混排（逐格复核后取用）；边界标签表 $w^K=(w^K_{-k},\dots,w^K_k)$ 在文本层分裂为两行，按 §3.2 正文与 Figure 1 复原。
- **原文未给出**：训练用的最大序列长度、$b$ 与 $n$ 的具体取值（只给每 GPU 4096 token 与约 25,000 源/目标 token 的批），故 $n/(bh)$ 只保留公式形式，不给具体倍数。
