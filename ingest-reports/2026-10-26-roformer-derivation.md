# 维护报告：RoPE / RoFormer 推导补全与数值回打

## 源与版本

- 源：`raw/Su 等 - 2023 - RoFormer Enhanced Transformer with Rotary Position Embedding.pdf`，只读，未改动。
- 提取文本置于 `<worktree>/.tmp/roformer.txt`（`pdftotext -layout`，911 行），不进入 `raw/`。
- 版本：PDF 首页标注 `arXiv:2104.09864v5 [cs.CL] 8 Nov 2023`。arXiv abs 页的提交历史为 v1 2021-04-20、v2 2021-10-09、v3 2022-08-05、v4 2022-08-09、v5 2023-11-08，与 PDF 一致，故以 v5 为准。v1 摘要只给中文初步结果并写明英文基准待补，v5 补入 WMT、掩码语言建模、GLUE、Performer 四组实验，作者名单由五人增至六人（新增 Ahmed Murtadha）。
- 通读范围：§1 引言、§2（2.1–2.3，含式 (3)–(10) 的加法展开与各方案的替换位置）、§3（3.1 形式化式 (11)；3.2 二维解与高维推广式 (12)–(16)；3.3 两条性质与线性注意力式 (17)–(19)；3.4.1 二维推导式 (20)–(33)；3.4.2 高效实现式 (34)；3.4.3 长距离衰减式 (35)–(37) 与 Figure 2）、§4（4.1 WMT 与 Table 1；4.2 MLM 与 Figure 3 左；4.3 GLUE 与 Table 2；4.4 Performer 与 Figure 3 右；4.5 中文数据、Table 3–5、§4.5.5 局限）、§5 结论。

## 修改

- `wiki/rope.md` — WHY：本轮的主要落点。原文的重心就是式 (11)–(37) 这条推导链，而旧页只有结论式，衰减一句「由 $\theta_i$ 的指数衰减和 Abel 变换保证」无法核对。改动：
  - 新增「二维情形：解出旋转」，把式 (23)–(33) 的模条件 / 辐角条件分开写，交代二维 $\theta$ 的自由度如何在高维被频率分配固定。
  - 新增「高维」小节，补 $R^d_{\Theta,m}$ 的分块结构与两种频率编号（§3.2.2 的 1 基 $10000^{-2(i-1)/d}$ 与 §3.4.3 的 0 基 $10000^{-2i/d}$ 是同一条 schedule）。
  - 新增「数据流与维度变换」：Stage 1（投影支路 + 角度支路 + 按对 2×2 旋转汇合 + 逐元素实现）、Stage 2（打分与复数分解）、Stage 3（softmax / 值通路 / 多头拼接 / 输出投影）、Stage 2′（线性注意力写法）。
  - 重写「长期衰减」为可核对的链：式 (35) 复数分组 → 式 (36) Abel 变换 → 式 (37) 三角不等式 → 范数界 → $B(\Delta)$ 两段形态 → 三条成立前提 → 这条界的适用范围（不紧、非单调、不直接约束 softmax）。
  - 新增「与相对位置编码的关系」：各类方案注入点对照表、保范 / 无参数 / 线性注意力兼容三条性质的推导来源，并写清相对性来自 $R_m^\top R_n=R_{n-m}$ 而非 $p_m\to\tilde p_{m-n}$ 替换。
  - 修正三处旧表述：提出时间由「2023 年」改为 v1 2021-04 / v5 2023-11；旧页把「长期衰减」写成无条件普遍性质，改为带前提的上界陈述；旧页对比表把 RoPE 外推能力写作 ✅，改为「外推：性质表述与实测的分界」：论文的「序列长度灵活」是定义层面的性质，无插值时的实测失效由 ALiBi 一侧给出，两者在正文分述；冲突只落在这一条性质表述上，故不设 `## 争议`、frontmatter 保持 `status: active`。
  - 删除无源清单：旧页「在 LLM 中的统治地位」列了 LLaMA / Mistral / Qwen / DeepSeek / Gemma / Phi 等采用情况，仓库内没有支撑源；改为在「外推」节用 [[yarn|YaRN]]（Llama 2 4k→128k）[^src-yarn] 说明 RoPE 被后续沿用。source_count 4→6。
- `wiki/source-roformer.md` — WHY：frontmatter `source_count: 0` 且正文无脚注，与「每个事实句带引用」冲突。按实际引用改为 1 并补脚注定义；摘要重写为 493 个汉字（旧页约 700 余字且只列 GLUE 三项上升、隐去三项下降）；补版本行（五个版本的日期与 v1→v5 内容差异）；局限节按 §4.5.5 两条原意改写；删除无源的「影响」节（LLM 采用清单）。
- `wiki/roformer.md` — WHY：旧「关键性能」表只挑 GLUE 三项上升，砍掉 SST-2 / QNLI / MNLI 三项下降，属选择性引用；改为 Table 2 六项全列并注明论文原话「六项中的三项」。同时把 Table 1、Figure 3、Table 5 的实验设置（fairseq + 37k BPE + beam 4 + 5 checkpoint 平均；Performer Enwik8 12 层 768 维 12 头字符级；中文预训练六阶段）补齐，补 Table 4 的 67.4% / 63.4% 两个端点。source_count 1→3。
- `wiki/learnable-frequency-scaling.md` — WHY：旧页写「频率随维度指数衰减，高维编码细粒度位置信息」，方向与 $\theta_j=base^{-2j/d_k}$ 的定义相反（$j$ 大则频率小、旋转慢、波长长）；同时「base 通常设为 $10^4$ 到 $10^5$」在原论文只有 $10^4$ 一处，改为原始设定并引主源。方向性错误以页内「更正」callout 保留旧句与出处，未静默替换。source_count 不变（仍为 2）。
- `wiki/linear-attention-bias.md` — WHY：对照表 Rotary 行无引用；补 `[^src-roformer]`，并在「与其他位置编码的区别」后补一段说明 RoPE 的乘性注入、保范与线性注意力共存的原因，以及两种衰减（显式距离偏置 vs 多频旋转相位差）的来源差异。source_count 3→4。
- `wiki/siren-rope.md` — WHY：对照表「标准 RoPE」行（无时间信号、无可学习频率）无引用，补 `[^src-roformer]`；其余表述与原论文不冲突，未动。source_count 不变。
- `wiki/index.md` — WHY：[[rope]] 摘要变成含推导的内容，条目改为指向数据流与衰减上界；其余条目未变（无新页）。
- `wiki/log.md` — WHY：追加一条 maintenance 记录。

## 数据流自洽检查

逐行核对 `\bcancel` 所标消去维与结果形状：

| 行 | 标注 | 检查 |
|---|---|---|
| $Q=XW_Q$ | $[N,\bcancel{d_{\text{model}}}]\times[\bcancel{d_{\text{model}}},d]\to[N,d]$ | 左矩阵列 = 右矩阵行 $=d_{\text{model}}$，结果 $[N,d]$ ✓ |
| $\Phi=\mathbf m\Theta^\top$ | $[N,1]\times[1,d/2]\to[N,d/2]$ | 外积消去长度 1 维 ✓ |
| $\tilde Q_{m,i,:}$ | $[N,d/2,\bcancel{2}]\times[\bcancel{2},2]\to[N,d/2,2]$ | 每个 $(m,i)$ 上做 $[2]\times[2,2]$ 的 $2\times2$ 乘，$\bcancel{2}$ 是被换掉的组内维 ✓ |
| $\operatorname{rot}(Q)=QP_{\text{rot}}$ | $[N,\bcancel d]\times[\bcancel d,d]\to[N,d]$ | 与 $Q\odot C_{\text{dup}}$ 同形，可相加 ✓ |
| $S^{\text{att}}=\tilde Q\tilde K^\top/\sqrt d$ | $[N,\bcancel d]\times[\bcancel d,N]\to[N,N]$ | 消去单头维 ✓ |
| $A=\operatorname{softmax}(S^{\text{att}})$ | $[N,N]\xrightarrow{\text{沿列维逐行 softmax}}[N,N]$ | 规约发生在列维、形状不变 ✓ |
| $O=AV$ | $[N,\bcancel N]\times[\bcancel N,d_v]\to[N,d_v]$ | 消去 key 维 $N$ ✓ |
| $O^{\text{out}}=O^{\text{cat}}W_O$ | $[N,\bcancel{Hd_v}]\times[\bcancel{Hd_v},d_{\text{model}}]\to[N,d_{\text{model}}]$ | $H d_v$ 两侧对齐 ✓ |
| $u_m,z_n$ | $[N,\bcancel d]\times[\bcancel d,d]\to[N,d]$ | $R^d_{\Theta,m}$ 是 $d\times d$ 正交方阵，旋转前后同为通道维 $d$ ✓ |
| $\text{num}_m$ | $[N,\bcancel{d}]\times[\bcancel{d},d_v]\to[N,d_v]$ | 状态 $\in\mathbb{R}^{d\times d_v}$，与 $m$ 无关 ✓ |
| $\text{den}_m$ | $[N,\bcancel{d}]\times[\bcancel{d},1]\to[N,1]$ | 每行一个归一化标量 ✓ |

论文按单头书写，$d$ 取每头通道维还是模型维未指定；页面在多头处标 [INFERENCE]，未把推断写成原文。

## 长距离衰减结论的成立前提

页面把结论写成 $|S^{\text{att}}_{mn}|\sqrt d\le 2\|q_m\|\|k_n\|\,B(\Delta)$（$S^{\text{att}}$ 已除 $\sqrt d$，与式 (35) 的未归一内积差一个常数）、$B(\Delta)=\sum_{i=1}^{d/2}|S_{i+1}|$，并列出三条前提：(1) $d$ 偶且按相邻两维成对分组；(2) 频率为比值 $10000^{-2/d}$ 的几何序列；(3) $\|q_m\|,\|k_n\|$ 有界且不随位置增长（旋转保范使其成立）。两段形态（$\Delta=0$ 时 $|S_{j+1}|=j$、$B\approx\frac12(d/2)^2$，随机相位下落到 $\Theta(d^{3/2})$ 后持平）标为量级估计，并注明上界不紧、不给出逐对单调性、不直接约束 softmax。

## 关键数字的出处

- Table 1（§4.1.3）：Transformer-base 27.3，RoFormer 27.5。
- Table 2（§4.3.2）：MRPC 89.5/88.9、SST-2 90.7/93.5、QNLI 88.0/90.5、STS-B 87.0/85.8、QQP 86.4/71.2、MNLI 80.2/79.8 对 84.6/83.4。
- Figure 3 左（§4.2.3）：MLM 损失下降更快，图内无逐点数字；右（§4.4.2）：Performer 加 RoPE 后 Enwik8 损失更低。
- Table 4（§4.5.2）：六阶段准确率 65.0 / 66.8 / 64.6 / 63.4 / 67.4 / 66.2，最高在第 5 阶段（最大长度 1536）。
- Table 5（§4.5.4）：BERT-512 64.13/67.77、WoBERT-512 64.07/68.10、RoFormer-512 64.13/68.29、RoFormer-1024 66.07/69.79（验证/测试）。
- 式 (35)–(37)（§3.4.3）与 Figure 2：上界分解形式与「随相对位移下降」的图示。

## 矛盾处理

1. **外推能力**：论文摘要与 §3.3 把「序列长度灵活」列为 RoPE 的性质；[[source-alibi]] 记录 ALiBi 的评测结论是 rotary 位置编码在超出训练长度后劣化。两条陈述指向不同对象——前者是旋转对任意 $m$ 有定义，后者是无插值时的实测失效——因此不构成需要等待证据的源间矛盾，[[rope]] 写成「外推：性质表述与实测的分界」正文限定语，不设 `## 争议`，frontmatter 保持 `status: active`（`status: disputed` 要求正文存在 `## 争议`，二者不脱节）。未经插值或缩放的实测失效与后续扩展工作的位置都保留在正文，未把任一侧降级为过时论断。
2. **旧页对衰减的无条件表述**：`wiki/rope.md` 旧版写「这由 $\theta_i$ 的指数衰减和 Abel 变换保证」，是未带前提的普遍断言。本轮以同一源的式 (35)–(37) 替换，并把成立条件显式列出；旧措辞属同页同源的表述不准，不构成跨源历史论断，故直接改写而非另设「历史论断」节。
3. **GLUE 的选择性引用**：`wiki/roformer.md` 与 `wiki/source-roformer.md` 旧版只保留三项上升。源内 Table 2 与正文「三项 out of 六项」都支持全列，属于对同一源的片面摘取，直接补全并在页内点明论文原话。

## 数值备注

- §4.5.4 正文称 1024 长度比 WoBERT「绝对提升 1.5%」，Table 5 测试列差值为 1.69（69.79 对 68.10）、验证列差值为 2.00（66.07 对 64.07），三处口径并列记录，未择一。
- Figure 2 纵轴标注为 relative upper bound，式 (37) 后的说明给出所画量为 $\frac{1}{d/2}\sum_{i=1}^{d/2}|S_i|$，故纵轴即 $B(\Delta)/(d/2)$；曲线自约 20 降到 250 处的约 8，带周期性起伏不归零，与页面两段形态一致。
- 频率定义在文中出现两种编号：§3.2.2 式 (15) 为 $10000^{-2(i-1)/d},\,i\in[1,d/2]$，§3.3 与式 (35) 为 0 基的 $10000^{-2i/d}$。二者是同一 schedule 的下标平移，页面两种都写并说明关系，未改动任何一侧。
- §3.4.1 式 (31) 推出 $f_q=q\,e^{\mathrm i(m\theta+\gamma)}$、式 (33) 再取 $\gamma=0$；$\gamma$ 的取值在原文中逐次写入（式 (30)(31)(33)），页面按最终 $\gamma=0$ 的形式引用。
- 式 (32) 的提取文本写作 $q=f_q(x_m,0)=W_qx_n$，右端下标与左端不一致，疑为 PDF 下标转写误差（上下文均为 $x_m$）；页面不复述该式，只取其实质（空位置时 $f$ 退化为 $W_qx$）。
- Table 5 caption 称「第一列数字表示最大截断长度」，实际长度写在 Model 名内（BERT-512 / RoFormer-1024），引用时按行名给出。
- Table 2 caption 把 GLUE 拼作 "GLEU"；Table 3 caption 用 'abs'/'rel' 标注位置嵌入类型。两处排版笔误不影响数值。
- §4.2 的预训练只给 MLM 损失曲线（Figure 3 左），没有给出损失终值或收敛步数，页面只写「下降更快」。
- pdftotext 缺陷：式 (4) 被压平为 `pi,2t+1 = cos(k/100002t/d )`。原文是 $p_{i,2t+1}=\cos\!\big(k/10000^{2t/d}\big)$，分母带正指数，等价于逆频率 $10000^{-2t/d}$，与 RoPE 的 $\theta_{t+1}$（1 基）同源；页面据此写「同源」，未引用该式的具体数值。
