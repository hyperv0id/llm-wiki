# Ingest 报告：Extending Context Window of Large Language Models via Positional Interpolation（PI，arXiv:2306.15595v2）

源：https://arxiv.org/abs/2306.15595 ，v2（2023-06-28），Meta Platforms（Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian）。PDF 归档 `raw/position-interpolation-2306-15595.pdf`（`*.pdf` 规则排除，不入 git）；`pdftotext -layout` 抽出的 996 行文本（§1–§5 与 Appendix A–C）仅用于通读，未入库。

通读范围：全文，重点 §2.1（RoPE 背景）、§2.2（外推失效）、§2.3（PI 定义与 Theorem 2.1）、§3.1（训练与模型设置）、§3.2–3.5（Table 1–6）、§4（相关工作，含与 ViT 位置嵌入插值的区分）、§5、Appendix A（界证明）、Appendix B（$B(s)$ 可视化）、Appendix C（绘图代码）。

## 创建

- wiki/source-position-interpolation-2306-15595.md — WHY：全库此前没有 PI 原文的 source 页，PI 只作为二手表格行出现在 yarn、context-window-extension 两页；本页给问题、机制、理论界、Table 1–5 数字与边界各一条可引用落点。
- wiki/position-interpolation.md — WHY：PI 是 RoPE 扩窗谱系的起点，此前没有独立 technique 页；新建页承载 $m\to m/s$ 的推导、四阶段数据流、插值界与外推界的系数对比，以及按波长分档缺失的边界。

## 修改

- wiki/context-window-extension.md — WHY：PI 行的机制与最大扩展原为「等比例拉伸频率 / ~8x」且无引用；改为全部 $d/2$ 对同除 $s$、2048→32768（16×）并挂原文，补一句谱系定位，关键挑战第 1 条的机制半句加原文引用，新增 `## 历史论断` 记录被取代的 ~8x，source_count 4→5。
- wiki/ntk-aware-interpolation.md — WHY：动机段原只引 YaRN 说「PI 等比例拉伸导致高频被移除」，缺少 PI 侧事实；补 PI 的机制（频率统一除以 $s$）与它自己报告的扩窗结果，source_count 1→2。
- wiki/yarn.md — WHY：谱系块里 PI 一项改为可链接文本并指向新页；长序列 perplexity 表的 PI 行与 PI 原文数值差一个量级，补协议注说明两者口径不同，source_count 1→2。
- wiki/rope.md — WHY：后续发展一栏的上下文扩展列表缺 PI，补 wikilink（结构性句子，不加引用，source_count 不变）。
- wiki/learnable-frequency-scaling.md — WHY：频率策略对照表的 PI 行原为纯文本，加 wikilink，source_count 不变。
- wiki/ntk-by-parts-interpolation.md — WHY：优势段两处「PI」为纯文本，新页入库后补 wikilink 指向 [[position-interpolation]]，source_count 不变。
- wiki/dynamic-scaling.md — WHY：动机第 1 条讲固定 $s$ 的问题，固定 $s$ 就是 PI 的用法，加 wikilink 指明对应技术页。
- wiki/index.md — WHY：Sources、Techniques 各新增一条，插入位置按字母序（`source-pn-train` 之后、`position-extrapolation` 之后），刷新 last_updated。
- wiki/log.md — WHY：一条 ingest 条目 + 一条 contradiction 条目（情况 1）。

## 新建交叉链接

- [[position-interpolation]] ↔ [[context-window-extension]]
- [[position-interpolation]] ↔ [[ntk-aware-interpolation]]
- [[position-interpolation]] ↔ [[ntk-by-parts-interpolation]]（PI 页指向其分档做法）
- [[position-interpolation]] ↔ [[yarn]]
- [[position-interpolation]] ↔ [[dynamic-scaling]]（固定 $s$ 与自适应 $s$）
- [[position-interpolation]] ↔ [[learnable-frequency-scaling]]
- [[position-interpolation]] ↔ [[rope]]
- [[source-position-interpolation-2306-15595]] ↔ [[position-interpolation]]

## 数值备注

- **扩展倍数自相矛盾**：§3 开头写「extend context window up to 32 times of the original size」，实验最大只到 32768，对 2048 是 16×（Table 1、Table 2 最右列）。页面写 16×，并在 log 数值备注里记录 32× 的原句。
- **退化幅度正文与表格不一致**：§3.4 称扩到 8192 的模型在原窗口基准上「degradation of up to 2%」；Table 5 中 7B 的 BoolQ 从 76.1 到 73.2 为 2.9 点、Race-M 55.7→53.8 为 1.9 点。正文与表格并列保留，页面两处都写。
- **32768 窗口的 perplexity 抬头**：Table 2 上 7B 在 32768 评估窗口为 2.48，高于其 16384 窗口的 2.24；13B 为 2.35 vs 2.15。§3.2 的「趋势延伸到 32768 不衰减」对应 PG19（Table 1 确实单调降到 6.77），两份数据不冲突但不通用。
- **Table 4 的异常格**：7B/32768 的 PI 行在 600 步一档记为 18432，其余步数（200/400/800/1000）均为 32768；照抄未平滑。
- **与 YaRN 一侧 PI 数值差距大**：wiki/yarn.md 的 Proof-pile PI 行为 5.70/4.64/3.97/3.57（2k/8k/16k/32k），PI 原文 Table 2 给 7B 的 2k/8k/16k 为 2.79/2.39/2.25。差距来自微调协议不同（原文 Pile 1000 步全参数），已在 yarn 页加注说明，不互相校准、不改 YaRN 页数值。
- **源转换缺陷**：`pdftotext -layout` 在双栏处把页标题混入正文（如 §2.3 前的 "4" 页码），Theorem 2.1 与式 (5)、(8) 的求和上下限断行，Appendix A 的式 (11)–(17) 需按行拼接；Table 1–6 的数值块逐行对齐后可读，页面数字均按表格列位置取。
- **未核**：PI 论文只提并行的 kaiokendev blogpost，未给其链接之外的细节；本仓库无该博客源，未据此补任何结论。

## 数据流与维度自洽检查

- Stage 1：$[N]\to[N]$ 逐元素缩放，$\Delta_{mi}$ 由 $[N]\times[N]$ 给出 $[N,N]$，与 Stage 3 中 $a'(m,i)$ 的两个位置下标对应。
- Stage 2：$[d,N]$ 按 $j$ 分 $d/2$ 对，每对乘 $e^{\mathrm{i}(m/s)\theta_j}$，输出仍 $[d,N]$（旋转保模长），与 [[rope]] 的列向量 $x\in\mathbb{R}^d$ 写法一致。
- Stage 3：沿 $j$ 求和的 $[d/2]$ 维消去为标量，与 Stage 2 的配对维一致；$m$、$i$ 只在差中出现，与「绝对下标抵消」的结论一致。
- Stage 4：损失对 $t=1..L'$ 求和，位置下标取 $p_t=t/s$，与 Stage 1 的映射同一条；无新参数，故梯度只更新原有权重。

## 未做

- 未提交、未跑 lint、未改 `.obsidian/`、未改既有 `raw/` 文件（`raw/` 只新增 `position-interpolation-2306-15595.pdf` 一份副本）。
- 未改 [[attention-temperature-scaling]]、[[position-extrapolation]]：两页涉及 PI 的表述均以 YaRN 为源、且不与 PI 原文冲突（温度缩放与 PI 无关；extrapolation 页讲 ALiBi 口径的 Rotary 外推距离），无需补第二引用。
