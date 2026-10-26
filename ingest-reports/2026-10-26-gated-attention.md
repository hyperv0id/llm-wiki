# Ingest 报告：Gated Attention for Large Language Models（arXiv:2505.06708）

## 通读范围

PDF 全文（`raw/gated-attention-for-large-language-models-2505-06708.pdf`，pdftotext `-layout`）：§1 Introduction、§2 Gated-Attention Layer（式 1–5）、§3 Experiments（§3.1 设置、§3.2.1 MoE 表 1、§3.2.2 dense 表 2）、§4 Analysis（§4.1 非线性式 6–8 + 表 3、§4.2 稀疏 + 表 4、§4.3 sink、§4.4 长上下文 + 表 5）、§5 Related Works、§6 Conclusion、Limitations、附录 A.1（表 6）、A.2（Fig. 4–5、稀疏比例）、A.3–A.5（逐层激活、门控分布、截断实验）。版本 arXiv:2505.06708v1（PDF 首页 2025-06-11 为编译日期，水印标 v1）。

## 创建

- `wiki/source-gated-attention-for-large-language-models-2505-06708.md` — WHY：该源的核心论断（$G_1$ 位置、非线性与稀疏两条机理、sink 消除、长上下文外推）在库内无落点，且清单条目 16 要求入库；页面名按 slug 规则取 raw 文件名，未用任务建议的 `source-qwen-2025-gated-attention`（否则 `[^src-*]` 与 `[[source-*]]` 的一一对应会断）。
- `wiki/sdpa-output-gating.md` — WHY：库内已有 `gated-temporal-attention`、`cross-gated-mechanism`、`gated-linear-units` 等「gated attention」页，全部是时序/门控线性单元语境，直接叫 `gated-attention-output` 会与它们混淆；以机制名 `sdpa-output-gating` 建 technique 页，承载门控维度、参数量推导、代数链条与五维消融网格。

## 修改

- `wiki/attention-sink.md` — WHY：库内 sink 页此前只有二手转述（TriAttention related work、MoBA 特例），本页补一手量测（F-Attn 0.467→0.048、M-Act 1053→94）与「大激活非 sink 必要条件」的反例行（$G_2$ 位置 125/0.297）；`[!warning]` 的二手转述说明收窄到 StreamingLLM/VATP 两条论断，source_count 2→3，confidence low→medium。
- `wiki/native-sparse-attention.md` — WHY：该页「范围与边界」原有「门控值分布未分析」一条，本页补「NSA 增益未拆出门控贡献、其 sigmoid 权重是分支级而非逐头逐通道」的粒度对照；source_count 2→3。
- `wiki/moba-attention.md` — WHY：只加一条「块级 top-$k$ 门控 vs 输出侧逐元素门控」的相关页面链接，未改任何论断，source_count 保持 4。
- `wiki/index.md` — WHY：Sources 段插 `source-gated-attention-for-large-language-models-2505-06708`（`source-fstllm` 之后）、Techniques 段插 `sdpa-output-gating`（`scale-wise-autoregressive-decoding` 之后），均按字母序各一条。
- `wiki/log.md` — WHY：追加本次 ingest 条目。

## 新建交叉链接

- [[attention-sink]] ↔ [[sdpa-output-gating]]
- [[sdpa-output-gating]] ↔ [[native-sparse-attention]]（分支级 vs 逐 head 逐通道）
- [[sdpa-output-gating]] ↔ [[moba-attention]]（块级 vs 输出侧逐元素）
- [[sdpa-output-gating]] ↔ [[rmsnorm]]（表 3 的零参数非线性对照）
- [[sdpa-output-gating]] ↔ [[gated-temporal-attention]]（同名异物，仅作区隔说明）

## 同名异物处理

库内 `factost`、`ustd`、`soft-shape`、`gated-temporal-attention` 等页出现的「gated attention」是时序预测模型内部的门控注意力/门控单元，来源、作用对象（时序或线性单元）与本源的 softmax attention 输出门控无关。处理方式：新建独立 technique 页 `sdpa-output-gating`，不改那些页的任何论断，仅在它的「相关页面」中以一行区隔说明指向 `gated-temporal-attention`。

## 数值备注

- **表 1 第 11 行 score shape**：记 $n\times q$，但其新增参数 0.2M 只在 $n\times k$ 下成立（$d_\text{model}\cdot k\cdot24=2048\times4\times24=0.197$M）；按同表其余行自洽，判为表注把 $k$ 写成 $q$。页面按「每 KV head 一个标量」写。
- **表 1 第 12/13 行 head-shared**：score shape 记 $n\times d_k$，但新增参数（201M / 25M）与逐元素变体相同；§3.2.1 正文说明 head-shared 是「对 $q$ 维平均，从 $n\times q\times d_k$ 得到 $n\times d_k$」，即 $W_\theta$ 仍按逐元素尺寸、参数不因共享而减少。两处口径并存，页面按正文实现口径写，并注明 $W_\theta$ 尺寸。
- **$d_\text{model}$ 与层数**：正文未给。由表 1 反解为 $d_\text{model}=2048$、24 层，且与其余各行自洽——逐元素 $G_1$：$2048\times32\times128\times24=201.3$M（表记 201）；headwise $G_1$：$2048\times32\times24=1.57$M（记 1.6）；$G_5$：$2048^2\times24=100.7$M（记 100）；$k=8$ 增量 $4\times2\times2048\times128\times24=50.3$M（记 50）；$q=48$ 增量 $16\times2\times2048\times128\times24=201.3$M（记 201）。页面标 [INFERENCE]。
- **表 4 第 5 行与表 1 第 6 行同配置不同值**：$G_2$ 逐元素（sigmoid、25M）在表 1 与表 3 记 PPL 5.820 / Hellaswag 74.38 / MMLU 59.17 / GSM8k 53.97，表 4 同配置记 5.820 / 74.38 / 59.17 / **51.33**。页面正文引用表 4 时只用门控均值/M-Act/F-Attn/PPL，未采用 51.33。
- **表 5 的小数位**：YaRN 行 baseline 32k 的跌幅记 −41.56，同行其余为一位小数；原表如此，页面照录 −41.6。
- **「超过 10 分」的口径**：摘要与 §1 称 RULER 收益 10 分以上，表 5 内 4k/8k 的差距是 5.2 / 8.5 分，≥10 分从 16k 起；64k/128k 只有 YaRN 外推行有数字，训练内两行该列为「-」。页面把这条限定写进边界。
- **表 2 第 7 行的表述**：§3.2.2 称 sandwich norm「restores convergence」但提升可忽略，表 2 该行 7.407 相对 4.0e-3 基线的 7.421 只差 0.014；页面按数字陈述，不引申。
- **massive activation 的源层**：§4.3 正文称源自 early layers "(e.g., layer 5)"，附录 A.3 称源自第 6 层 FFN 输出，两处层号不一致；页面引附录 A.3 的第 6 层。
- **摘要的 PPL 口径**：摘要称 "up to 0.2 PPL reduction"，表 1 实际 6.026−5.761=0.265，§3.2.1 称 "more than 0.2"，摘要口径偏保守；页面引用摘要说法时并列两行原值。
- **§5.2 的 dense 参数量**：§5.2 称 "dense (1B-parameter)"，表 2 实验为 1.7B，论文内部不一致。
