---
title: Trace as State 方法复现技术笔记
type: analysis
tags: [analysis, reproduction, trace-as-state, long-context, token-usage]
created: 2026-09-13
last_updated: 2026-09-13
source_count: 4
confidence: medium
status: active
---

# Trace as State 方法复现技术笔记

本文是 [[trace-as-state]] 的复现核对页：逐步还原论文实验的可操作细节（模型配置、运行流程、序列化、判分、token 统计），核对公开材料，并划出可还原与不可还原的边界。全部核对基于论文 v1（arXiv:2609.02702v1，2026-09-02，21 页含附录 A–F）与其 arXiv TeX 源包，未实际调用任何商业 API、未做付费推理 [^src-trace-as-state]。机制与结果数字的解读见 [[trace-as-state-evidence]]，理论部分见 [[conditional-state-update]]，相关工作见 [[trace-as-state-positioning]]。

## 1. 复现边界总览

论文公开了部分固定文本、插入边界与判分规则，但没有发布实验 runner、原始 records 或逐题配置（本页所查的论文正文、TeX 源与 arXiv 页面未见发布声明）。因此：**论文给出的固定文本、边界与判分规则以及 token 聚合口径可从公开材料核对；完整 runner 行为、解码参数、数据快照版本、NUB 的 preamble 与 judge prompt、逐题/逐美元成本不可还原** [^src-trace-as-state]。本页不声称完成过任何复现，只给出认真复现所需的核对清单。

## 2. 模型配置核对（§4.1 Setup 与 Table 1-1）

三个模型来自三个不同供应商，均为主贡献方官方渠道调用 [^src-trace-as-state]：

| 配置项 | Qwen 3.7 Max | DeepSeek V4 Pro Preview | GLM-5.2 |
| --- | --- | --- | --- |
| 供应商 | Aliyun Bailian | DeepSeek 官方 | Bigmodel |
| 架构归因 | GDN + GA（Gated Deltanet + Gated Attention） | CSA + HCA + mHC（Compressed Sparse Attention + Heavily Compressed Attention + Manifold-Constrained Hyper-Connections） | DSA + IC（DeepSeek Sparse Attention + IndexCache，1M-token MoE） |
| Input Budget | 983,616 | 1,048,576 | 1,048,576 |
| Output Budget | 65,536 | 131,072 | 65,536 |
| Reasoning | `xhigh`（经 prompt） | `max` | `max` |

（Table 1-1 与 §4.1，数值逐项照录 [^src-trace-as-state]。）

核对要点：

- **版本消歧**：论文明确 DeepSeek V4 Pro 有两个同名版本（2026-4-24、2026-8-13 发布），实验用 **2026-4-24 版**；Table 1-1 称 "DeepSeek V4 Pro Preview"，§4.1 模型列表同 [^src-trace-as-state]。
- **effort 投递位置**：Qwen 3.7 Max 的 `xhigh` 是官方评估指引推荐的 **prompt 文本**，不是 provider 侧 `reasoning_effort` 参数值；MRCRv2 的 runner 把该指令前置于 system message，GraphWalks 与 NUB-1M 的 runner 前置于单条 user message，且同一数据集内投递格式跨条件固定（Appendix C）[^src-trace-as-state]。指令全文照录：
  > Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.

  （Appendix C 逐字引用，无改动 [^src-trace-as-state]。）
- DeepSeek 与 GLM 的 `max` effort 如何传给 provider（参数名、是否同为 prompt 约定）论文未写明，只能按各 provider 文档自行对齐 [^src-trace-as-state] [INFERENCE]。
- **未设置 max-output**：作者声明「We do not explicitly pass a custom maximum-output value」——run records 证明客户端没有请求上限，但 provider 生效的默认输出上限不可知且可能随时间变化（§4.1）[^src-trace-as-state]。因此 **Table 1-1 的 Output Budget 是 provider 侧的预算信息，不能冒充调用时传入的 `max_tokens`** [^src-trace-as-state]。
- 三个模型在所评估接口均支持约 1M-token 输入（§4.1）[^src-trace-as-state]。
- **未公开的解码参数**：论文未报告 solver 调用的 temperature、top_p 或采样 seed；文中出现的 seed 0 属统计过程（paired bootstrap 20,000 resamples，Appendix E 与 Table 5），不是模型调用参数 [^src-trace-as-state]。这是逐轮复现的主要缺口之一。

## 3. 数据与运行流程（Table 1-2 与 §4.1）

| 项 | GraphWalks | MRCRv2 | NUB-1M |
| --- | --- | --- | --- |
| 子集 | 256K | 256K, 512K, 8-needle | Season 2 |
| Problems | 200 | 200 | 20 |
| Repeats | 5 | 5 | 5 |
| Scoring | EM, set F1 | EM, Seq.（SequenceMatcher ratio） | Acc. |

（Table 1-2 逐项照录 [^src-trace-as-state]。）

流程逐步（§4.1、§4.4、Appendix C）[^src-trace-as-state]：

1. **5 条首轮 traces**：每个 (model, task, problem) 先跑 baseline $\mathcal M([x,q])$ 5 次，得 5 组 $(r_j, a_j)$（reasoning trace + separate visible answer）；问题 $q$ 与长上下文 $x$ 分离，问题永远在输入末尾 [^src-trace-as-state]。
2. **固定 serializer**：5 个 reasoning 字段按 repeat-index 顺序经同一 serializer 构成 $T$（见 §4）；**separately returned visible answers 一律留在 $T$ 之外**（Appendix C）[^src-trace-as-state]。
3. **same-T 前后置**：同一个已实现的 $T$ 被同时用于 $\mathcal M([T,x,q])$（Trace as State）与 $\mathcal M([x,T,q])$（Trace Append），两条件输入仅差 $T$ 的位置；trace-count 消融明确写道 "For every $n\geq1$, TRACE AS STATE and TRACE APPEND receive the same realized $T$"（§4.4）[^src-trace-as-state]。
4. **5 次 second-pass repeats**：每个 trace-backed 条件每题做 5 次 fresh second-pass repeats（§4.1 说 "We evaluate each problem with 5 repeats and include all 5 reasoning traces"；§4.4 消融明说 "averages five fresh second-pass repeats per problem"）[^src-trace-as-state]。
5. **异常处理**：blocked、overlong、malformed、missing、nonterminal、content-filtered 一律计失败（§4.1）[^src-trace-as-state]。

GraphWalks 选 256K 桶、MRCRv2 选 256K/512K 桶的原因：各模型 tokenizer 不同，1M 桶内许多题会超长，选 1M 以下最长桶并给 Trace as State / Trace Append 追加文本留余量（§4.1）[^src-trace-as-state]。

## 4. 序列化与插入边界（Appendix C）

### 4.1 trace block 结构（符号化转写）

论文以表格给出固定 serializer 的结构（Appendix C Table "trace-serializer"）。下为**结构转写**，展示五条 trace 的组织方式，**不是完整可运行 prompt**：标签与 trace 编号行是论文给出的逐字接口文本，trace 内容本身用符号 $r_j$ 表示；$P_\mathcal D$ 是数据集专属 preamble，属于 $T$，不属于 $x$ [^src-trace-as-state]：

```
P_D
<first_run_reasoning_traces>
[Trace 1] r_1
[Trace 2] r_2
[Trace 3] r_3
[Trace 4] r_4
[Trace 5] r_5
</first_run_reasoning_traces>
```

论文对实际调用时 $r_j$ 的内部文本无逐行披露（trace 是模型首轮 reasoning 全文，经截断规则处理后整体放入），复现者只能从自己的首轮运行产生这些内容 [^src-trace-as-state]。

### 4.2 逐字固定文本（论文原文保留）

**GraphWalks 的 $P_\mathcal D$**（Appendix C 逐字）[^src-trace-as-state]：

> Below are selected reasoning traces or trace tail windows from independent first attempts on the same graph problem. They may contain mistakes. Use them only as scratchpad hints, and verify against the graph.

**GraphWalks system prompt**（Appendix C 逐字）[^src-trace-as-state]：

> You solve directed-graph algorithm problems. Use only the graph and operation in the user message. Return exactly one visible line in this format: Final Answer: [node1, node2]. Use [] for the empty set. Do not include any text before or after that line.

**GraphWalks 答案格式指令**（runner 追加到 user message，Appendix C 逐字）[^src-trace-as-state]：

> Return exactly one line in this format: Final Answer: [node1, node2]. Use [] for the empty set.

**MRCRv2 的 $P_\mathcal D$**（Appendix C 逐字）[^src-trace-as-state]：

> Below are reasoning traces from independent first attempts on the same MRCR problem. They may contain mistakes. Use them only as scratchpad hints; verify against the conversation and final request. Do not copy any trace text into the visible answer.

### 4.3 插入边界

- **GraphWalks $[T,x]$**：完整 trace block 前置于完整 released GraphWalks prompt，答案格式指令跟在 prompt 之后 [^src-trace-as-state]。
- **GraphWalks $[x,T]$**：runner 在 released prompt 的**最后一个 `Operation:` 块之前**切开——图指令与边列表在前，然后是同一 trace block，再接最后的 operation 与答案格式指令（Appendix C）[^src-trace-as-state]。这与数据集卡描述的 prompt 结构（3-shot 示例 + 图 + operation）一致 [^src-graphwalks-dataset]。
- **MRCRv2**：只给了 preamble 文本，未给出 $T$ 相对多轮对话的具体插入位置、system prompt 或 user message 结构 [^src-trace-as-state]。
- **NUB-1M**：$P_\mathcal D$ 的具体文本、插入边界、system prompt **论文均未给出**（Appendix C 无 NUB 段落）[^src-trace-as-state]。以下没有官方模板可抄——复现者须自定或向作者索取，不得伪造"官方"文本。

### 4.4 截断与净化规则

- **50,000 是每条 reasoning 字段的字符上限**，非 token：超过的字段替换为前 50,000 字符加 `...`，随后 block formatting 去除首尾空白（Appendix C）[^src-trace-as-state]。
- **visible answer 不进入 $T$**，但论文只声明 separate visible answers 留在 $T$ 外，**未说明对 trace 内部复述的候选答案做任何净化**；reasoning 正文仍可能携带答案文本（Appendix C 未提，[INFERENCE]）[^src-trace-as-state]。
- **措辞不一致点**：§4.1 举例性写作 "delimiters such as `<trace_start>` and `<trace_end>`"，而 Appendix C 的实际固定标签是 `<first_run_reasoning_traces>`/`[Trace k]`；以附录 C 逐字文本为准 [^src-trace-as-state]。GraphWalks preamble 提到 "trace tail windows"，但实际实现是**截前** 50,000 字符（§4.1、Appendix C）[^src-trace-as-state]。

## 5. 判分规则与异常（Appendix B）

### GraphWalks

- 评估器取**最后一行** `Final Answer: [...]`，解析节点集与 gold 集比较 [^src-trace-as-state]。
- 空集规则：两个空集 EM 与 F1 均 **1**；一空一非空均 **0**；响应缺失、答案缺失/畸形/非 terminal 记 **0**（Appendix B）[^src-trace-as-state]。
- Question First 资格记录：保守地把 null/unknown 终止标记当 length-limited——121 个 null、0 个 unknown、60 个原生标记，共 181 个 length-limited（160 BFS + 21 Parents），另有 1 个 stopped BFS 输出缺 terminal 语法；**182 个输出全部记 0**（Appendix B）[^src-trace-as-state]。这是资格性记零（终止/格式问题），不等于 182 个"明确推理错误" [^src-trace-as-state]。

### MRCRv2 8-needle

- 每题带一个随机前缀，必须同时作为 prediction 与 reference 的开头；scorer 校验并移除前缀后再算 EM 与 `difflib.SequenceMatcher` ratio；剩余串完全一致 EM=1 否则 0；论文把该 ratio 标为 *Seq.*，与 GraphWalks set F1 是两个不同指标；前缀缺失或非法则两指标均 0（Appendix B）[^src-trace-as-state]。
- 部分 prompt 触发 provider 内容过滤；**被过滤输出留在分母并记 0**（Appendix B）[^src-trace-as-state]。

### NUB-1M（judge 归因）

- 每题维护一个 reference answer，用 **DeepSeek V4 Pro** 做 judge，比较抽取的 solver answer 与 reference；judge prompt **排除 solver 身份与 feedback order（盲评）**；judge 返回布尔判定 + 简短 rationale，取布尔为二值分（Appendix B）[^src-trace-as-state]。
- 作者对手工复核 judged outputs 的自述：**"We also manually reviewed the judged outputs and found no errors."**——这是作者自述的复核结论，不可外推为独立审计（Appendix B）[^src-trace-as-state]。
- judge prompt 全文与**具体的 NUB preamble**论文未给出，不可补造 [^src-trace-as-state]。

### 数据集卡与论文判分的口径分离

GraphWalks 数据集卡公开了示例抽取/F1 代码（取最后一行 + `Final Answer:` 正则 + F1 表达式），但其 F1 表达式在 recall+precision=0 时返回 1；论文 Appendix B 自述的空集与异常规则（一空一非空 = 0）与之不同口径。**实验评分以论文 Appendix B 为准，不能声称论文直接使用了卡片代码** [^src-graphwalks-dataset]。MRCR 数据集卡官方示例代码为 startswith 校验 + `removeprefix` 后直接 `SequenceMatcher(...).ratio()`，**无 whitespace strip 语句**（卡片文字提到 stripped，文字与代码不一致）；论文 Appendix B 同样未声明 whitespace normalization，且在 ratio 之外另加 EM 指标 [^src-mrcr-dataset]。

## 6. Token 统计（Appendix F，Table 6 全 36 行）

Table 6 是 provider API 返回的 solver 调用聚合 token 量（单位：百万），按"有 provider-reported usage 的 retained responses"聚合；missed input = 总输入 − cached input；**reasoning tokens 计入 output**；First Pass 是构建 $T$ 的共享首轮；TRACE AS STATE 与 TRACE APPEND 的 Total 列把 First Pass 分量逐项相加（Appendix F）[^src-trace-as-state]。36 行按模型分列照录：

### DeepSeek V4 Pro

| Benchmark | Condition | Cached in | Missed in | Output | Total Cached | Total Missed | Total Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GraphWalks 256K | First Pass | 206.766 | 51.092 | 62.507 | — | — | — |
| GraphWalks 256K | TRACE APPEND | 271.017 | 68.554 | 35.231 | 477.783 | 119.646 | 97.738 |
| GraphWalks 256K | TRACE AS STATE | 277.074 | 62.501 | 27.975 | 483.840 | 113.593 | 90.482 |
| MRCRv2 256K | First Pass | 50.817 | 46.961 | 1.149 | — | — | — |
| MRCRv2 256K | TRACE APPEND | 79.982 | 20.820 | 0.690 | 130.799 | 67.782 | 1.839 |
| MRCRv2 256K | TRACE AS STATE | 51.875 | 50.051 | 1.355 | 102.692 | 97.012 | 2.504 |
| MRCRv2 512K | First Pass | 155.283 | 38.195 | 0.989 | — | — | — |
| MRCRv2 512K | TRACE APPEND | 178.675 | 18.526 | 0.702 | 333.958 | 56.721 | 1.691 |
| MRCRv2 512K | TRACE AS STATE | 157.261 | 39.940 | 1.241 | 312.544 | 78.135 | 2.230 |
| NUB-1M Season 2 | First Pass | 31.651 | 8.418 | 0.817 | — | — | — |
| NUB-1M Season 2 | TRACE APPEND | 39.589 | 4.460 | 0.184 | 71.240 | 12.878 | 1.001 |
| NUB-1M Season 2 | TRACE AS STATE | 27.418 | 16.637 | 0.398 | 59.069 | 25.055 | 1.215 |

### Qwen 3.7 Max

| Benchmark | Condition | Cached in | Missed in | Output | Total Cached | Total Missed | Total Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GraphWalks 256K | First Pass | 260.213 | 97.483 | 5.402 | — | — | — |
| GraphWalks 256K | TRACE APPEND | 264.906 | 117.866 | 2.865 | 525.119 | 215.350 | 8.267 |
| GraphWalks 256K | TRACE AS STATE | 271.742 | 111.030 | 3.615 | 531.955 | 208.514 | 9.017 |
| MRCRv2 256K | First Pass | 38.337 | 63.495 | 2.111 | — | — | — |
| MRCRv2 256K | TRACE APPEND | 60.227 | 51.195 | 0.989 | 98.564 | 114.690 | 3.100 |
| MRCRv2 256K | TRACE AS STATE | 62.213 | 49.209 | 1.397 | 100.550 | 112.704 | 3.508 |
| MRCRv2 512K† | First Pass | 134.607 | 63.738 | 2.263 | — | — | — |
| MRCRv2 512K† | TRACE APPEND | 154.827 | 53.489 | 1.211 | 289.434 | 117.226 | 3.474 |
| MRCRv2 512K† | TRACE AS STATE | 153.971 | 54.345 | 1.724 | 288.577 | 118.083 | 3.988 |
| NUB-1M Season 2 | First Pass | 34.253 | 7.547 | 0.489 | — | — | — |
| NUB-1M Season 2 | TRACE APPEND | 38.184 | 5.773 | 0.253 | 72.437 | 13.320 | 0.742 |
| NUB-1M Season 2 | TRACE AS STATE | 32.414 | 11.549 | 0.419 | 66.667 | 19.096 | 0.908 |

### GLM-5.2

| Benchmark | Condition | Cached in | Missed in | Output | Total Cached | Total Missed | Total Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GraphWalks 256K | First Pass | 134.748 | 150.760 | 19.463 | — | — | — |
| GraphWalks 256K | TRACE APPEND | 55.612 | 265.396 | 9.127 | 190.361 | 416.156 | 28.591 |
| GraphWalks 256K | TRACE AS STATE | 64.390 | 256.618 | 16.491 | 199.138 | 407.377 | 35.954 |
| MRCRv2 256K | First Pass | 73.782 | 24.718 | 1.891 | — | — | — |
| MRCRv2 256K | TRACE APPEND | 82.413 | 20.940 | 2.518 | 156.195 | 45.658 | 4.409 |
| MRCRv2 256K | TRACE AS STATE | 82.464 | 20.888 | 1.516 | 156.246 | 45.606 | 3.407 |
| MRCRv2 512K | First Pass | 151.247 | 42.262 | 1.441 | — | — | — |
| MRCRv2 512K | TRACE APPEND | 157.740 | 40.508 | 2.002 | 308.987 | 82.770 | 3.443 |
| MRCRv2 512K | TRACE AS STATE | 157.458 | 40.790 | 1.648 | 308.705 | 83.052 | 3.089 |
| NUB-1M Season 2 | First Pass | 1.272 | 41.123 | 0.995 | — | — | — |
| NUB-1M Season 2 | TRACE APPEND | 3.840 | 43.087 | 0.484 | 5.111 | 84.209 | 1.478 |
| NUB-1M Season 2 | TRACE AS STATE | 0.457 | 46.475 | 0.856 | 1.728 | 87.598 | 1.850 |

### 解释与边界

- **这是什么聚合**：provider API 返回的 usage 字段聚合值，覆盖 main-result 的 solver 调用；**不是逐题、逐美元、逐墙钟、不剔失败的账单**，也没有 usage 缺失调用的处理说明（Appendix F 只说 "For each retained response with provider-reported usage"）[^src-trace-as-state]。
- **output 含 reasoning tokens**：论文只声明 "Reasoning tokens are included in the output"，未给出可见答案与 reasoning 的拆分列，故 output 列不能当作"可见答案"成本，也不能从表中推断 reasoning 与可见答案的比例 [^src-trace-as-state]。
- **缓存不必然更少**：GLM GraphWalks 中，首轮 cached 134.748M，而输入更长的 TRACE APPEND 仅 cached 55.612M（missed 265.396M）；GLM NUB 的 TRACE AS STATE cached 仅 0.457M，约 1% 命中率 [^src-trace-as-state]。这一现象与调用时序、provider 路由或缓存机制的关联属 [INFERENCE]，论文未测缓存成因。
- **已测条件间的对照**：两条件的 Total 都含同一个 First Pass，未舍入数值上的差应等于 second-pass 之差；原表分别保留三位小数，直接对印刷数值相减时部分分量会差 0.001M。下节按 `Tokens` 三列计算，保留原始 `Total` 列，不将其改成由已舍入值重算的版本。[^src-trace-as-state]
- **†号悬空**：Table 6 中 Qwen MRCRv2 512K 带 †，但 TeX 源该表无任何 tablenotes，PDF 全文亦检索不到 † 的说明文字——v1 中该标记未解释 [^src-trace-as-state]。

### 已测条件的增量与定价边界

[INFERENCE] 下表由本次核算直接从 Table 6 的 `Tokens` 三列相减得到，定义为 **Trace as State − Trace Append**，单位仍是百万 token。正数表示该分量更多，负数表示更少；它们不是论文报告的新实验，也不是逐题或逐美元成本。[^src-trace-as-state]

| 模型 | 基准 | Δ Cached input | Δ Missed input | Δ Output |
|---|---|---:|---:|---:|
| DeepSeek V4 Pro | GraphWalks 256K | +6.057 | −6.053 | −7.256 |
| DeepSeek V4 Pro | MRCRv2 256K | −28.107 | +29.231 | +0.665 |
| DeepSeek V4 Pro | MRCRv2 512K | −21.414 | +21.414 | +0.539 |
| DeepSeek V4 Pro | NUB-1M Season 2 | −12.171 | +12.177 | +0.214 |
| Qwen 3.7 Max | GraphWalks 256K | +6.836 | −6.836 | +0.750 |
| Qwen 3.7 Max | MRCRv2 256K | +1.986 | −1.986 | +0.408 |
| Qwen 3.7 Max | MRCRv2 512K | −0.856 | +0.856 | +0.513 |
| Qwen 3.7 Max | NUB-1M Season 2 | −5.770 | +5.776 | +0.166 |
| GLM-5.2 | GraphWalks 256K | +8.778 | −8.778 | +7.364 |
| GLM-5.2 | MRCRv2 256K | +0.051 | −0.052 | −1.002 |
| GLM-5.2 | MRCRv2 512K | −0.282 | +0.282 | −0.354 |
| GLM-5.2 | NUB-1M Season 2 | −3.383 | +3.388 | +0.372 |

[INFERENCE] 这些分量的方向并不统一：DeepSeek GraphWalks 的前置条件少输出 7.256M token，GLM GraphWalks 却多输出 7.364M；后者同时少 8.778M 未命中输入。不能从准确率提升推出“输出一定更短”或“费用一定更低”。GraphWalks 成本还是 BFS 与 Parents 合计，不能把全部增量归到单独的 Parents 增益。[^src-trace-as-state]

若已知**实验对应模型版本、计费时段和单位**的单价，可用以下记账式计算所记录调用的 token 费用部分；这里的单价不代入当前 API 报价：[INFERENCE]

$$
C_{\mathrm{tokens}}
=p_{\mathrm{cached}}N_{\mathrm{cached}}
+p_{\mathrm{missed}}N_{\mathrm{missed}}
+p_{\mathrm{output}}N_{\mathrm{output}}.
$$

`N` 为百万 token 数时，`p` 应为每百万 token 单价。两条件之差用对应的 `ΔN` 代入；计算整条已记录 pipeline 时用原表 `Total tokens`，而非只用第二遍。论文没有提供完整计费明细、缺失 usage 的调用成本或 NUB judge 的独立费用表，因此这个公式不能补出整场评测的真实总费用（Appendix F）。[^src-trace-as-state]

[INFERENCE] 同样不能把聚合 token 量当作逐调用提示严格配对的审计：例如 DeepSeek MRCRv2 256K 的第二遍总输入为 Append 100.802M、As State 101.926M，相差 1.124M。Appendix F 只统计有 usage 的 retained responses；缺少逐条 records 时，无法从这两项合计判定差异来自保留记录范围、tokenization 或其他实现细节。这是公开材料的核对边界，不足以断言 same-`T` 设计被违反。[^src-trace-as-state]

## 7. 公开材料核查

### 论文自身材料

- **arXiv 摘要页**（https://arxiv.org/abs/2609.02702，访问 2026-09-13）：Comments 仅 "preprint"，无代码/数据链接栏；PDF、HTML、TeX Source 三个全文入口 [^src-trace-as-state]。
- **arXiv TeX 源包**（本仓库 `downloads/trace-as-state.tar.gz`，41 个成员 = 33 个普通文件 + 8 个目录；33 个普通文件为 json 1、bst 1、sty 1、tex 17、pdf 12、bib 1）：`00README.json` 声明全部条目为 TeX 源与图（toplevel 仅 `main.tex`，其余 usage: ignore，编译器 pdflatex、TeXLive 2025）；`analysis/` 下只有 `paired_placement_bootstrap/paired_placement_bootstrap_table_compact.tex` 一个统计表源文件；**没有 runner、没有原始 run records** [^src-trace-as-state]。
- **官方实现状态**：在所核查的论文正文、references.bib 与 arXiv 页面中**未见**作者实验代码/runner 的发布声明；论文以 "run records" 指内部记录（§4.1），但该记录未随论文公开 [^src-trace-as-state]。不据此做"全网不存在官方实现"的断言。

### 列出的公开基准来源

- **GraphWalks**：https://huggingface.co/datasets/openai/graphwalks（论文引用，注明数据集卡，2026-08-03 访问）[^src-trace-as-state]。快照核查见 [[source-graphwalks-dataset]]：3-shot prompt 结构、`bfs`/`parents` 两类、官方抽取/F1 示例代码、2025-04-12 初版与 2026-02-27 bugfix changelog [^src-graphwalks-dataset]。**论文未声明采用的 GraphWalks commit/快照**，2/27/26 修复针对 `128k_and_shorter` 的 24/400 parents 样本与 BFS 重访歧义，不能推断论文 256K 桶用了哪份版本 [^src-graphwalks-dataset]。
- **MRCR**：https://huggingface.co/datasets/openai/mrcr（论文引用，同上）[^src-trace-as-state]。快照核查见 [[source-mrcr-dataset]]：2/4/8-needle 任务定义、hash 前缀门槛、SequenceMatcher ratio 官方示例、每桶 100 样本、2025-12-05 bugfix（约 10% 样本 needle 过多、约 5% gold 错误，加 `date_added` 字段）[^src-mrcr-dataset]。**"MRCRv2" 这一命名在数据集卡上不存在**，卡片未定义任何 v2 版本指向；论文 200 题 = 256K/512K 两桶 × 每桶 100 样本恰好吻合卡片口径，但属于 [INFERENCE] 的版本对应，论文未声明 [^src-mrcr-dataset]。
- **NUB-1M**：https://github.com/xz-keg/Novel-Understanding-Bench（论文引用，注明 Project repository，2026-08-03 访问）[^src-trace-as-state]。快照核查见 [[source-nub-1m-benchmark]]：每季一部新小说 20 题、Season 2 目录只含 `novel.txt` + `probs.txt`（**当季答案不公开**，下季才放出）、README 排行榜为 1-pass 口径（DeepSeek V4 Pro 0424: 12、GLM 5.2: 9、Qwen 3.7 Max: 8，满分 20），与论文 5-repeat 平均准确率口径不同——论文 First Pass 准确率为 DeepSeek 60.0、GLM 43.0、Qwen 37.0（Table 2），与排行榜（12/20=60%、9/20=45%、8/20=40%）接近但不可互为核对表 [^src-nub-1m-benchmark]。论文自述 "We maintain a reference answer for each problem"（Appendix B）[^src-trace-as-state]；所查 Season 2 目录未公开答案，作者 reference answers 的来源不能由该目录确证，也不据此断言其他渠道是否存在 [^src-nub-1m-benchmark]。

## 8. 可还原与不可还原清单

**论文公开文本可核对的项**（论文给出固定文本或明确规则）[^src-trace-as-state]：

- trace block 结构、五个数据集 fixed interface 文本（GraphWalks preamble/system/answer-format、MRCR preamble、Qwen xhigh 指令全文）；
- 50,000 字符截断 + `...` + strip 规则、repeat-index 顺序、visible answer 排除；
- GraphWalks $[x,T]$ 在最后一个 `Operation:` 块前切分；
- 判分规则：GraphWalks 末行/空集/异常、MRCR 前缀校验/SequenceMatcher/内容过滤记 0、NUB 盲评布尔、异常计失败、Question First 182 记零的构成；
- Table 1-1/1-2 全部数字与 Table 6 全部聚合 token。

**不可还原（需作者提供或实验侧自行决策）**[^src-trace-as-state]：

1. temperature/top_p/seed 等全部解码参数；
2. NUB-1M 的 $P_\mathcal D$ preamble 文本、插入边界、system prompt 与 judge prompt 全文；
3. MRCRv2 的 $T$ 插入位置与消息结构；
4. DeepSeek/GLM `max` effort 的传递方式与 provider 生效默认输出上限；
5. GraphWalks 数据快照 commit、MRCR "v2" 与卡片版本的对应、NUB 季节小说/题目的确切版本与作者自持 reference answers；
6. 逐题/逐调用 records、失败与 usage 缺失调用明细、墙钟与美元成本；
7. provider 缓存行为依赖的调用时序（Table 6 的 cached/missed 列因此不可独立复现）；
8. Appendix B 开头承诺的 "record-selection rules" 细则正文未完整展开，仅有 GraphWalks 资格记录与 Appendix F 的 retained-with-usage 一句。

## 相关页

[[trace-as-state]] · [[source-trace-as-state]] · [[trace-as-state-evidence]] · [[trace-as-state-positioning]] · [[conditional-state-update]] · [[source-graphwalks-dataset]] · [[source-mrcr-dataset]] · [[source-nub-1m-benchmark]]

[^src-trace-as-state]: [[source-trace-as-state]] —— Xu Zou, Jie Tang. *Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers*. arXiv:2609.02702v1, 2026-09-02。本页所有论文内事实引自该来源的 §4、Appendix B/C/F、Table 1-1/1-2/6 及 arXiv TeX 源包；PDF/HTML/TeX 视为同一来源。
[^src-graphwalks-dataset]: [[source-graphwalks-dataset]] —— OpenAI GraphWalks 数据集卡，<https://huggingface.co/datasets/openai/graphwalks>，访问 2026-09-13，快照 `downloads/graphwalks-dataset.md`。
[^src-mrcr-dataset]: [[source-mrcr-dataset]] —— OpenAI MRCR 数据集卡，<https://huggingface.co/datasets/openai/mrcr>，访问 2026-09-13，快照 `downloads/mrcr-dataset.md`。
[^src-nub-1m-benchmark]: [[source-nub-1m-benchmark]] —— xz-keg *Novel-Understanding-Bench* 仓库，<https://github.com/xz-keg/Novel-Understanding-Bench>，访问 2026-09-13，快照 `downloads/nub-1m-benchmark.md`。
