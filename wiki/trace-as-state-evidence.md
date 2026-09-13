---
title: Trace as State 实验证据页：主结果、消融与不确定性
type: analysis
tags: [analysis, trace-as-state, long-context, evaluation, ablation, statistics]
created: 2026-09-13
last_updated: 2026-09-13
source_count: 1
confidence: medium
status: active
---

# Trace as State 实验证据页：主结果、消融与不确定性

本页是 [[trace-as-state]] 的证据切片，覆盖论文 §4.2–4.4、Appendix B/D/E，以及 Table 2、Table 3、Table 5 与 Figure 2–6。所有分数逐格照录自论文表格与图内标注；凡属我的解释、机制猜测或方法学边界，一律标 `[INFERENCE]`，与作者自述分开。机制与符号定义见 [[conditional-state-update]]，运行配置与可复现边界见 [[trace-as-state-reproduction]]，相关工作见 [[trace-as-state-positioning]]。全部数字来自论文 v1（arXiv:2609.02702v1，2026-09-02，21 页含附录 A–F）[^src-trace-as-state]。

## 1. 实验网格：3 个任务家族、实际 5 个子任务、GraphWalks/MRCRv2 每格 100 题、NUB 20 题

Table 1-2 给出数据集与判分口径：GraphWalks 256K 与 MRCRv2 8-needle 各有 **200 题**、NUB-1M Season 2 有 **20 题**，三者均为 **5 repeats**；判分分别为 GraphWalks 的 EM + set F1、MRCRv2 的 EM + SequenceMatcher ratio（论文记作 Seq.）、NUB-1M 的 model-judged accuracy（§4.1，Table 1-2）[^src-trace-as-state]。

Table 2 的列结构把 3 个家族拆成 **5 个子任务**与 **9 个指标列**：GraphWalks 256K 分 BFS、Parents（各含 EM 与 set F1 两列），MRCRv2 8-needle 分 256K、512K（各含 EM 与 Seq. 两列），加上 NUB-1M accuracy 一列 [^src-trace-as-state]。子任务的题量在 Appendix D 的分箱题数中可核对：BFS 分箱题数 $17+25+21+15+22=100$，Parents 分箱题数 $13+17+25+20+13+12=100$，因此 GraphWalks 的 200 题正好折半 [^src-trace-as-state]。

样本量口径需分清三处：Table 1-2 的 200/200/20 是题目数；Table 5 的置信区间以 **$n=100$ 每格**做 problem-cluster bootstrap，即 GraphWalks 的 BFS/Parents 各 100 题、MRCRv2 的 256K/512K 各 100 题 [^src-trace-as-state]；NUB-1M 是 **20 题 × 5 repeats**，论文将其表述为「20 problems over 5 repeats」的平均准确率（§4.1）[^src-trace-as-state]。`[INFERENCE]` 20 题 × 5 次不能当作 100 个独立问题来做区间估计；论文没有说明 Table 5 为何不纳入 NUB，只能观察到 Appendix E 指出 NUB 区间更宽、与其 20 题一致（Appendix E）[^src-trace-as-state]。

条件与字面顺序：baseline 单次 pass 为 $\mathcal M([x,q])$；两种 trace 回灌条件为 TaS $\mathcal M([T,x,q])$ 与 Trace Append $\mathcal M([x,T,q])$，二者共用同一 $T$，只有顺序不同，问题 $q$ 始终在结尾（§4.1、§3.3）[^src-trace-as-state]。主结果使用 5 条首轮 trace 拼成的 $T$，每条 trace 截断到前 50,000 字符（§4.1）[^src-trace-as-state]。失败记分口径：blocked、overlong、malformed、missing、nonterminal、content-filtered 一律记 0 且留在分母（§4.1）[^src-trace-as-state]。

## 2. 主结果 Table 2：27 个组合 × 3 个条件

Table 2 共 9 个指标列 × 3 个模型 = **27 个 model×task×metric 组合**，每个组合有 First Pass、Trace Append、TaS 三个条件值，全部为 5 repeats 平均后的百分数（Table 2）[^src-trace-as-state]。完整照录：

| 模型 | 条件 | GW BFS EM | GW BFS F1 | GW Par EM | GW Par F1 | MR 256K EM | MR 256K Seq. | MR 512K EM | MR 512K Seq. | NUB Acc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4 Pro Preview | First Pass | 31.6 | 36.4 | 29.2 | 46.5 | 53.8 | 78.5 | 45.4 | 63.6 | 60.0 |
| DeepSeek V4 Pro Preview | Trace Append | 41.8 | 48.3 | 43.0 | 65.3 | 66.6 | 79.6 | 49.4 | 66.9 | 71.0 |
| DeepSeek V4 Pro Preview | **Trace as State** | **58.8** | **65.9** | **81.8** | **91.3** | **76.8** | **88.7** | **52.6** | **73.3** | **73.0** |
| Qwen 3.7 Max | First Pass | 60.0 | 68.1 | 60.8 | 87.2 | 79.8 | 83.1 | 36.2 | 43.7 | 37.0 |
| Qwen 3.7 Max | Trace Append | 60.4 | 70.4 | 71.0 | 91.7 | 84.0 | 87.1 | 40.4 | 47.4 | 49.0 |
| Qwen 3.7 Max | **Trace as State** | **63.8** | **71.7** | **96.4** | **99.1** | **88.4** | **91.3** | **46.8** | **52.5** | **51.0** |
| GLM-5.2 | First Pass | 55.8 | 70.7 | 66.4 | 88.5 | 40.2 | 48.2 | 42.6 | 52.0 | 43.0 |
| GLM-5.2 | Trace Append | 60.0 | **75.8** | 83.2 | 92.7 | 40.0 | 58.3 | 44.6 | 59.9 | 65.0 |
| GLM-5.2 | **Trace as State** | **63.4** | 75.0 | **100.0** | **100.0** | **61.2** | **71.9** | **55.4** | **70.1** | **66.0** |

（Table 2，逐格照录；粗体为论文标注的「每个 model-metric 列内最优」，因此 GLM-5.2 的 GW BFS F1 列最优者是 Trace Append 75.8 而非 TaS 75.0 [^src-trace-as-state]。）

论文基于该表陈述的要点（§4.2）[^src-trace-as-state]：

- 27 个组合中 **TaS 高于 Trace Append 的为 26 个**；唯一反例是 GLM-5.2 的 GraphWalks BFS F1，Append 高 0.8 分（75.8 vs 75.0），而同一子任务的 EM 上 TaS 更高（63.4 vs 60.0）[^src-trace-as-state]。
- TaS **在所有报告的评估中都高于 first pass**；Append 在多数设置中也高于 first pass，说明 trace 文本本身携带有用信息 [^src-trace-as-state]。
- 收益最大处集中在 **GraphWalks Parents**：DeepSeek V4 Pro 的 F1 从 46.5 升到 91.3、EM 从 29.2 升到 81.8；Qwen 3.7 Max 的 F1 从 87.2 升到 99.1、EM 从 60.8 升到 96.4；GLM-5.2 在该子任务达到 100.0 EM 与 100.0 F1 [^src-trace-as-state]。

`[INFERENCE]` 两点需要与作者结论分开。第一，Append 并非在每格都优于 first pass：GLM-5.2 的 MRCRv2 256K EM 为 40.0，低于 first pass 的 40.2；「Append 提升多数设置」是描述性统计，不是逐格单调 [^src-trace-as-state]。第二，100.0 是 **100 题 5 次重复上的有限样本满分**，不能外推为「该模型在 Parents 上无条件正确」；同时 75.0 这类数值并非任何意义上的上限——同一列的 Trace Append 就是 75.8 [^src-trace-as-state]。

## 3. 消融 Table 3：10 个条件 × 4 个指标

Table 3 固定 DeepSeek V4 Pro Preview + GraphWalks 256K，比较 First Pass、Majority@5、Oracle@5、Question First $[q,x,q]$、Re2 $[x,q,x,q]$、Answer Feedback $[a,x,q]$、Random Trace $[T_{\mathrm{rand}},x,q]$、Trace Only $[T,q]$、Trace Append $[x,T,q]$、TaS $[T,x,q]$ 共 10 个条件（Table 3）[^src-trace-as-state]。下表逐格照录，并补一列各条件与 TaS 的差值（我计算的算术差，非论文数字）：

| 条件 | Prompt | BFS EM | BFS F1 | Par EM | Par F1 | Δ(TaS − 该条件) BFS EM / BFS F1 / Par EM / Par F1 |
| --- | --- | --- | --- | --- | --- | --- |
| First pass | $[x,q]$ | 31.6 | 36.4 | 29.2 | 46.5 | +27.2 / +29.5 / +52.6 / +44.8 |
| Majority@5 | — | 35.0 | 39.6 | 31.0 | 47.8 | +23.8 / +26.3 / +50.8 / +43.5 |
| Oracle@5 | — | 50.0 | 55.5 | 50.0 | 75.0 | +8.8 / +10.4 / +31.8 / +16.3 |
| Question First | $[q,x,q]$ | 33.4 | 36.8 | 53.0 | 64.6 | +25.4 / +29.1 / +28.8 / +26.7 |
| Re2 | $[x,q,x,q]$ | 49.0 | 52.9 | 50.0 | 68.9 | +9.8 / +13.0 / +31.8 / +22.4 |
| Answer Feedback | $[a,x,q]$ | 35.4 | 39.4 | 45.4 | 58.0 | +23.4 / +26.5 / +36.4 / +33.3 |
| Random Trace | $[T_{\mathrm{rand}},x,q]$ | 22.4 | 35.9 | 14.2 | 31.3 | +36.4 / +30.0 / +67.6 / +60.0 |
| Trace Only | $[T,q]$ | 46.2 | 52.7 | 43.8 | 67.3 | +12.6 / +13.2 / +38.0 / +24.0 |
| Trace Append | $[x,T,q]$ | 41.8 | 48.3 | 43.0 | 65.3 | +17.0 / +17.6 / +38.8 / +26.0 |
| **Trace as State** | $[T,x,q]$ | **58.8** | **65.9** | **81.8** | **91.3** | — |

（Table 3，逐格照录；差值列为我按表内数字计算的 [INFERENCE] 算术结果 [^src-trace-as-state]。）

### 3.1 各控制实际排除了什么

- **Random Trace**：trace 取自同一 GraphWalks 子任务的其它题目，四项指标为 22.4/35.9/14.2/31.3，**低于无 trace 的 first pass**（31.6/36.4/29.2/46.5），差距在 Parents 上达 −15.0 EM 与 −15.2 F1（Table 3）[^src-trace-as-state]。作者据此论证 TaS 的增益不是通用格式效应，且「类 trace 脚手架」本身在没有同题任务相关信息时不够（§4.3）[^src-trace-as-state]。`[INFERENCE]` 需要把两点分开：**实测结果**是随机 trace 条件下四项指标都低于 first pass；**「随机 trace 把错误任务状态前置、因此主动压低分数」只是一个未被论文验证的机制假说**，论文措辞停留在「不是通用格式效应」，没有给出压分的机制证据 [^src-trace-as-state]。
- **Question First $[q,x,q]$**：Parents 大幅改善（53.0/64.6，对比 first pass 29.2/46.5），BFS 仅与 first pass 相当（33.4/36.8）（§4.3，Table 3）[^src-trace-as-state]。作者的解释是「question 本身对某些任务可能是关键任务状态」——这是作者的机制解释，不是被证明的因果链 [^src-trace-as-state]。Table 3 中 Question First 是唯一附有记分资格说明的条件：Appendix B 说明对该条件，null 或未知终止标记被**保守地当作 length-limited**，实测记录里 **121 个 null 标记、0 个 unknown 标记、60 个原生标记为 length-limited**，另有 **1 个已停止的 BFS 输出缺少要求的终止答案语法**；**全部 182 个输出记 0**（Appendix B）[^src-trace-as-state]。`[INFERENCE]` 因此 182 个 0 分里，BFS 占 161 个（160 个 length-limited + 1 个格式错误）、Parents 占 21 个；但「121 个 null 标记」只是保守分类，**不能据此认定 181 个输出确实都是截断**，也不能把 182 个 0 分读作 182 个明确推理错误。该条件的得分同时混合了推理质量与输出资格（截断/格式），二者无法从表内拆开；因此不能推断其 BFS 表现「被截断压低」还是「被高估」。
- **Re2**：实测 prompt 格式为 $[x,q,x,q]$，论文在表格里同时标注了格式与引用（Table 3）[^src-trace-as-state]。`[INFERENCE]` 该条件可理解为论文把原始 Re2 的整体 query $[x,q]$ 视作一段长文本、再重复一次的长文适配实现；原始 Re2 论文没有报告本文这些任务与数字，论文也没有给出该条件所用的具体引导语，因此本行的结论只适用于论文实现的这个形式，不能断言它与原始方法的关系是「完整复刻」或「不是复刻」 [^src-trace-as-state]。作者自己的措辞是「In this evaluated setting, the comparison is consistent with $T$ carrying task-relevant information beyond that supplied by prompt repetition alone」，即明确限定在该实现内（§4.3）[^src-trace-as-state]。数值上 Re2 为 49.0/52.9/50.0/68.9，仍低于 TaS，Parents 上差距最大（+31.8 EM、+22.4 F1）[^src-trace-as-state]。
- **Answer Feedback $[a,x,q]$**：回灌的是首轮**可见答案** $a=[a_1,\ldots,a_{n_{\mathrm{tr}}}]$，不是 reasoning trace（§4.3，Table 3）[^src-trace-as-state]。`[INFERENCE]` 这是「答案信息」与「推理文本」的对照：答案只带来 35.4/39.4/45.4/58.0，Parents EM 上比 TaS 低 36.4 分，说明在该实现里答案文本不如序列化推理文本有用；但该对照**没有做等长或等 token 控制**，因此答案与 trace 在长度、信息量上的差异无法与内容类型（答案 vs 推理）的差异分开 [^src-trace-as-state]。
- **Trace Only $[T,q]$**：不带原始长 context，只有 trace 与问题；四项指标 46.2/52.7/43.8/67.3（Table 3）[^src-trace-as-state]。它高于 Trace Append 的全部四项（+4.4、+4.4、+0.8、+2.0），但仍远低于 TaS（+12.6、+13.2、+38.0、+24.0）。作者的解释是 trace 含答案相关信息、但原始输入仍有用（§4.3）[^src-trace-as-state]。`[INFERENCE]` 「Trace Only 在这四个指标上都不低于 Append」对该表而言是事实而非「有时」；它提示 Append 条件下把 $T$ 放在 $x$ 之后并没有比只给 $T$ 增加可见收益，但我无法从这张表区分这是「顺序无效」还是「$x$ 在该位置被忽略」。
- **Majority@5 / Oracle@5**：Oracle@5 取 5 次首轮输出中**回顾性最好的评测分数**，Majority@5 保留在至少 3 次 repeat 级预测集合中出现的答案元素（Appendix E 图注、§4.3）[^src-trace-as-state]。数值上 Majority@5 = 35.0/39.6/31.0/47.8，Oracle@5 = 50.0/55.5/50.0/75.0；TaS 四项均高于 Oracle@5，`[INFERENCE]` 这意味着第二次 pass 产生了 5 个首轮输出集合之外的答案，而不意味着 trace 承载的信息量超过 5 条 trace 的联合；Oracle@5 是对**这 5 个既有输出**的回顾性选择上界，不是「多花计算也超不过」的性能上界，因为继续采样或更强模型可以超过它。论文的表述是 TaS「outperforms Oracle@5, showing that the feedback pass improves on what can be obtained by retrospectively selecting the best of the five first pass outputs」（§4.3）[^src-trace-as-state]。
- **文本近因（recency）**：作者称「All above controls place a copy of the long context late in the prompt but remain below TaS」，即这一概括覆盖 $[q,x,q]$、$[x,q,x,q]$、$[a,x,q]$、$[T_{\mathrm{rand}},x,q]$ 四个条件——它们都把长 context 放在 prefix 之后、都仍低于 TaS，因此文本近因本身不足以解释 TaS 增益（§4.3）[^src-trace-as-state]。`[INFERENCE]` 需注意 Trace Only $[T,q]$ 是在该段之后才引入且不含 $x$，不属于作者这句概括的范围，因此不能用它反驳该概括 [^src-trace-as-state]。

### 3.2 消融没有排除什么

`[INFERENCE]` 以下均由表结构而非作者机制论述推出：

- **顺序与位置的混淆**：TaS 与 Append 共用同一 $T$，这一配对设计干净地隔离了「$T$ 在 $x$ 前 vs 后」；但 Trace Only $[T,q]$ 同样使用同一 $T$（只是移除了 $x$），因此「共用 $T$」本身不是 Append 独有的性质——Append 是与 TaS 同时保留同 $T$ 与同 $x$ 的主位置对照。该设计也没有隔离「$x$ 的绝对起始位置」「$q$ 与 $x$ 的距离」「注意力预算的分配」这些随顺序一起变化的因素 [^src-trace-as-state]。
- **长度与内容未分离**：Answer Feedback、Trace Only、Random Trace 与 TaS 的文本长度、格式差异都未被配对控制；在长度与内容上与被比较条件严格共享 $T$ 和 $x$ 的对照只有 Trace Append 一个 [^src-trace-as-state]。
- **单一模型单一家族的消融**：Table 3 全部消融只在 DeepSeek V4 Pro Preview + GraphWalks 256K 上完成（该家族含 BFS 与 Parents 两个子任务），其它模型与其它任务家族只有主结果（§4.3）[^src-trace-as-state]；因此「Random Trace 有害」等结论在表内未跨模型或跨任务家族复现 [^src-trace-as-state]。
- **截断与序列化选择**：复用的每条 trace 截到前 50,000 字符、固定序列化器加分隔符；这些选择适用于所有复用 trace 的条件（含同 $T$ 的 TaS/Append 对照与 Trace Only），非 trace 条件（First Pass、Majority@5、Oracle@5、Question First、Re2）不涉及 trace 截断 [^src-trace-as-state]。因此本表无法评估截断长度或序列化方式本身的影响 [^src-trace-as-state]。
- **NUB-1M 的判分主体**：NUB-1M 用 DeepSeek V4 Pro 作 judge（Appendix B）[^src-trace-as-state]；`[INFERENCE]` DeepSeek V4 Pro 同时是 NUB 解的生成者之一，判分隔离了 solver 身份与反馈顺序，但同族模型自评的潜在偏差不在本表可检验范围内。

## 4. 不确定性：Table 5 paired bootstrap 与 26/27、20/24 的区别

Table 5 报的是**严格同 $T$ 的配对比**：$\Delta$ = TaS − Trace Append，单位为百分点；先把 5 次 repeat 在题内平均，再对题目做 problem-cluster bootstrap，**20,000 次重采样、seed 0**，给出**未校正**的 95% percentile 区间，每格 $n=100$（Table 5 表注）[^src-trace-as-state]。NUB-1M 不在该表中。24 格全部照录（粗体为论文标注的「区间整体位于零以上」）：

| 模型 | 单元格 | Δ [95% CI] |
| --- | --- | --- |
| DeepSeek V4 Pro | GW BFS EM | +17.00 **[+11.00, +23.40]** |
| DeepSeek V4 Pro | GW BFS F1 | +17.63 **[+11.65, +23.97]** |
| DeepSeek V4 Pro | GW Par. EM | +38.80 **[+31.60, +46.20]** |
| DeepSeek V4 Pro | GW Par. F1 | +26.02 **[+20.52, +31.84]** |
| DeepSeek V4 Pro | MR 256K EM | +10.20 **[+4.80, +15.80]** |
| DeepSeek V4 Pro | MR 256K Seq. | +9.11 **[+4.77, +13.93]** |
| DeepSeek V4 Pro | MR 512K EM | +3.20 [−2.60, +9.40] |
| DeepSeek V4 Pro | MR 512K Seq. | +6.35 **[+1.69, +11.36]** |
| Qwen 3.7 Max | GW BFS EM | +3.40 **[+1.00, +6.40]** |
| Qwen 3.7 Max | GW BFS F1 | +1.29 [−0.56, +3.90] |
| Qwen 3.7 Max | GW Par. EM | +25.40 **[+17.60, +33.60]** |
| Qwen 3.7 Max | GW Par. F1 | +7.43 **[+4.63, +10.58]** |
| Qwen 3.7 Max | MR 256K EM | +4.40 **[+1.40, +8.00]** |
| Qwen 3.7 Max | MR 256K Seq. | +4.19 **[+1.41, +7.53]** |
| Qwen 3.7 Max | MR 512K EM | +6.40 **[+1.80, +11.40]** |
| Qwen 3.7 Max | MR 512K Seq. | +5.11 **[+0.67, +9.84]** |
| GLM-5.2 | GW BFS EM | +3.40 [−1.00, +7.80] |
| GLM-5.2 | GW BFS F1 | −0.83 [−3.94, +1.62] |
| GLM-5.2 | GW Par. EM | +16.80 **[+10.40, +23.80]** |
| GLM-5.2 | GW Par. F1 | +7.32 **[+3.93, +11.30]** |
| GLM-5.2 | MR 256K EM | +21.20 **[+14.40, +28.20]** |
| GLM-5.2 | MR 256K Seq. | +13.56 **[+8.33, +19.34]** |
| GLM-5.2 | MR 512K EM | +10.80 **[+3.40, +18.20]** |
| GLM-5.2 | MR 512K Seq. | +10.20 **[+4.65, +16.08]** |

（Table 5，逐格照录 [^src-trace-as-state]。）

Appendix E 明确说明：**24 格中有 20 格的区间完全位于零以上**，跨零的 4 格是 DeepSeek V4 Pro MRCRv2 512K EM、Qwen 3.7 Max GraphWalks BFS F1、GLM-5.2 GraphWalks BFS EM 与 F1（Appendix E）[^src-trace-as-state]。

三个计数**不能互换**，这是本页最容易出错的地方：

1. **26/27 是单元格均值的计数**，来自 Table 2 的 27 个 model×task×metric 组合（TaS 均值高于 Append 的组合数），不涉及区间 [^src-trace-as-state]。
2. **20/24 是配对区间整体的计数**，来自 Table 5 的 24 个同 $T$ 配对比（区间不跨零的格数），且排除 NUB [^src-trace-as-state]。26 与 20 的分母都不同：27 含 NUB 的 3 格与全部 9 个指标列，24 只含 GraphWalks 与 MRCRv2 的 8 格 × 3 模型。
3. **区间是未校正的**。Table 5 表注与 Appendix E 都称为 "unadjusted"；`[INFERENCE]` 因此这些 95% 区间可以对**单个格子**做名义上的点态判断（例如「该格的 Δ 为正」），但论文没有对 24 格做 familywise 或 FDR 校正，不能把「有 20 格不跨零」当作已控制多重比较后的结论，也不能把跨指标、跨模型的比较当成已校正面貌 [^src-trace-as-state]。
4. **配对设计只对同 $T$ 的 TaS/Append 对比成立**。Table 5 的 $\Delta$ 与 Table 3 的「TaS 减各控制」不是同类量：Table 3 的差值是我从单元格均值算出的算术差，没有配对区间 [^src-trace-as-state]。
5. **NUB 的 20 题**既不在 Table 5，也不能用 20×5 冒充 100 个独立题；Appendix E 只说 NUB 区间更宽、与该表 20 题一致（Appendix E）[^src-trace-as-state]。

### 4.1 正文一位小数与 Table 5 两位小数

- Table 2、Table 3 报告的是**一位小数**的百分数，Table 5 报告的是**两位小数**的 Δ（Table 2、Table 3、Table 5）[^src-trace-as-state]。
- 两者一般可对上：例如 DeepSeek V4 Pro GW Par. F1 的 Table 2 差为 $91.3-65.3=26.0$，Table 5 为 +26.02；Qwen 3.7 Max GW BFS EM 为 $63.8-60.4=3.4$，Table 5 为 +3.40；GLM-5.2 MR 256K EM 为 $61.2-40.0=21.2$，Table 5 为 +21.20。[INFERENCE] 这些是同一批未舍入数据的两种呈现 [^src-trace-as-state]。
- 但**差值不能反推**：GLM-5.2 GW BFS F1 的 Table 5 为 **−0.83**，而 Table 2 两位相减得 $75.0-75.8=-0.8$；同理 Qwen 3.7 Max GW BFS F1 的 Δ 为 **+1.29** 而表内相减得 1.3，DeepSeek V4 Pro MR 512K Seq. 的 Δ 为 **+6.35** 而表内相减得 6.4。`[INFERENCE]` 原因是 Table 5 在**未舍入的配对数据**上计算 Δ 并以两位小数呈现，而 Table 2 的单元格先各自舍入到一位：每个均值最多带 0.05 的舍入误差，两个均值相减可到 0.10，再叠加 Table 5 两位小数末位的舍入可到约 0.105；因此用 Table 2 反算 Δ 的偏差**上限约 0.1，而不是 0.05**，本例三处都落在这个范围内，不构成矛盾 [^src-trace-as-state]。

## 5. Figure 2：trace 数量消融（$n_{\mathrm{tr}}=0\ldots5$）

设计：只重跑 DeepSeek V4 Pro Preview + GraphWalks 256K，把复用的首轮 trace 数从 $n_{\mathrm{tr}}=1$ 变到 $n_{\mathrm{tr}}=5$；每个 $n_{\mathrm{tr}}$ 取**按 repeat 顺序的前 $n_{\mathrm{tr}}$ 条合格 trace**，因此相邻设置是嵌套的；对每个 $n_{\mathrm{tr}}\ge1$，TaS 与 Trace Append 收到**同一份实际 $T$**，且每个条件下每题再平均 5 次新做的第二次 pass；$n_{\mathrm{tr}}=0$ 点就是 first pass 的 5 次均值（§4.4）[^src-trace-as-state]。

![[wiki/assets/trace-as-state/graphwalk_trace_count_ablation.png]]

Figure 2 有两个 panel：A 为 BFS F1，B 为 Parents F1；阴影为 95% percentile 置信区间（Figure 2 图注）[^src-trace-as-state]。

**数据可得性说明**：该图的 PDF 只含 panel 标题、坐标刻度、$n_{\mathrm{tr}}$ 刻度与图例，**没有逐点数值标注**；arXiv 源包中也没有对应的中间点数据文件（源包 analysis/ 目录只有 paired bootstrap 的一张表）。因此本页对 Figure 2 **只作定性描述、不给出 1–5 的中间数值估计** [^src-trace-as-state]；两端锚点可用 Table 2 的已知值。

- $n_{\mathrm{tr}}=0$（first pass，5 repeats 均值，Table 2）：BFS F1 = 36.4，Parents F1 = 46.5 [^src-trace-as-state]。
- $n_{\mathrm{tr}}=5$（主结果口径，Table 2）：TaS 为 BFS F1 65.9 / Parents F1 91.3，Trace Append 为 48.3 / 65.3 [^src-trace-as-state]。
- 作者的定性结论：性能**一般随 trace 数增加而上升**，且 **TaS 在每一个 $n_{\mathrm{tr}}\ge1$ 都高于 Trace Append**；作者把 trace 数称为「inference-scaling parameter」，并强调该排序在 $n_{\mathrm{tr}}=1$ 到 $n_{\mathrm{tr}}=5$ 上持续（§4.4）[^src-trace-as-state]。
- `[INFERENCE]` 两个限定：作者写的是 generally，**不能据此声称曲线严格逐点单调**（图内无数值标注，本页也不给中间点数字）；且该消融只有一个模型、一个任务家族、一组 5 次新采样，因此它不是关于「更多 trace 必然更好」的普遍结论 [^src-trace-as-state]。

## 6. Figure 3/4：GraphWalks 难度分箱与事后黄区（Appendix D）

Appendix D 把 GraphWalks 的属性当**探索性诊断**：BFS 用请求深度 $d$，Parents 用 gold parent-set 大小 $k$。分箱与题数为：BFS 池化为 $1$–$2$、$3$–$4$、$5$–$6$、$7$–$8$、$9$–$10$，题数 $(17,25,21,15,22)$；Parents 池化为 $0$、$1$、$2$、$3$、$4$–$5$、$\ge6$，题数 $(13,17,25,20,13,12)$（Appendix D）[^src-trace-as-state]。对反馈时机 $p\in\{\text{pre},\text{post}\}$ 与分箱 $b$，报告相对公共 first pass 基线的增益

$$\Delta_p(b)=\operatorname{Score}_p(b)-\operatorname{Score}_{\text{first}}(b).$$

每个点先在题内平均 5 次存储的 repeat，再在分箱内对题平均（Appendix D）[^src-trace-as-state]。

作者明确声明：**分析与高亮区间是在看过结果之后开发的**，属于 hypothesis-generating，不是关于交互效应、threshold、capacity limit 或 mechanism 的确证性检验；黄色区域是描述性、outcome-informed 的，边界按模型、子任务、指标各自选定，**不能支持确证推断**（Appendix D）[^src-trace-as-state]。

![[wiki/assets/trace-as-state/graphwalk_em_delta_by_difficulty.png]]

Figure 3（Appendix D 第一张图）给 exact-match 增益：左列 BFS 深度 $d$，右列 gold parent 数 $k$；蓝/橙曲线分别为 $\Delta_{\mathrm{pre}}$（TaS 相对 first pass）与 $\Delta_{\mathrm{post}}$（Append 相对 first pass）（Figure 3 图注）[^src-trace-as-state]。

![[wiki/assets/trace-as-state/graphwalk_bfs_f1_delta_by_model.png]]

![[wiki/assets/trace-as-state/graphwalk_parents_f1_delta_by_model.png]]

Figure 4（Appendix D 第二张图）是同样分箱下的 set F1 增益：上排 BFS、下排 Parents，黄区同样按模型、子任务、指标分别选定（Figure 4 图注）[^src-trace-as-state]。

以下两表照录 Figure 3 与 Figure 4 的图内数值标注，每格分别列出 $\Delta_{\mathrm{pre}}$ 与 $\Delta_{\mathrm{post}}$，不从像素位置估计数值。[^src-trace-as-state] 本次核对确认，132 个带符号数据标签与 PDF 文本抽取一致；[INFERENCE] 再以 Appendix D 的题数加权还原 12 个模型—子任务—指标剖面，两种条件各自相对 Table 2 总体增益的最大绝对差为 **0.05 个百分点**，与图表分别舍入的精度相容。例如 DeepSeek BFS EM 回算为 +27.214/+10.207，而 Table 2 相减为 +27.2/+10.2。[^src-trace-as-state]

BFS（$d$ 分箱，题数 $17,25,21,15,22$）：

| Panel | 指标 | 1–2 | 3–4 | 5–6 | 7–8 | 9–10 |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4 Pro | EM | +16.5 / +11.8 | +48.8 / +11.2 | +30.5 / +6.7 | +12.0 / +8.0 | +18.2 / +12.7 |
| DeepSeek V4 Pro | F1 | +9.2 / +6.8 | +45.1 / +14.6 | +43.9 / +11.5 | +23.0 / +12.5 | +18.2 / +12.7 |
| Qwen 3.7 Max | EM | +1.2 / −1.2 | +8.0 / +0.8 | +4.8 / +0.0 | +2.7 / +1.3 | +0.9 / +0.9 |
| Qwen 3.7 Max | F1 | +0.2 / −0.2 | +1.0 / +0.7 | +14.3 / +8.7 | +0.7 / +0.9 | +0.9 / +0.9 |
| GLM-5.2 | EM | +7.1 / +1.2 | +9.6 / +4.8 | +6.7 / +0.0 | +12.0 / +6.7 | +3.6 / +8.2 |
| GLM-5.2 | F1 | +1.6 / +0.4 | +0.9 / +0.9 | +7.7 / +6.5 | +7.4 / +4.3 | +4.9 / +12.8 |

Parents（$k$ 分箱，题数 $13,17,25,20,13,12$）：

| Panel | 指标 | 0 | 1 | 2 | 3 | 4–5 | ≥6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4 Pro | EM | +21.5 / +15.4 | +32.9 / +12.9 | +64.0 / +8.8 | +61.0 / +14.0 | +58.5 / +21.5 | +70.0 / +15.0 |
| DeepSeek V4 Pro | F1 | +21.5 / +15.4 | +32.9 / +12.9 | +55.5 / +25.3 | +54.0 / +23.5 | +42.4 / +13.6 | +51.9 / +15.0 |
| Qwen 3.7 Max | EM | +1.5 / +1.5 | +11.8 / +1.2 | +32.0 / +7.2 | +47.0 / +17.0 | +56.9 / +20.0 | +71.7 / +16.7 |
| Qwen 3.7 Max | F1 | +1.5 / +1.5 | +11.0 / +3.5 | +10.8 / +2.4 | +16.8 / +7.8 | +11.7 / +5.6 | +19.2 / +7.0 |
| GLM-5.2 | EM | +13.8 / −13.8 | +10.6 / +0.0 | +28.0 / +22.4 | +36.0 / +31.0 | +50.8 / +12.3 | +76.7 / +43.3 |
| GLM-5.2 | F1 | +13.8 / −13.8 | +9.4 / +0.8 | +10.1 / +8.1 | +9.8 / +8.8 | +10.0 / +5.0 | +19.5 / +11.9 |

（每格为 $\Delta_{\mathrm{pre}}/\Delta_{\mathrm{post}}$，单位百分点，照录 Figure 3、Figure 4 的图内标注 [^src-trace-as-state]。）

作者从分箱图中读出的现象（Appendix D）[^src-trace-as-state]：

- DeepSeek V4 Pro 的**最大 BFS 时机差集中在 $d=3$–$6$**；Qwen 3.7 Max 在同一区间有较小的集中。
- GLM-5.2 在 **$d\le8$ 保持正的间隔，并在最后一个池化箱出现反转**。
- **Parents 的剖面不同**：在绘制的 exact-match 分箱里 **TaS 从不低于 Append**，只有 Qwen 3.7 Max 在 $k=0$ 处打平；Qwen 的分离度随 $k$ 增大；DeepSeek V4 Pro 与 GLM-5.2 非单调，但在部分中段/高 $k$ 箱有大间隔。
- 作者同时声明这些剖面只记录异质性，**不解释原因**，也不证明深度或 parent-set 大小对 trace 复用有 mediation 作用；确证性后续需要先定分箱/连续对比、再在留出题目上评估（Appendix D）[^src-trace-as-state]。

`[INFERENCE]` 与上表一致、且论文文字未展开的四处细节：

- **GLM-5.2 的高深度反转在 F1 与 EM 上都出现**：BFS $9$–$10$ 箱中 EM 为 $\Delta_{\mathrm{pre}}=+3.6$ 对 $\Delta_{\mathrm{post}}=+8.2$，F1 为 $+4.9$ 对 $+12.8$；这是 GLM-5.2 在 BFS 上 TaS 低于 Append 的现象在最深池化箱的集中体现，与 Table 2 中该模型 BFS F1 唯一反例同向 [^src-trace-as-state]。
- **Parents 的负值只出现在 GLM-5.2 的 $k=0$ 箱**：EM 与 F1 均为 $\Delta_{\mathrm{post}}=-13.8$，即 Append 在该箱低于 first pass，同时 TaS 为 $+13.8$；该箱题数只有 13，且这一负值来自图内标注而非可做区间估计的配对表，不能当作稳定结论 [^src-trace-as-state]。
- **Qwen 3.7 Max 在 BFS 最浅箱也有负值**：EM 的 $\Delta_{\mathrm{post}}=-1.2$、F1 的 $\Delta_{\mathrm{post}}=-0.2$；「Append 通常不低于 first pass」这类粗描述本身允许例外，此处即为例外，不是与论文结论的冲突 [^src-trace-as-state]。
- **Parents 的 EM 表中 TaS 从不低于 Append，与 Table 5 的 Parents 区间全部高于零一致**；但 BFS 上两边都出现过穿插，这与 Table 5 中 4 个跨零区间有 3 个落在 BFS 类指标上相符 [^src-trace-as-state]。

## 7. Figure 5/6：单元格级不确定性与控制消融区间（Appendix E）

Appendix E 除 Table 5 的配对区间外，还用同一套 problem-cluster bootstrap 给出单元格级区间：Figure 5 覆盖 Table 2 中严格同 $T$ 的 GraphWalks 与 MRCRv2 比较，并附 NUB-1M 支持性结果；Figure 6 覆盖 GraphWalks 控制消融（Appendix E）[^src-trace-as-state]。两点抽样说明是 Figure 5 与 Figure 6 的图注共同给出的：**各条件的点是均值、条是题内先平均 5 次后的 95% percentile 区间**；Figure 6 的区间来自 **20,000 次 problem-cluster 重采样（seed 0）**；Majority@5 保留在至少 3 次 repeat 级预测集合中出现的答案元素，Oracle@5 取 5 个输出中回顾性最高的评测分数，其它条件在题内平均 5 次计划重复并对缺失或非法输出记 0（Appendix E）[^src-trace-as-state]。

![[wiki/assets/trace-as-state/graphwalks-em.png]]

![[wiki/assets/trace-as-state/graphwalks-f1.png]]

![[wiki/assets/trace-as-state/mrcr-em.png]]

![[wiki/assets/trace-as-state/mrcr-sequence-matcher.png]]

![[wiki/assets/trace-as-state/nub-accuracy.png]]

![[wiki/assets/trace-as-state/graphwalk-controls-exact-match.png]]

![[wiki/assets/trace-as-state/graphwalk-controls-set-f1.png]]

以上图片依次为 Figure 5 的五个 panel（GraphWalks EM、GraphWalks set F1、MRCRv2 EM、MRCRv2 SequenceMatcher、NUB-1M accuracy）与 Figure 6 的两个 panel（控制消融的 EM 与 set F1）[^src-trace-as-state]。论文对这些图的可核查陈述只有两点：NUB-1M 的区间更宽，与其 20 道评测题一致；trace-count 阴影使用同一 bootstrap 过程（Appendix E）[^src-trace-as-state]。

`[INFERENCE]` 这些图不携带逐点数值标注，本页不从中读取数值，只使用它们的方法学信息：Figure 5/6 的区间与 Table 5 同源但对象不同——Table 5 是 **TaS − Append 的配对差**，Figure 5 是**各条件的单元格水平**，Figure 6 是**控制条件的单元格水平**。因此 Figure 5/6 的边际区间**不能替代** Table 5 的配对差区间，也不构成对多重比较的处理：两张图里的区间都是未校正的单条件区间。具体到读图，(a) 两个条件区间不重叠，只说明这一点态层面存在差异线索，仍不是经多重比较校正的显著性结论；(b) 区间重叠**不等于**两条件无差异——边际区间重叠与配对差是否为零是两件事，尤其在同一题集上高度相关的两个条件（如 TaS 与 Append）完全可能区间重叠而配对差稳定为正 [^src-trace-as-state]。

## 8. 作者自述与本页证据的强弱

| 内容 | 性质 | 依据 |
| --- | --- | --- |
| Table 2/3/5 的分数、$\Delta$、区间端点、分箱题数、$n_{\mathrm{tr}}=0$ 与 $n_{\mathrm{tr}}=5$ 端点值 | 论文表格中的可核查数字，本页逐格照录 | Table 2、Table 3、Table 5、Appendix D [^src-trace-as-state] |
| Figure 3/4 的分箱增益值 | 132 个数据标签与 PDF 抽取一致；按题数权重回算 12 个剖面，相对 Table 2 增益的最大绝对差 0.05 个百分点 | Figure 3、Figure 4、Appendix D；本次核算 [INFERENCE] [^src-trace-as-state] |
| Figure 2 的中间点（$n_{\mathrm{tr}}=1\ldots4$）数值 | **不可得**：图内无数值标注，源包无对应数据文件，本页不给估计值 | Figure 2、源包 analysis/ 目录 [^src-trace-as-state] |
| 「TaS 在 26/27 组合上优于 Append」「20/24 区间在零以上」 | 论文的计数性陈述，与表一致 | §4.2、Appendix E [^src-trace-as-state] |
| 「trace 作为不完美文本任务状态代理，放在 context 之前更有利」 | **作者的解释性假设**，论文写的是 consistent with the hypothesis，非被证明的机制 | §4.2 [^src-trace-as-state] |
| 「Random Trace 说明增益不是通用格式效应」 | 作者从表中得出的排除性论证；实验仅覆盖 DeepSeek 与 GraphWalks 这一任务家族 | §4.3、Table 3 [^src-trace-as-state] [INFERENCE] |
| 「TaS 超过 Oracle@5，说明第二次 pass 优于回顾性挑最好的首轮输出」 | 作者陈述 + 我的边界说明：Oracle@5 是这 5 个输出的选择上界，不是计算上界 | §4.3 [^src-trace-as-state] [INFERENCE] |
| 「yellow 区与深度/parent 数的关系」 | 作者明确标为事后、hypothesis-generating，不支持 threshold/capacity/mediation 论断 | Appendix D [^src-trace-as-state] |
| 「Append 多数设置优于 first pass」 | 作者概述；表中存在 GLM-5.2 MR 256K EM 的 −0.2 反例 | §4.2、Table 2 [^src-trace-as-state] [INFERENCE] |
| 「NUB-1M 提供同方向的支持性证据」 | 作者陈述；NUB 仅有 20 题、5 次重复，且不在 Table 5 配对表中 | §4.2、Appendix E [^src-trace-as-state] [INFERENCE] |

本页不涉及论文的商业、算力或复现可行性结论；成本与 token 统计见 [[trace-as-state-reproduction]]，形式化与最坏情况构造见 [[conditional-state-update]]。

[^src-trace-as-state]: [[source-trace-as-state]]
