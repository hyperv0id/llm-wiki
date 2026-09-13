---
title: "Trace as State 的定位：重读、上下文重复、状态观与证据回放"
type: analysis
tags:
  - positioning
  - long-context
  - reasoning
  - re-reading
  - inference-scaling
created: 2026-09-13
last_updated: 2026-09-13
source_count: 5
confidence: medium
status: active
---

# Trace as State 的定位

本页基于四篇原始论文（Re2、CoRe、State over Tokens、ReContext）核查 [[trace-as-state]] §2 Related Work 的定位：它重复什么、是否携带已推理状态、是单 prompt 重读还是跨 pass 反馈、是否检索/压缩证据、是否需要训练或改架构。结论先行：Trace as State 的自变量不是"重复什么"而是"已推理状态放在第二次 pass 的什么位置"——它不宣称首创重读（Re2、CoRe 在先），也不宣称首创"trace 即状态"的观念（SoT 在先）；主论文把自身贡献定位为在保留每 pass 因果处理的前提下于 pass 之间加入反馈。[^src-trace-as-state]把"状态从 pass 尾部挪到 pass 头部"概括为主论文的核心位置干预，是本页对 §1/§3.3 与 Figure 1D 的归纳 [INFERENCE]。

范围声明：主论文 §2 还提到 Ok & Lee 的 prompt 顺序效应、Racing Thoughts 的 contextualization 错误、Markovian Thinker 的有界文本历史、Geiping/Saunshi/LoopFormer 的架构循环与 LLaDA 类 masked diffusion——这些本页未读原文，仅按主论文转述记录，不作独立核查。[^src-trace-as-state]

## 四篇原始工作按维度对照

| 维度 | Re2 (EMNLP 2024) | CoRe (Findings NAACL 2025) | State over Tokens (preprint 2025-12) | ReContext (preprint 2026-07) | Trace as State (arXiv:2609.02702) |
|---|---|---|---|---|---|
| 干预对象 | question（写两遍） | 完整 context（重复 k̂ 次） | 不干预输入；概念框架 | 从原 context 选出的句子级 evidence spans | 首轮 reasoning traces 序列化成的 T |
| 是否带已推理状态 | 否（无任何生成文本） | 否 | 概念上：reasoning prefix 就是状态，但只在单次生成内部 | 否（evidence 是原文复制，不是模型推理） | 是：T = π(r1..r5)，模型自己的首轮推理 |
| 单 prompt 还是跨 pass | 单 prompt 内两次出现 | 单 prompt 内 k̂ 次出现 | 不适用 | 同一最终 prompt 内多轮读、一轮生成 | 跨 pass：pass 1 产出 T，pass 2 读 [T, x, q] |
| 检索/压缩证据 | 无 | 无 | 无 | 有：内部 attention 相关性选 top-K、映射回句子 | 无：T 是生成的，不从 x 中检索 |
| 训练/架构改动 | 无（纯 prompting） | 无（chat template） | 无（不提出方法） | 无（白盒读 attention） | 无（冻结模型、固定 serializer） |

Re2、CoRe、ReContext 的机制描述以各自原文为准[^src-re-reading-improves-reasoning][^src-core-context-repetition][^src-recontext]；SoT 是约 7 页的概念论文，不含作者实验，其"reasoning tokens 是跨 stateless 计算周期唯一持久信息载体"的框架是其核心产出。[^src-state-over-tokens]Trace as State 的机制见主论文 §3.2–3.3。[^src-trace-as-state]

## 重点核查：主论文 Table 3 的 Re2 行与原始 Re2 的差别

原始 Re2（Xu et al., §2.2）的实现是：在一条 prompt 内把 question 写两遍，`Q: {Input Query}` / `Read the question again: {Input Query}`，后接思维激发句；重复对象只有 question，因为在其 14 个数据集（GSM8K、CSQA、ARC 等）上 question 就是全部任务输入，没有独立的长 context。其动机是 "bidirectional encoding"：第一遍为第二遍的 token 提供全局信息（LLaMA-2 注意力热图为证）。[^src-re-reading-improves-reasoning]

主论文 Table 3 的 Re2 行标注 prompt 为 [x, q, x, q]（DeepSeek V4 Pro Preview，GraphWalks 256K）：BFS EM 49.0 / F1 52.9，Parents EM 50.0 / F1 68.9，高于 first pass（BFS EM 31.6、Parents EM 29.2），低于 TRACE AS STATE（BFS EM 58.8、Parents EM 81.8）；§4.3 文字称 "it remains below TRACE AS STATE, especially on Parents"。[^src-trace-as-state]

两者的差别必须显式记录，不能默认为同一机制的复现：

1. 重复对象不同：原作重复 question（在其设置中即任务输入全长）；Table 3 行重复的是整个 256K 长上下文 x 加 question q。[^src-re-reading-improves-reasoning][^src-trace-as-state]把 [x, q] 整体视作任务输入并重复两遍，可视为对原作"重读任务输入"思想在长上下文上的一种适配解读 [INFERENCE]；但原作论文本身只在短输入上做过重读，未对长上下文任务做过这一决定。[^src-re-reading-improves-reasoning]
2. 任务与模型不同：原作是 GSM8K 等短上下文推理、davinci-003/ChatGPT/LLaMA-2[^src-re-reading-improves-reasoning]；Table 3 是 GraphWalks 256K、DeepSeek V4 Pro Preview。[^src-trace-as-state]
3. 实现文本不可追溯：主论文 Appendix C 只给出 T 的 serializer 与 [T, x]/[x, T] 的插入边界，没有给出 [x, q, x, q]、[q, x, q]、[a, x, q] 等消融条件的具体分隔符或引导语（例如原作的 "Read the question again:" 是否保留无从确认）。[^src-trace-as-state]
4. 结论级别：因此 Table 3 的 Re2 行是主论文作者的适配实现：它不是原作论文报告的数字，且由于具体引导语未在附录给出，无法核对它与原作机制是否等价 [INFERENCE]。[^src-trace-as-state][^src-re-reading-improves-reasoning]可核实的是：原作论文的 14 个数据集中未见长上下文设置 [INFERENCE]。[^src-re-reading-improves-reasoning]

主论文对此的归因是克制的：§2 只说 Re2 "repeats the question within one prompt and provides a direct rereading baseline without prior reasoning text"，§4.3 只说该行 "is consistent with T carrying task-relevant information beyond that supplied by prompt repetition alone"——对比对象是同一实现内的适配版，而非原作论文的实验。[^src-trace-as-state]

## 问题驱动的对照

**重复的是"问题"还是"上下文"还是"推理"？** Re2 重复 question[^src-re-reading-improves-reasoning]；CoRe 重复完整 context[^src-core-context-repetition]；ReContext 复制原 context 中被 attention 选中的句子[^src-recontext]；Trace as State 不在第二遍 prompt 里拼第二份输入文本——第二遍 pass 仍完整重读长上下文 x 与 question q（字面顺序 [T, x, q]），其注入物是首轮推理文本 T。[^src-trace-as-state]Question First 消融（[q, x, q]）显示单是 question 前置就能在 Parents 上从 EM 29.2 提到 53.0，主论文据此说 question 本身对某些任务是 vital task state——这把"重复什么"与"何时可见"两个因素部分解耦，但 Question First 与 Re2 行都不是完整解耦。[^src-trace-as-state]

**是否携带已推理状态？** 只有 Trace as State 的干预内容包含模型的推理输出。Re2/CoRe 的重复内容完全来自任务输入；ReContext 的 evidence 池由原文句子 materialize 而来（"copied from the original prompt, so the evidence pool is grounded rather than freely generated"）。[^src-recontext]SoT 在概念层面论证 reasoning prefix 就是模型的计算状态（"the sole persistent information carrier across the model's stateless generation cycles"），但它描述的是单次生成内部的状态持续性，没有跨 pass 复用机制。[^src-state-over-tokens]Trace as State 对 trace 的定位与 SoT 相容且更弱：它明确"不把 trace 等同于形式条件 z 或特权内部状态"，只是 imperfect textual proxy（可能不完整、有损、有错）。[^src-trace-as-state]主论文 §2 同时引 SoT 与 Paul et al. 2024 支撑"模型未必可靠使用其声明中间步骤"；按主论文参考文献列表，其 "Levy et al., 2025" 条目即 SoT preprint，而 SoT 本身是立场论文，此类实证证据在其文中是通过二手综述呈现的 [INFERENCE]——引用精度上应注意这一层。[^src-state-over-tokens][^src-trace-as-state]

**单 prompt 重读还是跨 pass 反馈？** Re2 与 CoRe 都在一条 prompt 内完成，差别只在同一内容出现两次或 k̂ 次；Trace as State 明确跨 pass：pass 1 生成 trace，pass 2 以 [T, x, q] 重新读上下文。位置对照 TRACE APPEND [x, T, q] 表明内容相同、仅 T 位置不同即产生 26/27 组合的差异（如 Parents EM 43.0 vs 81.8）。[^src-trace-as-state]ReContext 的最终生成 prompt 是 [C; ϕ(E); q]——其回放证据位于 context 之后，方向上与 TRACE APPEND 同侧；但 ReContext 递归读多轮、每轮 evidence 池扩大，结构上更复杂 [INFERENCE] 两者在"干预内容"（选出的原文证据 vs 生成的推理轨迹）与"干预位置"（append 侧 vs 前置侧）上同时不同，不能把 ReContext 的增益归因于位置。[^src-recontext][^src-trace-as-state]

**检索/压缩证据还是反馈生成状态？** ReContext 属于"从原输入中重选并回放"一类：不丢 context、不做外部检索系统、不改 attention logits、复用 KV cache。[^src-recontext]Trace as State 不做任何选择或压缩判断——serializer 原样保留推理文本并只加固定标签，甚至不做语义过滤，只做 50,000 字符截断。[^src-trace-as-state]

**需要训练或改架构吗？** 四者都不需要。Re2/CoRe 是纯 prompting；ReContext 是 training-free 的白盒 harness；Trace as State 明确"retaining the causal transformer structure"、冻结模型，与 §2 中架构循环一类需要改动训练/推理基础设施的方向相对。[^src-trace-as-state][^src-re-reading-improves-reasoning][^src-core-context-repetition][^src-recontext]

## 尚无 head-to-head 的边界

以下空缺在本批材料中无法填补，任何跨论文数值比较都缺乏共同口径：

- 主对照（Table 2）覆盖三个模型与三个数据集家族（GraphWalks 256K、MRCRv2 8-needle、NUB-1M），对照条件为 first pass、TRACE APPEND 与 TRACE AS STATE[^src-trace-as-state]；限定在 DeepSeek V4 Pro Preview × GraphWalks 256K 上的是 §4.3 上下文顺序消融（Question First、Re2 适配版、Answer Feedback、Random Trace、Trace Only、Majority@5/Oracle@5，Table 3）与 §4.4 trace 数量消融。[^src-trace-as-state]CoRe 与 ReContext 没有出现在主论文任何对照中。[^src-trace-as-state]
- CoRe 的证据在多跳 QA（HotpotQA、2WikiMultihopQA、MuSiQue）与合成任务上，量级为检索式短上下文[^src-core-context-repetition]；ReContext 的证据在 128K 的 8 个数据集上，骨干是 4B–8B 开源模型[^src-recontext]；Re2 的证据在短上下文推理基准上。[^src-re-reading-improves-reasoning]三者与主论文的 256K–1M frontier 模型设置之间不存在统一的模型、上下文长度、样本与指标口径（CoRe 与 ReContext 虽都用 HotpotQA，但 CoRe 用标准检索式多跳设定、ReContext 用 HELMET 128K 构造，设置并不相同）[^src-core-context-repetition][^src-recontext]；把 Re2 的 GSM8K 数字、CoRe 的 F1 提升、ReContext 的 128K 平均准确率与主论文的 GraphWalks 数字放在同一序列里排名是没有依据的 [INFERENCE]。
- SoT 不构成可比对象：它不提出方法、不做实验[^src-state-over-tokens]，任何"Trace as State 优于 SoT"的表述都是范畴错误；把两者关系概括为"SoT 提供状态观、Trace as State 将其操作化为跨 pass 的位置干预"是本页归纳 [INFERENCE]。[^src-state-over-tokens][^src-trace-as-state]
- 主论文自述的对比声明也止步于"同一实现内"：T 携带的信息超出 prompt repetition alone（Re2 行）、超出 answers alone（Answer Feedback 行）、超出 retrospective best-of-5（Oracle@5）。[^src-trace-as-state]

[INFERENCE] 综合而言：在"跨 pass、前置、携带已推理状态、免训练"这四个属性同时成立的干预里，本批材料中 Trace as State 是唯一实例；Re2/CoRe 缺"跨 pass 与已推理状态"，ReContext 缺"前置与已推理状态"。但这只是属性组合上的区分，不是效果排序——属性组合之间没有任何同设置对比。

## 本页相关页面

- [[trace-as-state]] — 主入口与机制
- [[conditional-state-update]] — §3.1/Appendix A 的条件状态更新数学定义
- [[trace-as-state-evidence]] — Table 2/3/5 与消融的完整数字
- [[source-trace-as-state]] — 主来源
- [[source-re-reading-improves-reasoning]]、[[source-core-context-repetition]]、[[source-state-over-tokens]]、[[source-recontext]] — 四篇原始论文摘要

[^src-trace-as-state]: [[source-trace-as-state]]
[^src-re-reading-improves-reasoning]: [[source-re-reading-improves-reasoning]]
[^src-core-context-repetition]: [[source-core-context-repetition]]
[^src-state-over-tokens]: [[source-state-over-tokens]]
[^src-recontext]: [[source-recontext]]
