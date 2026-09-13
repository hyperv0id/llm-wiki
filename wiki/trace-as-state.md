---
title: "Trace as State：让前一遍推理参与下一遍阅读"
type: technique
tags:
  - llm
  - long-context
  - reasoning
  - inference-scaling
  - textual-state
created: 2026-09-13
last_updated: 2026-09-13
source_count: 1
confidence: medium
status: active
---

# Trace as State：让前一遍推理参与下一遍阅读

Xu Zou（Z.ai）与 Jie Tang（Tsinghua University）的论文 *Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers*，本次整理依据 arXiv:2609.02702v1（2026-09-02，preprint），完整阅读 21 页正文与附录 A–F。本文所述模型、接口与结果均按该版本的实验记录归因，不当作后续同名 API 的保证。[^src-trace-as-state]

> [!abstract] 核心问题与方法
> 读完长文本后才发现的任务信息，能否帮助模型重新理解之前读过的材料？论文把首轮推理轨迹作为可能有错的文本状态代理，放到原文前面，再启动一次新的 causal pass。其关键对照使用完全相同的轨迹，只把轨迹放在原文后面（§3.2–3.3）。[^src-trace-as-state]

## 阅读路径

- 本页：一个具体例子、方法机制、主要证据及适用边界。
- [[conditional-state-update|条件状态更新]]：状态定义、残差映射计数、两个 cut 的证明、最坏情况构造与定理范围。
- [[trace-as-state-evidence|实验与证据]]：Table 2 的全部主结果、Table 3 的全部对照、Table 5 的全部配对区间，以及 Figure 2–6。
- [[trace-as-state-reproduction|复现与成本]]：模型版本、序列化、原始提示文本、评分规则、全部 token 用量和公开材料缺口。
- [[trace-as-state-positioning|与相关工作的区别]]：Re2、CoRe、State over Tokens、ReContext 的原始论文对照。
- [[source-trace-as-state|来源摘要]]：论文著录、原文与下载位置。

## 1. 先遇到一个需要回看的问题

[INFERENCE] 以下是本页自建的教学例子，不是论文公布的模型输出。给模型一份按如下顺序排列的有向边表，最后问“从 A 出发，最短距离恰好为 2 的节点有哪些？”答案是 D、E。

```text
B -> D
C -> E
A -> B
A -> C

Operation: BFS from A, depth 2
Answer: D, E
```

[INFERENCE] 问题不在于边特别复杂，而在于信息的用途可能晚于信息本身出现。读到前两条边时，处理器还未从后两条边得知“A 的第一层邻居是 B、C”。当这条中间结论出现，之前的 `B -> D`、`C -> E` 才成为当前搜索前沿需要的出边。这对应论文所说的“输入顺序与推理所需状态出现的顺序不一致”（§1、§3.2）。[^src-trace-as-state]

必须把这个例子的边界讲清：causal attention **不是禁止后面的输出 token 访问前文**。后续推理仍可以利用前缀信息；受限制的是，后来得到的状态不能改变**同一遍**中较早位置已经形成的表示。论文检验的是让这些状态在新一遍处理原文时就可用，是否比仅在原文读完后提供它们更有帮助（§1、§3.3）。[^src-trace-as-state]

[INFERENCE] 假设第一遍留下这样的中间笔记：“第一层前沿是 B、C；已找到 D，还要核对 C 的出边。”这份笔记可能不完整，却已经指出重读时该关注哪些关系。Trace as State 的做法是将它放在边表前面，而不是只在原来的阅读顺序末尾继续接一句“再想想”。这个假设笔记仅用于说明状态代理的作用，不是忠实内部推理的证据。[^src-trace-as-state]

## 2. 实际操作：收集轨迹，再交换它与原文的位置

### 2.1 先区分四个对象

论文用 `x` 表示长上下文，`q` 表示最后的问题；每个首轮调用分别返回 reasoning trace `r_j` 和可见答案 `a_j`。序列化后的 `T` 来自 reasoning text，单独返回的 `a_j` 不加入 `T`（§3.2、§4.1、Appendix C.1）。[^src-trace-as-state]

| 条件 | 实验中的输入布局 | 新一遍读原文时是否已见到 `T` |
|---|---|---|
| First Pass | `[x,q]` | 没有轨迹 |
| Trace Append | `[x,T,q]` | 没有；轨迹在原文后面 |
| Trace as State | `[T,x,q]` | 有；轨迹在原文前面 |

表中 `q` 始终位于末尾。论文 §3 的简写 `[T,x]`、`[x,T]` 省略了实验里的末尾问题；GraphWalks 的具体边界是最后一个 `Operation:` block，不能随意把所有文字视作一个不可拆的 `x`（§4.1、Appendix C.3）。[^src-trace-as-state]

### 2.2 主实验不是“只调用两次模型”

作者对每题收集 5 次首轮调用的轨迹，按 repeat-index 顺序纳入同一个 `T`；两种放置条件复用相同的 `T`，各进行 5 次新的第二遍调用并报告平均得分。轨迹数消融才把纳入的首轮轨迹数从 1 改到 5；这些是同题首轮的多次采样，不是五轮串行自我改写（§4.1、§4.4）。[^src-trace-as-state]

```text
                 same problem [x,q]
                          |
                   first-pass runs
                          |
                r1, r2, r3, r4, r5
                          |
                   fixed serializer
                          |
                          T
                         / \
                        /   \
                [x,T,q]     [T,x,q]
                Append       As State
                fresh runs   fresh runs
```

两支的首轮轨迹、原文和模型相同，操纵变量是 `T` 相对原文的顺序；生成出的第二遍轨迹和输出长度不必相同。因此，这是输入材料匹配的位置对照，而非对运行时计算量严格逐 token 匹配的实验（§3.3、Appendix F）。[^src-trace-as-state]

### 2.3 序列化不是训练一个新的状态编码器

已评估实现保留 reasoning text，只增加固定说明、标签和分隔符；每条超过 50,000 **字符**的轨迹取前 50,000 字符，追加三个点，再进行首尾空白处理。`T` 还包含告知模型“轨迹可能有错、必须回到原文核对”的说明；它不是经过真值筛选的答案，也不是论文已经训练好的压缩模块（§4.1、Appendix C.1）。[^src-trace-as-state]

“可见答案字段不纳入”不等于“轨迹中没有答案线索”。[INFERENCE] reasoning text 仍可能写出候选节点、目标引用或候选答案；论文没有报告从轨迹正文中逐项删除所有答案内容。Answer Feedback 与 Trace Only 对照正是理解这类信息贡献的重要材料。[^src-trace-as-state]

### 2.4 新的一遍改变了什么

Trace as State 保留每一遍内部的 causal processing，也不更新被测模型权重；新增的是**两遍之间的文本反馈**。先前产生的 `T` 成为新一遍的前缀，因而能够参与后续原文 token 的处理。它没有回写上一遍已经计算好的表示，也没有把原模型变为 bidirectional attention（§3、§4.1）。[^src-trace-as-state]

![[wiki/assets/trace-as-state/trace_as_state_overview_v9.png]]

原论文 Figure 1：A 为因果状态更新；B 为条件先后顺序；C 明确将 `T` 标作 task state proxy；D 展示使用同一 `T` 的两种新遍输入。图由论文 TeX 包原始 PDF 转为 PNG，未改图内内容。[^src-trace-as-state]

## 3. 理解方法时需要保留的三个区分

### 任务条件与材料不是一回事

在教学例子里，“目前应检查 B、C 的出边”是任务状态，边表是待处理材料。论文的形式化问题把一个条件 `z` 当作初始状态，再让信息序列逐项更新它：条件先给时跟踪一条已确定的状态路径；条件后给时可能必须保留多种条件下的不同结果（§3.1）。[^src-trace-as-state]

### 文本状态代理与内部状态不是一回事

作者没有声称 reasoning trace 是内部状态的完整、忠实导出。`T` 可以不完整、有损或错误；论文只检验它是否含有对下一遍有用的信息。因此，“轨迹能影响答案”“轨迹有助于重读”“轨迹忠实解释内部计算”是不同命题；本论文的实验不能把三者直接等同（§3.2、§2）。[^src-trace-as-state]

### 保留上下文与让上下文重新条件化不是一回事

Trace Append 并非没有原文，也并非不能继续推理；它仍看见 `[x,T,q]`。但此时 `x` 的表示形成在 `T` 之前。Trace as State 把同一 `T` 移到前面，检验的是它在**原文被处理时**可用的额外价值，而不仅是末尾回答时“多看一些字”（§3.3）。[^src-trace-as-state]

## 4. 关键数字：先看增益来自哪一个比较

论文的三个基准家族包含 GraphWalks BFS/Parents、MRCRv2 256K/512K 与 NUB-1M Season 2 五个子设置。前四个子设置各有两个指标，NUB 有一个指标，所以每模型有 9 个指标格，三模型共 27 格；“26/27”是指标格的均值比较，不是 27 个独立数据集或 27 次显著性检验（Table 1-2、Table 2）。[^src-trace-as-state]

下表只摘取 GraphWalks 256K Parents 的 EM（百分数，5 次重复平均）；完整主表见 [[trace-as-state-evidence]]。最后一列为按 Table 2 相减得到的**百分点**。[^src-trace-as-state]

| 模型 | First Pass | Trace Append | Trace as State | As State − Append |
|---|---:|---:|---:|---:|
| DeepSeek V4 Pro Preview | 29.2 | 43.0 | 81.8 | +38.8 |
| Qwen 3.7 Max | 60.8 | 71.0 | 96.4 | +25.4 |
| GLM-5.2 | 66.4 | 83.2 | 100.0 | +16.8 |

相对 First Pass 的提升同时包含“取得多条首轮轨迹”和“额外读一遍”；As State 对 Append 的比较才控制了两者使用相同 `T`，更直接对应放置顺序。GLM-5.2 在这些 Parents 题上的 100.0% 也不能推广为任意图结构或任意上下文长度下都正确（Table 2、Appendix E）。[^src-trace-as-state]

另外三条限定必须与主结果一起读：

- 唯一均值反例是 **GLM-5.2 的 BFS F1**：As State 75.0，Append 75.8；同一模型的 BFS EM 则为 63.4 对 60.0。两个指标度量不同，不能删去不利的一格（Table 2）。[^src-trace-as-state]
- Appendix E 的配对表只包含 GraphWalks 与 MRCRv2 的 **24 格**；20 格的未校正 95% bootstrap 区间完全大于 0，4 格跨 0。不能把 26/27 的均值优势改写成 26/27 都已得到区间支持。[^src-trace-as-state]
- NUB-1M 上，As State 对 Append 的 accuracy 增量分别只有 **2、2、1 个百分点**，而且只有 20 个问题，每题 5 次重复；重复调用不能当作新增独立题目。作者把这部分描述为长小说阅读理解的支持性结果，配对 Table 5 未列这三格（Table 2、Appendix E）。[^src-trace-as-state]

## 5. 消融中更值得追问的现象

### 有原文，不一定就比只给轨迹更高

DeepSeek 的 GraphWalks 256K 对照中，Trace Only 的 BFS EM 为 46.2，而 Append 为 41.8；Parents EM 为 43.8，而 Append 为 43.0。As State 分别为 58.8、81.8（Table 3）。[^src-trace-as-state]

[INFERENCE] 这说明在该设置下，不能用“原文和轨迹都在窗口里”来保证更高得分。原文何时被处理，与原文是否存在，是两个需要分开测量的因素。不过该比较本身没有定位到某个 attention head 或具体内部状态的因果中介。[^src-trace-as-state]

### 第二遍不只是从五个完整答案里挑一个

在同一 DeepSeek GraphWalks 设置下，Oracle@5 对 BFS 与 Parents 的 EM 都是 50.0，As State 为 58.8 与 81.8。Oracle@5 是按真值事后选首轮五个答案里得分最高者；As State 还进行了新的计算，因此这不是“突破一切五次采样计算的上限”，而是超出了**仅选择已有完整答案**的结果（Table 3、Figure 6 caption）。[^src-trace-as-state]

[INFERENCE] 对教学例子而言，多份不完整笔记可能分别含有不同搜索线索，回到原文后仍有机会组合出新的完整答案；这个解释与数据相容，但不能代替真实轨迹级因果分析。[^src-trace-as-state]

### 错题轨迹会造成负效应

Random Trace 从同一 GraphWalks 子任务的其他问题抽轨迹，DeepSeek 的 BFS/Parents EM 降到 22.4/14.2，低于首轮的 31.6/29.2（Table 3）。作者据此认为通用的“像推理的文字格式”不足以解释收益，内容需要与当前问题相关。不能据此再推出“同题错误轨迹永远无害”：后者不是这个随机替换对照测量的命题。[^src-trace-as-state]

其余控制——Question First、论文采用的 Re2、Answer Feedback、Majority@5——以及轨迹数 1–5 的曲线，集中放在 [[trace-as-state-evidence]]，避免在主入口重复整张消融表。

## 6. 理论究竟承诺了什么

论文构造了一个有限状态任务族：对于**确定性、所有输入都精确、只读一遍**的处理器，条件先到只需保存当前状态；条件后到要在读条件前区分所有可能的残差映射。Appendix A 取长度 `n=1`，让每个输入字母代表一个状态空间到自身的函数，从而达到最坏情况的指数级差异。4 个状态时，两种 cut 分别需要 2 与 8 比特（§3.1、Appendix A）。[^src-trace-as-state]

这是一项关于特定计算模型和输入族的**存在性、最坏情况、cut-state 内存**结论。作者明确说形式任务不是现实长上下文任务的字面模型；它为位置干预提供定性动机，不给出当前 LLM 的显存节省比例、实际错误率或预期提升幅度（§3.2、Appendix A Scope）。[^src-trace-as-state]

[INFERENCE] 论文最直接的实证结论是“在所测设置中，同一文本状态代理前置通常取得更高分”；更强的“这些提升就是残差映射内存下界在真实模型中的表现”还需要额外证据。本文没有给出任务到形式状态空间的定量映射，也没有测量真实模型是否接近该下界。[^src-trace-as-state]

完整公式、构造、可达性、有限 KV cache 的口径以及不发生指数差距的教学反例，见 [[conditional-state-update]]。

## 7. 作者自述的边界与成本

1. **需要可用的状态接口。** 该实现依赖可读的 raw reasoning traces；只返回最终答案的模型/API 需要另一种接口，不能直接声称获得同样效果（Limitations）。[^src-trace-as-state]
2. **增加推理开销。** 首轮采样与新的长文本 pass 增加 latency 和 token cost，轨迹前置还可能减少多轮场景的 KV cache 复用；Appendix F 只给 provider-reported token 统计，不给本方法通用的美元或墙钟结论。[^src-trace-as-state]
3. **任务覆盖有限。** 只有三个模型与三个长上下文任务家族，没有多轮 agent 实验；其他领域、上下文长度和交互设置仍需检验（Limitations）。[^src-trace-as-state]
4. **状态选择与压缩属于扩展方向。** 作者提出筛选/压缩轨迹、学习更好的文本状态接口、优化反馈位置，以及联合学习模型和反馈接口；这些不是本文已完成的训练实验（§5）。[^src-trace-as-state]
5. **伦理声明。** 作者认为仅复用模型生成的轨迹不会引入超出底层模型与任务的新伦理或滥用风险（Limitations 末段）。[^src-trace-as-state] [INFERENCE] 这是一项作者声明，不能替代本文并未单独开展的安全性评估。[^src-trace-as-state]

## 8. 本页分析：可以怎样使用这个想法

[INFERENCE] 这篇论文提供了一个适合先做小规模对照的设计问题：**当模型已经形成与任务有关的中间信息时，让原始材料在这些信息之后再被处理，是否比只把信息留在材料末尾有效？** 它不要求先假设轨迹完全正确，也不要求先设计复杂的记忆架构。[^src-trace-as-state]

[INFERENCE] 严肃复现应先固定同题的 `T`，保持模型、原文、问题、序列化和失败处理一致，只交换位置；再分别测准确率、输出长度、缓存命中和成本。然后才检验“用一条轨迹是否足够”“压缩会丢什么”“错误中间结论是否被重复强化”。这是一组由本文实验设计导出的研究建议，不是论文已验证的优化方案。[^src-trace-as-state]

不要把推理文本前置到超出模型容量的位置，也不要把任务中的证据或候选结论当成不可质疑的指令；原论文自己的 preamble 明确要求回原文核对。本次整理没有调用商业模型复现实验，所得结论仍是对原论文证据的核查与分析。[^src-trace-as-state]

## 9. 与已有 wiki 的连接

按不同设计维度继续阅读：

- 窗口长度：[[context-window-extension]]、[[long-context-scaling-gap]]。
- 持久推理缓存：[[kv-cache-compression]]。
- 推理时上下文中的演示与反馈：[[in-context-learning]]。
- 前置信息的构造：[[prompt-as-prefix]]。
- 输入 token 的排列：[[structural-token-sorting]]。
- 奖励经验对输出分布的作用：[[non-parametric-policy-memory]]。

这些链接是 wiki 的组织方式，不表示上述方法与 Trace as State 已有组合实验或共同基准排名。

[^src-trace-as-state]: [[source-trace-as-state]]
