---
title: "Structural Token Sorting"
type: technique
tags:
  - llm-for-time-series
  - tokenization
  - attention
  - lead-lag-dependency
created: 2026-09-05
last_updated: 2026-09-13
source_count: 2
confidence: medium
status: active
---

# Structural Token Sorting

**Structural Token Sorting（STS）** 是 [[lagllm|LagLLM]] 提出的 token 重排机制：按 lead–lag 图上的重要性把 patch token 从"最 lead"到"最 lag"排序，把图结构转译为显式序列顺序，使 LLM 的自回归注意力天然先见 lead、后见 lag，无需隐式推断[^src-lagllm]。

## 机制

1. **初始化重要性**：节点空间中心度取空间结构矩阵的特征向量 $V_N$，经分配矩阵聚类成组中心度 $B_G=S\cdot\mathrm{MLP}(V_N)$；patch 时间位置 $T_p$ 嵌入为 $B_T$；广播相加后 $\mathrm{MLP}+\mathrm{softmax}$ 得初始分数 $B^{(0)}\in\mathbb{R}^{N_g\times N_p}$[^src-lagllm]。
2. **图上消息传递**：$B^{(l+1)}=\mathrm{MessagePassing}(A_L, B^{(l)})$，$L$ 层后每个 patch 的分数同时反映自身与上游（lead 方）加权影响[^src-lagllm]。
3. **重排**：$\hat{H}=\mathrm{Sorting}(H_{Token}, B^L)$，lead 在前、lag 在后[^src-lagllm]。

直觉：lead 更多、图上位置更中心的 token 排在前面，可以成为后续所有 token 的更丰富上下文，从而在自回归注意力中最大化其影响[^src-lagllm]。

## 与 [[sandglass-attention|STD-PLM 精炼注意力]] 的对比

两者都让 LLM 主干感知时空结构，但手段不同：STD-PLM 在 token 外围加 precoder/decoder 做注意力精炼；STS 不改注意力本身，而是改输入顺序——把结构信息编码进位置。论文消融：PEMS08 上去掉排序 MAE 13.21→13.34，仅按空间拓扑排序（ssort）为 13.28，按 lead–lag 图排序最优[^src-lagllm]。

## 与 RoPE/位置编码的关系

[[rope|RoPE]] 类方法在特征维度编码相对位置；STS 在序列轴上重排物理因果顺序。二者正交：前者告诉模型"相隔多远"，后者告诉模型"谁先谁后"。STUNet 的 [[query-aggregate-attention|Query-Aggregate Attention]] 用两套 RoPE 让传感器在邻接矩阵行列上定位上下游，是同一问题（让 LLM/Transformer 感知空间方向性）的另一条路线——改注意力偏置而非改序列顺序。

## 配套设计

- **Prefix tokens**：可学习前缀 $P\in\mathbb{R}^{M\times d_p}$（编码时间戳与原始值的状态/趋势），拼接在排序后 token 之前，作为全局聚合槽稳定表示——排序是动态的，前缀提供不变锚点[^src-lagllm]。

## 与长上下文轨迹前置的对照

[[trace-as-state|Trace as State]] 不重排图节点或时序 patch，而是把首轮推理轨迹整体放在长上下文之前，再启动一次新的 causal pass；论文用 [[conditional-state-update|条件状态更新]]说明“条件先到”和“条件后到”在某些确定性、精确、单遍任务上可有不同的内存要求（§3、Appendix A）。[^src-trace-as-state]

[INFERENCE] STS 与 Trace as State 的共同设计视角是调整前缀中已可用的信息，但排序对象与信息来源不同：STS 依据 lead–lag 图重排输入 token，Trace as State 依据同题前一遍的推理文本改变跨遍输入布局。物理 lead–lag 依赖与“推理后才发现的任务状态”不能混作同一个因果关系；本文也没有提供二者的共同基准或组合实验。[^src-lagllm][^src-trace-as-state]

[^src-lagllm]: [[source-lagllm-icml2026]]
[^src-trace-as-state]: [[source-trace-as-state]]
