---
title: "Non-Parametric Policy Memory (非参数策略记忆)"
type: concept
tags:
  - llm-agents
  - reinforcement-learning
  - memory
  - retrieval
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Non-Parametric Policy Memory（非参数策略记忆）

Non-parametric policy memory 是 JitRL 论文提出的关键概念定位：把 agent 的经验记忆当作**策略分布本身**而非 ICL 素材——记忆 $\mathcal{M}=\{(s_i,a_i,G_i)\}$ 隐式刻画环境的经验动态分布，检索邻域上的回报均值直接充当价值/优势估计，"软更新"作用在 LLM 的输出 logits 上，实现无参数更新的策略改进[^src-jitrl]。

## 与文本记忆方法的分界

论文在 related work 中梳理了既有记忆增强 agent（本页对其描述转引自该论文）[^src-jitrl]：

- MemGPT / Generative Agents：层级记忆存储历史交互；
- Voyager / Reflexion：检索过去技能或失败的文本描述以改进后续行为；
- A-MEM：动态索引构建互联知识网络。

这些方法的共同点：记忆检索的结果**以文本形式回到 prompt**，靠模型的上下文理解间接发挥作用。JitRL 的分界在于"Rather than merely retrieving text for in-context learning"——检索结果经价值估计转化为标量优势，直接修改 logits（式 10）。论文的消融支持这一分界的实际意义：同样的检索内容放入 prompt（Prompt Update）一致劣于直接改 logits（Logit Update），论文归因于长上下文中注意力难以稳定落在检索线索上。

## 概念结构

这一视角下，策略 $\pi(a|s)$ 被拆成两个因子（见 [[kl-regularized-policy-optimization]] 的闭式解）：

- **参数部分** $\pi_\theta$：冻结的基模型，提供语言先验与候选动作；
- **非参数部分** $\exp(\beta\hat A(s,a))$：由检索记忆在线估计的优势因子，承载所有部署后的学习。

学习因此完全外置：改进体现在记忆库的增长与更新（每条 episode 结束后由 [[reflective-stepwise-reward|逐步奖励]] 折算的 $(s,a,G)$ 三元组入库），模型参数不动，从机制上排除了参数更新型 RL 的灾难性遗忘问题[^src-jitrl]。

## 与参数化记忆蒸馏的对照

[[ts-memory]] / [[parametric-memory-distillation]]（KDD 2026，时序基础模型）代表相反的取舍：把在线检索知识**离线蒸馏进轻量参数模块**，换取推理时 O(1)、免外部数据库；非参数策略记忆则保留在线检索，换取经验持续累积与可解释的奖励归因，代价是每次决策都有检索开销（JitRL 报告约 15–47ms，相对 LLM 推理可忽略）[^src-jitrl]。

## 相关页面

- [[jitrl]] — 提出并实例化该概念的方法
- [[test-time-policy-optimization]] — 所属问题类
- [[kl-regularized-policy-optimization]] — 参数/非参数两因子的数学形式
- [[reflective-stepwise-reward]] — 记忆条目中回报的来源
- [[ts-memory]] / [[parametric-memory-distillation]] — 参数化记忆的对照路线
- [[in-context-learning]] — 文本记忆所属的 ICL 范式
- [[source-jitrl]] — 源摘要

[^src-jitrl]: [[source-jitrl]]
