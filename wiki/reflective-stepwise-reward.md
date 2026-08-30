---
title: "Reflective Step-wise Rewards (反思式逐步奖励)"
type: technique
tags:
  - llm-agents
  - reinforcement-learning
  - credit-assignment
  - reward-design
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# Reflective Step-wise Rewards（反思式逐步奖励）

Reflective step-wise rewards 是 JitRL 提出的 credit assignment 技术：episode 结束后，用一个 LLM-based Evaluator 回看完整轨迹 $\tau=(s_1,a_1,\dots,s_T,a_T)$，为每一步动作打标量奖励，即学习映射 $\mathcal{E}:\tau\to\{r_t\}_{t=1}^T$，每个 $r_t$ 量化动作 $a_t$ 对任务整体成功的独立贡献；随后按标准折扣聚合为逐步回报 $G_t=\sum_{u=t}^T\gamma^{u-t}r_u$ 存入记忆[^src-jitrl]。

## 解决的问题

长轨迹下的 credit assignment：环境只给终点稀疏奖励（任务成败）时，无法区分一路上哪个动作真正有贡献。JitRL 不训练价值/奖励网络，而是利用 LLM 的自我反思能力在 episode 末尾离线回溯归因，让每条 $(s,a,G)$ 记忆条目携带"这一步值得重复/避免"的信号[^src-jitrl]。

## 具体设计（附录 G 的 prompt）

- **WebArena evaluator**：−3..+3 分档，评分必须同时考虑有用性与确定性（clearly useful & certain +3；might be useful & very uncertain +1；对 harmful 同理取负；中性 0），并输出 Result/Usefulness/Certainty/Score 的结构化结论。
- **Jericho evaluator**：按游戏自身奖励尺度校准分值，要求分析完整后果链而非即时结果，正分=值得重复，负分=浪费时间/造成循环。

## 在框架中的位置

逐步奖励是 [[non-parametric-policy-memory|非参数记忆]] 的数据来源：$G_t$ 的质量直接决定检索价值估计 $\hat V,\hat Q$ 与优势 $\hat A$ 的质量，进而决定 [[kl-regularized-policy-optimization|闭式 logit 更新]] 的方向。JitRL 论文在局限中明确对应关系：若 evaluator 错误归因，优势估计随之失准并可能劣化策略[^src-jitrl]。

## 与相邻方法的差异

与 Reflexion 的区别：Reflexion 的反思产物是**文本性总结**（prepend 到后续 prompt，见 [[in-context-learning]]），而本技术把反思产物**量化为标量奖励**并进入 RL 形式的回报聚合，使经验可被检索式价值估计消费。与 GRPO 的组相对归一（组内轨迹横向对比）不同，本技术是**纵向的步间归因**（同一条轨迹内跨时间步），见 [[grpo-for-forecasting]] 的 GRPO 机制页[^src-jitrl]。

## 相关页面

- [[jitrl]] — 提出该技术的方法
- [[non-parametric-policy-memory]] — 消费该奖励的记忆机制
- [[action-value-function]] — 回报/价值/优势的定义链
- [[source-jitrl]] — 源摘要

[^src-jitrl]: [[source-jitrl]]
