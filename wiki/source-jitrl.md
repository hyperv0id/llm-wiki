---
title: "Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates"
type: source-summary
tags:
  - llm-agents
  - reinforcement-learning
  - test-time-learning
  - continual-learning
  - memory
  - icml-2026
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# JitRL（ICML 2026 版）

JitRL（Just-In-Time Reinforcement Learning）是新加坡国立大学 Bryan Hooi 组提出的免训练（training-free）LLM agent 持续学习框架，ICML 2026 接收（PMLR 306，论文首页印有 venue；arXiv v3, 2026-06-08）[^src-jitrl]。

问题：LLM agent 部署后权重冻结，无法从交互中继续学习；梯度 RL（PPO/GRPO/WebRL）成本高、易灾难性遗忘，ICL 类方法受上下文长度限制且缺乏 RL 的奖励驱动通用性[^src-jitrl]。

机制：JitRL 维护非参数记忆库 $\mathcal{M}=\{(s_i,a_i,G_i)\}$，存抽象状态、动作与折扣回报；episode 结束后由 LLM Evaluator 对整条轨迹打逐步奖励（reflective step-wise rewards）改善长轨迹的 credit assignment。推理时检索 top-k 相似转移，以邻域均值估计 $\hat V(s)$、动作匹配子集均值估计 $\hat Q(s,a)$，未见动作按概率 λ 给 optimism bonus $\alpha/|N(s)|$；优势 $\hat A=\hat Q-\hat V$ 以加性规则直接修改输出 logits：$z'(s,a)=z(s,a)+\beta\hat A(s,a)$，候选集扩充检索到的历史动作（基 logit 置 0）[^src-jitrl]。

理论：三条递进定理——加性 logit 更新是 KL 约束优势最大化的精确闭式解（Lagrangian 推导，$\pi^*\propto\pi_\theta e^{\beta A}$）；在非平稳策略序列下 $\hat V,\hat Q,\hat A$ 依概率收敛到真值（状态正则、噪声、kNN、慢漂移等假设，误差分解为状态失配+策略漂移+方差三项）；策略更新收敛到真优势诱导的 KL 正则最优策略[^src-jitrl]。

证据：WebArena 上 46.98/51.35（Avg/Final 成功率）超全部 training-free 基线（Static/Memory/Reflexion/AWM/EvoTest），held-out WebArena-Lite 上 60.0 对 WebRL 46.06（跨骨干：JitRL 用 Gemini-2.5-flash，WebRL 用 Llama-3.1-70B）；Jericho 三游戏全部第一；同骨干受控实验：8B on-the-fly 胜 WebRL（32.97 对 27.27），70B 离线记忆设定下 40.88 接近 WebRL 46.06 而成本 $200 对 $9,900（约 34 倍差距）；GRPO 调参后可逼近 JitRL 但样本消耗 64 倍。消融：logit 更新优于同检索的 prompt 更新；k∈[8,14] 稳健；文本状态表示优于稠密嵌入[^src-jitrl]。

局限（论文自述）：只能在基模型提出的候选集上重加权，无法发现基模型不会生成的动作；依赖 LLM evaluator 的逐步奖励质量；文本检索不适配空间推理等难以文本化的任务；记忆库存真实轨迹有隐私隐患[^src-jitrl]。

[^src-jitrl]: [[source-jitrl]]
