---
title: "KL-Regularized Policy Optimization (KL 约束策略优化)"
type: concept
tags:
  - reinforcement-learning
  - policy-optimization
  - kl-divergence
  - llm
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# KL-Regularized Policy Optimization（KL 约束策略优化）

KL 约束（正则化）策略优化指在最大化期望优势的同时，用 KL 散度惩罚把新策略拉住、不偏离参考策略的目标形式（JitRL 论文中的写法）[^src-jitrl]：

$$\pi^*=\arg\max_{\pi'}\ \mathbb{E}_{a\sim\pi'}\big[\hat A(s,a)\big]-\frac{1}{\beta}D_{\mathrm{KL}}(\pi'\|\pi_\theta)$$

温度参数 $\beta$ 控制约束强度。该目标的闭式解为**指数加权**形式：

$$\pi^*(a|s)\propto \pi_\theta(a|s)\exp\big(\beta\hat A(s,a)\big)$$

即参考策略与优势因子的乘积。落到 logit 空间（$\pi_\theta=\mathrm{Softmax}(z)$）就是加性规则 $z'(s,a)=z(s,a)+\beta\hat A(s,a)$，$\beta$ 每次对 logits 做的只是按优势平移。

## 闭式解的推导要点（JitRL 附录 B）

对目标加"分布归一"约束后写 Lagrangian，对 $\pi'(a)$ 求导置零得 $\log\pi'(a)=\beta\hat A(s,a)+\log\pi_\theta(a)+\text{const}$；常数项即配分函数 $Z$，指数化后得到上式。推导只用到目标是 $\pi'$ 的线性函数加熵正则这一结构，与具体优势估计器无关——优势可以来自 critic 网络，也可以来自 JitRL 的检索记忆[^src-jitrl]。

## 在 LLM agent 中的两处出现

- **梯度路线**：WebRL 的策略更新目标与上式同形（JitRL 论文附录 E 引其公式），KL 约束用于防止在线更新中的分布漂移，通过反向传播作用于参数；
- **免梯度路线**：JitRL 的 Theorem 4.1 证明同一目标的精确解可在推理时**无参数地**实现——把 $\hat A$ 加到 logits 上即完成一次"策略改进步"。其 Theorem 4.2/4.3 再补齐统计链条：在非平稳策略序列与 kNN 假设（$k\to\infty$、$k/N\to 0$、慢策略漂移 $\Delta_t\to 0$ 等）下，检索估计 $\hat V,\hat Q,\hat A$ 依概率收敛到当前策略的真值，误差分解为状态失配、策略漂移、方差三项；由连续映射定理，策略更新随之收敛到真优势诱导的 $\pi_t^*$[^src-jitrl]。

## 解读边界

闭式解的"最优"是**该目标函数**的最优：它最大化的是检索得到的估计优势 $\hat A$，而非环境真回报；估计质量由 Theorem 4.2 的渐近假设保证，样本有限或假设（如状态抽象的 Lipschitz 正则性）不成立时无此保证。这一点为论文自述的局限（LLM evaluator 归因错误会污染优势估计）提供了理论对应[^src-jitrl]。

## 相关页面

- [[jitrl]] — 免梯度实现（Theorem 4.1–4.3）
- [[test-time-policy-optimization]] — 所属问题类
- [[non-parametric-policy-memory]] — $\hat A$ 的非参数来源
- [[action-value-function]] — 优势 $A=Q-V$ 的定义
- [[grpo-for-forecasting]] — 梯度路线的组相对优势估计
- [[source-jitrl]] — 源摘要

[^src-jitrl]: [[source-jitrl]]
