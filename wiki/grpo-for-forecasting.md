---
title: "GRPO for Forecasting (ST-Vision-LLM)"
type: technique
tags:
  - reinforcement-learning
  - grpo
  - large-language-model
  - forecasting
created: 2026-06-08
last_updated: 2026-11-24
source_count: 2
confidence: medium
status: active
---

# GRPO for Forecasting (ST-Vision-LLM)

This page describes how [[st-vision-llm|ST-Vision-LLM]] applies **Group Relative Policy Optimization (GRPO)** as the second of its two training stages to directly optimize traffic-forecasting accuracy after supervised fine-tuning[^src-st-vision-llm]. Unlike supervised learning, which only imitates historical labels, the RL stage lets the model directly optimize prediction metrics[^src-st-vision-llm].

## Why GRPO

GRPO (introduced in DeepSeekMath) is a **memory-efficient, critic-free** variant of PPO[^src-st-vision-llm]. Standard PPO is actor-critic and needs a separate, computationally expensive critic/value network as the advantage baseline; GRPO foregoes the critic, making it a practical choice for fine-tuning large models[^src-st-vision-llm].

## Mechanism

For each input prompt (the visual context plus textual instruction), ST-Vision-LLM — acting as the **policy network** — samples a **group of G candidate output sequences**, each scored by the reward function[^src-st-vision-llm]. The baseline is computed directly from group performance (typically the average reward), and each sequence's **advantage is defined relative to this group average**; the policy is then updated to raise the likelihood of higher-than-average sequences[^src-st-vision-llm]. A **KL-divergence penalty against the frozen reference model** (the initial SFT-tuned model) keeps the policy from drifting too far from the well-behaved SFT solution[^src-st-vision-llm].

## Reward Function

The reward `R` blends an accuracy term with structural penalties[^src-st-vision-llm]:

$$ R = \exp\!\left(-\frac{\log 2}{x_h}\cdot E\right) \;-\; \frac{|L_{out}-L_{gt}|}{L_{gt}}\cdot 0.5 \;+\; \delta_{dec} $$

- `E` is the **Normalized Root Mean Square Error (NRMSE)** over the K prediction steps; the accuracy term `exp(−(log2/x_h)·E)` lies in [0,1][^src-st-vision-llm].
- `x_h` is the **half-score rate** hyperparameter: the reward equals 0.5 when `E = x_h`[^src-st-vision-llm].
- The **length-mismatch penalty** `−(|L_out−L_gt|/L_gt)·0.5` discourages outputs whose length deviates from the ground-truth sequence length[^src-st-vision-llm].
- `δ_dec = −0.5` on **decoding failure**, else 0 — penalizing structurally invalid outputs[^src-st-vision-llm].

Training stops when validation NRMSE ceases to decrease[^src-st-vision-llm].

## Effect

The ablation shows that removing the GRPO stage consistently raises MAE, RMSE, and NRMSE relative to the full model, confirming that this second-stage optimization improves accuracy on top of SFT[^src-st-vision-llm].

## Relation to Other GRPO Work

GRPO appears elsewhere in this wiki in different domains: [[flow-grpo|Flow-GRPO]] adapts online GRPO to **flow-matching image generation** (via ODE-to-SDE conversion), and [[streasoner|STReasoner]]'s **S-GRPO** is a spatial-aware GRPO for **multi-step spatio-temporal reasoning**[^src-st-vision-llm]. ST-Vision-LLM's use is distinct in that the reward is a regression-accuracy signal (NRMSE) for **numerical forecasting**, not a preference/aesthetic or reasoning-correctness reward[^src-st-vision-llm].

## 同槽位奖励对照：固定尺度 NRMSE vs 实例校准 ImpRatio

[[source-posttime|PostTime]]（NeurIPS 2026）同样采用「CoT-SFT 接 Stage-2 GRPO」两阶段框架，但在同一奖励函数槽位上与 [[st-vision-llm|ST-Vision-LLM]] 构成**同槽位互斥**对照[^src-posttime][^src-st-vision-llm]：

- **奖励结构差异**：ST-Vision-LLM 采用固定半衰尺度的指数 NRMSE 奖励 $\exp(-(\log 2 / x_h)\cdot E)$ 辅以长度与解码惩罚[^src-st-vision-llm]；PostTime 则以冻结 [[timesfm|TimesFM-2.5]] 先验误差 $B$ 为逐实例基准，构造截断线性改善比（ImpRatio）$R = \max\!\left(0,\, 1 - e_i / (2(B+\varepsilon))\right)$，一旦修订误差达到两倍先验误差即截断为零，避免绝对尺度跨序列波动[^src-posttime]。
- **实证归因分化**：ST-Vision-LLM 报告去掉 GRPO 会在 MAE/RMSE/NRMSE 上稳定退化[^src-st-vision-llm]；而 PostTime 的主精度增益集中在 Stage-1 `CoT-SFT` 轨迹筛选与 $0/5$ 保守回退配方，Stage-2 GRPO 仅将 OOD nMAE 从 $0.772$ 调至 $0.746$、ID 从 $0.739$ 调至 $0.738$，作者对 ImpRatio 的主要支撑来自训练动力学（提升组内奖励方差、将近零奖励组占比从 $28\%$ 降至 $4.7\%$）而非终态大幅涨点[^src-posttime]。

## Related Pages

- [[st-vision-llm]] — the model using this RL stage
- [[source-st-vision-llm]] — source summary
- [[source-posttime]] — 同槽位互斥对照：以先验校准 ImpRatio 替代固定尺度 NRMSE 奖励（NeurIPS 2026）
- [[flow-grpo]] — GRPO for flow-matching generation (vision)
- [[streasoner]] — S-GRPO for spatio-temporal reasoning
- [[direct-numerical-encoding]] — the encoding whose decode failures the reward penalizes

[^src-st-vision-llm]: [[source-st-vision-llm]]
[^src-posttime]: [[source-posttime]]
