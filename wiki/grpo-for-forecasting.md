---
title: "GRPO for Forecasting (ST-Vision-LLM)"
type: technique
tags:
  - reinforcement-learning
  - grpo
  - large-language-model
  - forecasting
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
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

## Related Pages

- [[st-vision-llm]] — the model using this RL stage
- [[source-st-vision-llm]] — source summary
- [[flow-grpo]] — GRPO for flow-matching generation (vision)
- [[streasoner]] — S-GRPO for spatio-temporal reasoning
- [[direct-numerical-encoding]] — the encoding whose decode failures the reward penalizes

[^src-st-vision-llm]: [[source-st-vision-llm]]
