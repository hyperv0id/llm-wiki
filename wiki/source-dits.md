---
title: "DiTS — Multimodal Diffusion Transformers Are Time Series Forecasters"
type: source-summary
tags:
  - time-series
  - diffusion
  - transformer
  - flow-matching
  - probabilistic-forecasting
  - arxiv-2026
  - tsinghua
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# DiTS: Multimodal Diffusion Transformers Are Time Series Forecasters

**Authors**: Haoran Zhang*, Haixuan Liu*, Yong Liu*, Yunzhong Qiu, Yuxuan Wang, Jianmin Wang, Mingsheng Long (Tsinghua University, School of Software, BNRist). arXiv:2602.06597, Feb 6 2026.

## Core Idea

DiTS frames **endogenous and exogenous variates as distinct modalities** in an MM-DiT-style dual-stream architecture, using **flow matching** for probabilistic time series forecasting[^src-dits]. Unlike prior DiT-based time series models (TimeDiT, LDT, Sundial) that compress covariates into scalar AdaLN modulation parameters, DiTS enables **fine-grained token-level interaction** between target and covariate streams via joint attention[^src-dits].

## Architecture

- **Dual-stream backbone**: Endogenous variate $x$ and exogenous variates $c$ flow as parallel streams through $L$ DiTS blocks, interacting via shared time attention, joint variate attention, and shared FFN[^src-dits].
- **Time Attention**: Shared self-attention over the temporal dimension, capturing intra-variate dependencies (similar to PatchTST's CI strategy)[^src-dits].
- **Variate Attention**: Joint attention where $x$ and $c$ project through independent QKV then attend jointly — capturing cross-variate correlations with per-stream output projections[^src-dits].
- **Adaptive Modulation**: A global conditioning embedding $Z_y = \text{Mean}(\text{VariateEmbed}(x_i)) + \text{Sinusoidal}(t)$ modulates every sub-layer via AdaLN scale/shift/gate parameters $\{\alpha_m, \beta_m, \gamma_m\}$[^src-dits].
- **Flow Matching**: Uses rectified flow with $x_t = (1-t)x_0 + t\epsilon$, training the velocity field $v_\theta(x_t, t, c)$ to predict $(\epsilon - x_0)$. Inference via Euler ODE solver, typically 5 steps[^src-dits].

## Key Results

- **FEV-Bench**: Avg WQL 0.070 (vs Chronos-2 0.074, Sundial-Base 0.119), avg MASE 0.601 (best among 11 models including zero-shot foundation models)[^src-dits].
- **EPF (deterministic)**: Avg MSE 0.274 (vs TimeXer 0.350, DAG 0.332), 10%+ improvement. Especially strong at long horizon (H=360): NP MSE 0.384 vs TimeXer 0.568[^src-dits].
- **EPF (future-agnostic)**: Avg MSE 0.285, still beats all baselines even without future covariate information[^src-dits].
- **Univariate LTSF**: Consistently beats PatchTST across 6 benchmarks (ETTh1/h2/m1/m2/ECL/Traffic)[^src-dits].

## Ablation Findings

- **Attention mechanism**: DiTS's joint variate attention > TimeXer-style > iTransformer-style > Timer-XL-style (flattening)[^src-dits].
- **Condition control**: DiTS (Joint + AdaLN) > Joint-only > Cross-attention-only > AdaLN-only. The MM-DiT-style concurrent use of both is essential[^src-dits].
- **Flow matching config**: Log-Normal noise scheduling > Cosine/Linear; **5 inference steps** optimal — more steps increase MSE while CRPS stays flat (metric misalignment with time series' low information density)[^src-dits].

## Limitations

- Code not public at time of writing.
- Preliminary work (not yet peer-reviewed conference publication).
- Single RTX 4090 experiments; scaling behavior to larger datasets unexplored.
- Univariate target only in FEV-Bench evaluation; multivariate target scenarios not tested.

[^src-dits]: [[source-dits]]
