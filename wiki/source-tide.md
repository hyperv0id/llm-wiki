---
title: "TiDE: Time-series Dense Encoder for Long-term Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - mlp
  - covariates
  - long-term-forecasting
  - LTSF
  - google
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

## Summary

**TiDE (Time-series Dense Encoder)** is an MLP-based encoder–decoder for long-term time-series forecasting by Das, Kong, Leach, Mathur, Sen & Yu (Google Research / Google Cloud / UCSD; arXiv:2304.08424v5, 2024). Motivated by evidence that simple linear models can beat many Transformer LTSF designs, TiDE keeps linear-model simplicity and speed while adding residual MLPs that handle static/dynamic covariates and non-linear dependencies—without self-attention, recurrence, or convolution.[^src-tide]

## Core Arguments

**1. Dense residual MLP encoder–decoder.** TiDE is applied in a [[channel-independence|channel-independent]] manner with globally shared weights. Dynamic covariates at each time step are first projected by a residual block (feature projection) to width $\tilde r \ll r$. The dense encoder concatenates look-back $y_{1:L}$, projected past/future covariates $\tilde x_{1:L+H}$, and static attributes $a$, then maps them through residual MLP blocks. A dense decoder emits a horizon of decoded vectors that a per-step [[temporal-decoder|temporal decoder]] fuses with projected future covariates; a global linear residual from look-back to horizon keeps pure linear maps (as in [[ltsf-linear|DLinear]]) as a subclass of the model.[^src-tide]

**2. Linear analogue is near-optimal for LDS.** When residual paths dominate and encoding size is large enough, TiDE reduces to a linear map from context+covariates to the horizon. The paper proves that a short-window linear auto-regressive predictor achieves near-optimal error against LDS predictors when the transition matrix has spectral radius bounded away from 1; synthetic LDS experiments show Linear beating LSTM and Transformer.[^src-tide]

**3. Accuracy with 5–10× efficiency vs best Transformer.** On Weather, Traffic, Electricity, and four ETT datasets (horizons 96/192/336/720, look-back 720 for TiDE), TiDE matches or beats [[patchtst|PatchTST]], N-HiTS, and [[ltsf-linear|DLinear]], with >10% lower MSE than PatchTST on Traffic (horizon 720). Inference/training scale linearly in context length ($\tilde O(n_e h^2 + hL)$ encoder) vs quadratic attention in PatchTST; on Electricity, TiDE is roughly an order of magnitude faster and survives $L \ge 1440$ where PatchTST OOMs.[^src-tide]

## Experiments

- **LTSF benchmarks:** Standard 7:1:2 splits, MSE training, time-derived global covariates (minute/hour/day, etc.). TiDE, PatchTST, N-HiTS, and DLinear form a clear first tier over FEDformer/Autoformer/Informer/Pyraformer/LogTrans; residual ablations and context-size plots support residual paths and longer look-backs.[^src-tide]
- **M5 demand:** Static hierarchical attributes + promotions/events; TiDE WRMSSE $0.611 \pm 0.009$ vs DeepAR $0.789$ and PatchTST $0.976$ (no covariates). Date-only TiDE still beats both baselines.[^src-tide]
- **Temporal decoder ablation:** Semi-synthetic Electricity events show faster adaptation to future covariates and less post-event lag without the temporal highway.[^src-tide]
- **S4 comparison:** TiDE substantially outperforms S4 numbers reported in the S4 paper on Weather/ETT/Electricity long horizons.[^src-tide]

## Limitations

Transformers remain more parameter-efficient while being memory/compute heavier; the authors note this may matter for extremely large pretrained models. Theory covers linear analogues under LDS assumptions, not full residual MLPs with non-linearity. Evaluation focuses on standard multivariate LTSF + M5 rather than multimodal exogenous settings later studied by TimeXer/ExoST/KITE-class work.[^src-tide]

## Key Terminology

- **TiDE / Time-series Dense Encoder**: residual-MLP encoder–decoder for LTSF with covariates
- **Feature projection**: per-step residual dimensionality reduction of dynamic covariates
- **Temporal decoder**: per-horizon-step residual block fusing decoded vector with projected future covariates
- **Global linear residual**: look-back→horizon linear path that subsumes DLinear-style maps

## Related Pages

- Entity: [[tide]]
- Technique: [[temporal-decoder]]
- Concepts: [[lstf]], [[channel-independence]], [[direct-forecast]], [[instance-normalization]]
- Related models: [[ltsf-linear]], [[patchtst]], [[nbeatsx]], [[tft]]
- Early exogenous residual MLP: [[source-nbeatsx]] (NBEATSx; EPF short-horizon, interpretable basis stacks)
- Early interpretable multi-horizon + covariates: [[source-tft]] (TFT; static/known/observed + quantiles, 2020)

---

[^src-tide]: [[source-tide]]
