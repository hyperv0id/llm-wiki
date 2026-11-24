---
title: "TiDE"
type: entity
tags:
  - time-series
  - forecasting
  - mlp
  - encoder-decoder
  - covariates
  - LTSF
  - google
  - channel-independence
created: 2026-07-13
last_updated: 2026-11-24
source_count: 4
confidence: medium
status: active
---

# TiDE (Time-series Dense Encoder)

**TiDE** is a residual-MLP encoder–decoder for long-term time-series forecasting proposed by Das et al. (Google Research / Google Cloud / UCSD; arXiv:2304.08424). It aims to keep the speed of linear LTSF models while modeling non-linearities and static/dynamic covariates, without self-attention, RNNs, or convolutions.[^src-tide]

## Problem Setting

Given look-back $y^{(i)}_{1:L}$, dynamic covariates $x^{(i)}_{1:L+H}$ known into the horizon, and static attributes $a^{(i)}$, the model predicts horizon $y^{(i)}_{L+1:L+H}$ per series under a global shared parameterization. Dynamic covariates may be global (calendar features) or series-specific (promotions); static attributes capture fixed series metadata.[^src-tide]

## Architecture

TiDE is applied in a [[channel-independence|channel-independent]] fashion: one series' history and covariates map to that series' multi-step forecast ([[direct-forecast|direct forecast]] / multi-horizon head). Residual blocks (one hidden ReLU layer, linear skip, dropout, layer norm) are the basic unit.[^src-tide]

```
Dynamic covariates x_t
  → Feature projection ResidualBlock → x̃_t (temporalWidth ≪ r)
Flatten look-back y_{1:L} + x̃_{1:L+H} + static a
  → Dense Encoder (n_e residual MLPs, hiddenSize)
  → Dense Decoder (n_d residual MLPs) → reshape to d_t ∈ ℝ^p for t=1..H
  → Temporal Decoder(d_t ; x̃_{L+t}) → ŷ_{L+t}
  + Global linear residual: Linear(y_{1:L}) → ℝ^H
```

### Encoding

1. **Feature projection.** Per-step residual map reduces covariates so flattening costs $(L+H)\tilde r$ rather than $(L+H)r$.[^src-tide]
2. **Dense encoder.** Concatenate projected past/future covariates, static attributes, and look-back; stack residual MLPs to embedding $e$.[^src-tide]

### Decoding

1. **Dense decoder.** Map $e$ to $g \in \mathbb{R}^{H \cdot p}$, reshape to per-horizon decoded vectors $d_t$.[^src-tide]
2. **[[temporal-decoder|Temporal decoder]].** Residual block of output size 1 combines $d_t$ with projected future covariates at $L+t$, forming a highway for strong contemporaneous covariate effects (e.g., holidays/promotions).[^src-tide]
3. **Global linear residual.** Linear look-back→horizon path ensures pure linear models such as [[ltsf-linear|DLinear]] remain a subclass of TiDE.[^src-tide]

## Theory

If residual paths dominate and encoding capacity is sufficient, TiDE's linear analogue is a finite-context linear map. Under LDS data with transition spectral radius $\gamma < 1$, a short-window linear AR predictor is competitive with the best LDS predictor (near-optimal excess risk with $k=\Theta(\log 1/\varepsilon)$). On synthetic LDS series, Linear beats LSTM and Transformer MSE.[^src-tide]

## Empirical Results

| Setting | Highlight |
|---------|-----------|
| LTSF (ETT/Weather/ECL/Traffic) | First-tier with PatchTST / N-HiTS / DLinear; Traffic H=720 ~10.6% better MSE than PatchTST |
| Efficiency vs PatchTST | ~5× inference, >10× training; linear scaling; PatchTST OOM for $L \ge 1440$ on Electricity timing setup |
| M5 demand | Full covariates WRMSSE 0.611 vs DeepAR 0.789 vs PatchTST 0.976 |
| Ablations | Temporal decoder speeds event adaptation; residual removal hurts Electricity H=96–336; longer context helps on Traffic |

Training uses mini-batch MSE with rolling (look-back, horizon) pairs; TiDE uses look-back 720 for all reported LTSF horizons. Time-derived covariates help TiDE even though they hurt pure linear models. Optional [[instance-normalization|RevIN]] is a tuned hyperparameter.[^src-tide]

## Historical Position

TiDE sits between the [[ltsf-linear|LTSF-Linear / DLinear]] challenge to Transformers and the [[patchtst|PatchTST]] revival of attention via patching+CI. Relative to DLinear it adds non-linear residual capacity and first-class covariates; relative to PatchTST it trades attention for linear-time dense maps and stronger covariate highways.[^src-tide] Earlier residual exogenous work [[nbeatsx|NBEATSx]] (EPF, basis stacks + TCN exo encoder) is a short-horizon interpretable precursor; [[tft|TFT]] (2020) is an earlier attention multi-horizon baseline with static/known/observed covariates and quantiles. Later exogenous-aware models (e.g., TimeXer, ExoTST, KITE, DAG) often still cite TiDE as a strong MLP/covariate baseline for long-horizon forecasting.

- **同槽位互斥对照（vs [[source-timeseed|TimeSeed]], ICML 2026）**：TiDE assumes complete look-back $y_{1:L}$ and projects covariates directly to multi-step horizons; under extreme endogenous sparsity (4% or single point with dense exogenous covariates), TimeSeed argues direct projection suffers scale and causality mismatch, replacing it with two-stage input-side history reconstruction via multi-scale upsampling + time/frequency exogenous linear aggregation before forecasting (though two-step imputation + PatchTST beats TimeSeed on ETTm1, 0.041 vs 0.068)[^src-timeseed].
- **复现口径与机制对照（vs [[source-xlinear|XLinear]], AAAI 2026）**：同属纯 MLP 协变量/外生预测阵营，TiDE 采用通道独立（CI）残差块并通过 temporal decoder 直连已知未来协变量；XLinear 则引入可学习全局 token 与时间/变量双 MLP 门控从历史外生变量跨通道注入信号，以 $O(Td)$ 线性开销在多数常规外生基准占优，但在 862 维 Traffic Many-to-Many 上因单全局 token 容量受限落后于 TimeXer（$0.432$ vs $0.428$）与 [[itransformer|iTransformer]][^src-xlinear]。
- **机制增量 $\Delta$（vs [[slide|SLiDE]], NeurIPS 2026）**：TiDE 将历史序列与展平后的过去/未来协变量一同送入 dense residual MLP 编码，再经 [[temporal-decoder|Temporal Decoder]] 逐预测步并入未来协变量；SLiDE 则将时序演化抽离为历史编码与未来滚动共享的分块线性递推 $W$，外生变量仅作为跨步共享 $\operatorname{MLP}_u$ 产生的纯加性推力进入潜状态，在 EPF-DE OOR 与跨驱动迁移上显著降低偏移误差，且推理速度接近 TSMixer[^src-tide][^src-slide]。

## Limitations

- No explicit cross-series / spatial graph modeling (CI-only).
- MLP parameter count can exceed Transformers at extreme scale despite better memory/compute scaling in $L$.
- Theory does not fully characterize non-linear residual stacks.

## Connections

- Paper: [[source-tide]]
- Technique: [[temporal-decoder]]
- Related models: [[ltsf-linear]], [[patchtst]], [[nbeatsx]], [[tft]], [[informer]], [[autoformer]], [[fedformer]], [[deepar]], [[source-timeseed]], [[source-xlinear]], [[slide]]
- Concepts: [[lstf]], [[channel-independence]], [[direct-forecast]], [[instance-normalization]]

---

[^src-tide]: [[source-tide]]
[^src-timeseed]: [[source-timeseed]]
[^src-xlinear]: [[source-xlinear]]
[^src-slide]: [[source-slide]]
