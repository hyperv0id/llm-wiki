---
title: "Direct Forecast"
type: concept
tags:
  - time-series-forecasting
  - multi-step-forecast
  - learning-objective
created: 2026-07-13
last_updated: 2026-11-24
source_count: 7
confidence: high
status: active
---

## Definition

**Direct forecast (DF)** is a multi-step forecasting paradigm that maps a history window $L \in \mathbb{R}^{L \times D}$ to the entire future window $Y \in \mathbb{R}^{T \times D}$ in one forward pass via a multi-output head $\hat Y = g(L)$, then optimizes a step-wise loss (typically MSE) over all horizon steps jointly.[^src-fredf]

## Contrast with Iterative Forecast (IF)

| | Direct forecast (DF) | Iterative forecast (IF) |
|--|----------------------|-------------------------|
| Generation | All $T$ steps in parallel | One step at a time, feeding predictions back |
| Label structure | Implicitly assumes step independence under MSE | Naturally respects sequential dependence |
| Failure mode | Ignores [[label-autocorrelation]] → biased likelihood training | Error propagation / high variance on long horizons |
| Modern use | Dominant since Informer; used by TimesNet, PatchTST, iTransformer, [[tide|TiDE]][^src-tide]; early multi-horizon quantile exemplar [[tft|TFT]] | Early RNNs / DeepAR-style models |

DF became dominant for long-horizon tasks because of faster inference, simpler implementation, and better empirical accuracy than IF under error accumulation. Earlier multi-horizon direct models such as [[tft|TFT]] and MQRNN already forecast all horizons jointly with quantile heads before the LTSF Transformer wave.[^src-fredf][^src-tft]

## Likelihood Gap

FreDF's Theorem 3.1: the DF MSE objective equals the conditional negative log-likelihood only if $Y_t \perp Y_{t'} \mid L$. When labels are autocorrelated, DF training diverges from maximum-likelihood principles. [[fredf|FreDF]] mitigates this by supervising in an orthogonal frequency domain; [[qdf|QDF]] learns a quadratic $L_\Sigma$ with adaptive $\Sigma$ (autocorrelation + heterogeneous step weights).[^src-fredf][^src-qdf]

[[distdf|DistDF]] ([[source-distdf]], arXiv:2510.24574, ICLR 2026) sharpens the statement: the bias is $\|Y-\hat Y\|^2_{\Sigma_{|x}^{-1}}-\|Y-\hat Y\|^2_2$ under the *conditional* covariance $\Sigma_{|x}$, so it disappears only when future steps are conditionally independent given history, and Fourier/PCA transforms that diagonalize only the marginal covariance leave it in place. DistDF therefore keeps the DF multi-output head but replaces the likelihood objective with [[joint-distribution-wasserstein-alignment|joint-distribution Wasserstein alignment]] between $\mathbb{P}_{Y\mid X}$ and $\mathbb{P}_{\hat Y\mid X}$, using a [[bures-wasserstein|Bures–Wasserstein]] term on joint mean and covariance plus an MSE anchor; inference stays a single forward pass.[^src-distdf] See [[autocorrelation-bias]] for the formal bias and the Traffic case study behind this argument.

- **设定归属与末端退化修补（[[source-tailfix|TailFix]], NeurIPS 2026）**：针对 DF 误差沿预测步长向末端放大的现象，在 WPMixer 骨干外叠加三处零初始化近恒等门控残差并提出尾/头误差比指标 TAF（$\rho=0.25$），但 ETT 单组件消融仅变动 $0.004\text{–}0.009$（$1\%\text{–}3\%$）且 Traffic（$L=1200$）MSE $0.410$ 逊于原版 WPMixer（$0.403$）与 [[patchtst|PatchTST]]（$0.391$）[^src-tailfix]。

## Paradigm Contrast Under Sparse Endogenous Inputs

When endogenous history is extremely sparse (4% or single point) while exogenous variables remain dense, direct multi-step mapping from sparse endogenous points to future horizons suffers severe scale and causality mismatch. **范式对照（[[source-timeseed|TimeSeed]], ICML 2026）**：replacing direct forecasting with two-stage input-side history reconstruction (using dense exogenous priors and multi-scale upsampling to restore complete endogenous history before linear forecasting) drives its primary gain—reducing MSE by ~27.4% and MAE by ~14.2% on average on ETTh2/ETTm2 over direct forecasting—though two-step imputation + PatchTST still outperforms TimeSeed on ETTm1 (0.041 vs 0.068) and TSFMs are not evaluated.[^src-timeseed]

## Related

- [[label-autocorrelation]], [[heterogeneous-task-weights]]
- [[frequency-enhanced-direct-forecast]], [[quadratic-form-weighted-objective]], [[joint-distribution-wasserstein-alignment]], [[bures-wasserstein]]
- [[fredf]], [[qdf]], [[distdf]], [[tide]], [[tft]]
- [[source-fredf]], [[source-qdf]], [[source-distdf]], [[source-tide]], [[source-tft]], [[source-timeseed]], [[source-tailfix]]

---

[^src-fredf]: [[source-fredf]]
[^src-qdf]: [[source-qdf]]
[^src-tide]: [[source-tide]]
[^src-tft]: [[source-tft]]
[^src-distdf]: [[source-distdf]]
[^src-timeseed]: [[source-timeseed]]
[^src-tailfix]: [[source-tailfix]]
