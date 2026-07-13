---
title: "Variable Selection Network"
type: technique
tags:
  - time-series
  - feature-selection
  - tft
  - interpretability
  - exogenous
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# Variable Selection Network

**Variable selection networks** in [[tft|TFT]] produce **instance-wise** Softmax weights over input variables so the model can focus capacity on salient features and down-weight noisy ones. Separate networks process static, past, and future inputs.[^src-tft]

## Mechanism

Each raw variable is first mapped to a \(d_{\mathrm{model}}\) vector \(\xi_t^{(j)}\) (entity embedding for categoricals; linear map for continuous). Flattened inputs \(\Xi_t\) and optional static context \(c_s\) enter a [[gated-residual-network|GRN]]; Softmax yields selection weights:

\[
v_{\chi t}=\mathrm{Softmax}\bigl(\mathrm{GRN}_{v_\chi}(\Xi_t,c_s)\bigr).
\]

Each variable is also processed by its own time-shared GRN, then combined:

\[
\tilde\xi_t=\sum_j v_{\chi t}^{(j)}\,\tilde\xi_t^{(j)}.
\]

For static variables the external context \(c_s\) is omitted (static information is already available).[^src-tft]

## Interpretability

Aggregating \(v_{\chi t}^{(j)}\) over the test set (e.g. 10th/50th/90th percentiles) yields **global variable importance**. On Favorita Retail, TFT highlights entity IDs among static features, past log-sales among observed inputs, and promotions/national holidays among future-known inputs — matching domain expectations without hand-coded feature filters.[^src-tft]

## Ablation

Replacing Softmax selection weights by trainable coefficients (keeping per-variable GRNs) increases P90 loss by more than ~4.1% on average in the TFT study, with large effects when many weak features are present (e.g. Electricity calendar vs usage mix).[^src-tft]

## Related Pages

- [[tft]], [[source-tft]]
- [[gated-residual-network]]
- [[heterogeneous-covariates]] — later taxonomy of categorical/multimodal covariates beyond TFT’s tabular selection
- [[interpretable-multi-head-attention]] — complementary temporal (not variable-axis) interpretability

---

[^src-tft]: [[source-tft]]
