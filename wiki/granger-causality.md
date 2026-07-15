---
title: "Granger Causality"
type: technique
tags:
  - causal-inference
  - time-series
  - statistical-test
  - predictive-causality
created: 2026-07-14
last_updated: 2026-07-14
source_count: 4
confidence: high
status: active
---

# Granger Causality

**Granger causality** (Granger, 1969) is a statistical test of predictive causality: a time series X is said to "Granger-cause" Y if past values of X contain information that helps predict future values of Y, beyond the information contained in past values of Y alone[^src-causalx][^src-cora].

It is fundamentally a **predictive** notion of causality — not to be confused with structural or interventional causality. Granger causality captures whether a variable is predictively useful for another, which may arise from genuine causal links, shared latent drivers, or other statistical associations[^src-cora].

## Formal Definition

For a maximum lag order L, the Granger causality score from variable e to target variable tar is computed via F-tests or similar hypothesis tests on the significance of lagged e terms in a VAR model predicting tar[^src-causalx]. The null hypothesis is that e does not Granger-cause tar. The score is often summarized as:

$$\text{Granger}(e \to \text{tar}) = 1 - \frac{1}{B}\sum_{b=1}^{B} \min_{\ell \in [1,L]} \text{p-value}_{b,\ell}^{(e \to \text{tar})}$$

where B is batch size and the p-values are normalized to [0, 1] range[^src-causalx].

## Usage in Deep Learning

### Causal Graph Learning (CausalX)

In [[causalx|CausalX]], Granger causality is computed directly on observed input variables (not intermediate features) and broadcast across time steps and batches to form a supervision graph CG_Granger, constraining learned GAT attention weights via MSE loss[^src-causalx]. Among the four causal constraints, removing Granger caused the largest performance degradation on TCNM, underscoring its predictive importance[^src-causalx].

### Covariate Selection (CoRA)

In [[cora-tsfm|CoRA]], a learnable Causality Embedding vector automatically learns per-covariate Granger causal significance, and experiments confirm high correlation with traditional Granger-Geweke statistical tests[^src-cora]. Granger ≠ correlation: a covariate may be uncorrelated yet Granger-causal (e.g., periodic sine/cosine pairs)[^src-cora].

### Attention Modulation (KITE/DAG)

In [[kite|KITE]] and [[dag|DAG]], statistical priors including Pearson correlation and Granger causality are used to modulate cross-variable attention weights, with Granger being more effective on the historical side (past driving future) and Pearson on the future side (contemporaneous covariation)[^src-kite][^src-dag].

## Distinction from Structural Causality

Granger causality is a **predictive** test, not a structural one. A variable can Granger-cause another due to:
- Genuine direct causal influence
- Shared latent confounders
- Anticipatory behavior (e.g., economic agents acting on expectations)

For structural or interventional causal reasoning, complementary techniques like [[do-calculus]] or structural causal models are needed[^src-causalx].

## Links

- [[causalx]] — uses Granger as one of four multi-source causal constraints
- [[cora-tsfm]] — Granger-based covariate selection for TSFMs
- [[do-calculus]] — interventional complement to Granger's predictive view
- [[causal-time-series-forecasting]] — broader causal TS paradigm

[^src-causalx]: [[source-causalx]]
[^src-cora]: [[source-cora]]
[^src-kite]: [[source-kite]]
[^src-dag]: [[source-dag]]
