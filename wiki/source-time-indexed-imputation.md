---
title: "Are Time-Indexed Foundation Models the Future of Time Series Imputation? (TabPFN-TS / MoTM benchmark)"
type: source-summary
tags:
  - time-series
  - data-imputation
  - foundation-model
  - zero-shot
  - benchmark
  - tmlr-2026
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

**"Are Time-Indexed Foundation Models the Future of Time Series Imputation?"** is a large-scale empirical benchmark by Etienne Le Naour, Tahar Nabil, Adrien Petralia, and Ghislain Agoua (EDF R&D, France), published in **TMLR (01/2026)** (arXiv:2511.05980v2)[^src-time-indexed-imputation]. It is the first large-scale study of **time-indexed foundation models** — [[tabpfn-ts|TabPFN-TS]] and [[motm|MoTM]] — for **zero-shot** time series imputation. Code: github.com/taharnbl/tsfm_imputation.

## Time-Indexed Foundation Models

Both models learn a contextual representation $H(t)$ at every timestamp $t$, then apply a regressor $r_\theta(\cdot)$ mapping $H(t)\to x(t)$. This **continuous-time** design (unlike patch-based forecasters trained on dense fully-observed contexts) naturally handles irregular/unaligned series, different sampling rates, arbitrary missing regions, and **covariate integration at inference time**[^src-time-indexed-imputation]. See [[time-indexed-foundation-model]].

- **[[tabpfn-ts|TabPFN-TS]]** (Hoo et al. 2025): handcrafted features (normalized time index + Fourier sin/cos basis) + a highly expressive regressor — the pre-trained **TabPFN** transformer (trained on hundreds of millions of synthetic tabular regression tasks), which imputes via **in-context learning** in a single forward pass, no fine-tuning.
- **[[motm|MoTM]]** (Le Naour et al. 2025): conceptually inverse — a learned basis of $K$ modulated **Implicit Neural Representations** (INRs) forms $H(t)$, followed by a simple in-context **ridge regression** fit on the observed context.

## Benchmark

33 out-of-domain datasets (climate/energy/traffic, sampling 5min–1h), >1.3M incomplete windows, 4 missing scenarios (50% / 70% pointwise; 2-day / 4-day block)[^src-time-indexed-imputation]. Baselines span foundation (zero-shot), task-specific supervised (SAITS, BRITS, CSDI, TimesNet, TimeMixer++, TSLANet), and local (Linear, Seasonal naive, LOCF, Cubic Spline). Metric: z-normalized MAE (NMAE).

### Aggregate results (mean NMAE, lower better)

| Rank | Model | NMAE | Category |
|------|-------|------|----------|
| 1 | **TabPFN-TS** | **0.293** (avg rank 1.35) | Foundation (zero-shot) |
| 2 | MoTM | 0.371 | Foundation (zero-shot) |
| 3 | SAITS | 0.386 | Supervised |
| 4 | BRITS | 0.470 | Supervised |
| 5 | Linear | 0.506 | Local |
| — | Seasonal naive / LOCF | 0.581 / 0.611 | Local |
| — | [[csdi\|CSDI]] / TimesNet / TimeMixer++ | 0.664 / 0.677 / 0.681 | Supervised |

**TabPFN-TS is statistically superior to all competitors** (CD diagram, avg rank 1.35)[^src-time-indexed-imputation].

## Key Findings

1. **Time-indexed FMs lead** — both beat all supervised and local baselines despite being fully zero-shot; large-scale synthetic-pretrained regressor + explicit temporal encodings (TabPFN-TS) edges out ridge-on-learned-INRs (MoTM)[^src-time-indexed-imputation].
2. **Supervised models lack robustness** — SAITS competitive, but BRITS/CSDI/TimesNet sometimes worse than simple local heuristics, overfitting dataset-specific dynamics when data is scarce[^src-time-indexed-imputation].
3. **Local baselines resilient** — Linear Interpolation robust (beats Cubic Spline in 39/44 settings), competitive under sparse pointwise missingness, but both FMs clearly dominate under structured **block** missingness[^src-time-indexed-imputation].
4. **Covariates without retraining** — both FMs incorporate covariates by concatenation at inference, drastically improving accuracy (gains largest on Wind-France/PV-France)[^src-time-indexed-imputation].
5. **NuwaTS comparison** — in supplementary experiments, [[nuwats|NuwaTS]] (zero-shot, PLM-repurposing) beats MOMENT but **lags significantly behind TabPFN-TS on all datasets/settings and behind MoTM on 10/11**, strengthening time-indexed approaches over Transformer/PLM-based ones[^src-time-indexed-imputation].

## Computational Cost (the key trade-off)

TabPFN-TS consistently has the lowest error **but the highest inference time**: ~1s per 672-step chunk on an H100. **MoTM is up to two orders of magnitude faster** while staying competitive — positioned as the scalable alternative[^src-time-indexed-imputation]. Recommendation: TabPFN-TS when GPU is available / offline batch; MoTM for resource-constrained or high-throughput settings.

## Conclusion

Time-indexed foundation models are a powerful, practical step toward general-purpose zero-shot imputation; a promising direction is replacing MoTM's ridge with an in-context-trained regressor to merge accuracy and efficiency[^src-time-indexed-imputation].

[^src-time-indexed-imputation]: [[source-time-indexed-imputation]]
