---
title: "TT-Wasserstein"
type: technique
tags:
  - multimodal-time-series
  - metric
  - frequency-analysis
  - wasserstein-distance
  - alignment-quality
  - iclr-2026
created: 2026-05-03
last_updated: 2026-05-03
source_count: 1
confidence: medium
status: active
---

# TT-Wasserstein

**TT-Wasserstein** is a metric proposed by Li et al. (ICLR 2026) to quantify the level of **[[chronological-textual-resonance|Chronological Textual Resonance (CTR)]]** — the alignment between periodic properties of time-series-paired texts and their corresponding numerical time series[^src-language-in-the-flow-of-time].

## Definition

TT-Wasserstein is defined as the Wasserstein distance between the normalized spectral distributions of time series and paired texts:

$$\text{TT-Wasserstein}(\mathcal{D}) = W(P_{\text{texts}}, P_{\text{ts}}) = \inf_{\gamma \in \Pi(P_{\text{text}}, P_{\text{ts}})} \sum_{i=1}^{n} \sum_{j=1}^{m} \gamma_{ij} \cdot |\tilde{f}_{\text{texts}}^{(i)} - \tilde{f}_{\text{ts}}^{(j)}|$$

Subject to:
$$\gamma_{ij} \geq 0, \quad \sum_{j=1}^{m} \gamma_{ij} = \tilde{a}_{\text{texts}}^{(i)}, \quad \sum_{i=1}^{n} \gamma_{ij} = \tilde{a}_{\text{ts}}^{(j)}$$

Where $\tilde{f}$ and $\tilde{a}$ are normalized frequencies and amplitudes from Fourier analysis of both modalities[^src-language-in-the-flow-of-time].

## Intuition

TT-Wasserstein measures the "effort" required to transform the frequency distribution of texts into that of the time series. A lower value indicates that the two modalities share similar dominant frequencies — i.e., stronger CTR[^src-language-in-the-flow-of-time].

## Computation Process

1. Apply Fourier Transform to both time series and text embeddings
2. Normalize frequencies and amplitudes for both modalities
3. Compute Wasserstein distance between the two normalized spectral distributions

The text periodicity is analyzed via **lag-similarity**: $d_l = \sum_t \cos(e_t, e_{t+L})$, where $L$ is the lag. If text embeddings exhibit periodic patterns, the lag-similarity fluctuates periodically as lag increases. FFT is then applied to the lag-similarity to identify dominant text frequencies[^src-language-in-the-flow-of-time].

## Validation

TT-Wasserstein was validated on 9 Time-MMD datasets across monthly, weekly, and daily sampling frequencies by comparing original vs. shuffled versions:

| Dataset | Frequency | Original | TS Shuffled | Text Shuffled |
|---------|-----------|----------|-------------|---------------|
| Agriculture | Monthly | 0.026 | 0.088 | 0.106 |
| Climate | Monthly | 0.025 | 0.032 | 0.037 |
| Economy | Monthly | 0.022 | 0.098 | 0.099 |
| Security | Monthly | 0.049 | 0.054 | 0.053 |
| Social Good | Monthly | 0.027 | 0.069 | 0.072 |
| Traffic | Monthly | 0.035 | 0.102 | 0.104 |
| Energy | Weekly | 0.307 | 0.320 | 0.312 |
| Health | Daily | 0.233 | 0.268 | 0.277 |
| Environment | Daily | 0.302 | 0.358 | 0.364 |

Shuffled datasets consistently yield much larger TT-Wasserstein values, confirming the metric's ability to detect alignment disruption[^src-language-in-the-flow-of-time]. Notably, monthly-sampled datasets have much lower original TT-Wasserstein (0.022–0.049) than weekly (0.307) and daily (0.233–0.302) datasets, indicating stronger CTR in monthly data — which correlates with TaTS's larger performance improvements on monthly datasets[^src-language-in-the-flow-of-time].

## Practical Uses

1. **Dataset quality assessment**: Gauge whether paired texts provide meaningful complementary information
2. **Predicting TaTS effectiveness**: Lower TT-Wasserstein correlates with larger performance gains from [[tats|TaTS]]
3. **Filtering unsuitable datasets**: High TT-Wasserstein indicates texts may not help (e.g., daily lottery numbers)[^src-language-in-the-flow-of-time]

## Limitations

- **Empirical statistical metric**: Estimation stability depends on sample size (analyzed in Appendix A.1 of the paper)
- **Frequency-domain only**: Does not capture phase alignment or semantic content quality
- **Requires sufficient data**: Needs enough timestamps for reliable FFT[^src-language-in-the-flow-of-time]

## Related Pages

- [[chronological-textual-resonance]] — CTR phenomenon
- [[tats]] — TaTS framework
- [[texts-as-auxiliary-variables]] — core design concept
- [[source-language-in-the-flow-of-time]] — source summary
- [[multimodal-time-series-forecasting]] — task concept

[^src-language-in-the-flow-of-time]: [[source-language-in-the-flow-of-time]]