---
title: "Source: RSTIB-MLP — Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting"
type: source-summary
tags:
  - time-series
  - spatial-temporal
  - information-bottleneck
  - mlp
  - robustness
  - traffic-forecasting
  - icml-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 0
confidence: high
status: active
---

# Source: RSTIB-MLP

**Full title:** Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting  
**Authors:** Min Chen, Guansong Pang, Wenjun Wang, Cheng Yan (Tianjin University, Singapore Management University)  
**Venue:** ICML 2025 (Proceedings of the 42nd ICML, Vancouver)  
**Links:** [GitHub](https://github.com/mchen644/RSTIB)

## Summary

RSTIB-MLP investigates whether simple neural networks such as MLPs can achieve robust spatial-temporal forecasting (STF) while remaining computationally efficient. The paper first identifies a **dual noise effect** in STF: under the sliding window mechanism, the same data sequence can serve as either input or target in different windows, causing noise to harm both ends simultaneously[^src-rstib]. This dual noise leads to both sample indistinguishability and feature collapse (quantified by lower feature variance), effects that prior methods largely overlook[^src-rstib].

To address this, the authors propose the **Robust Spatial-Temporal Information Bottleneck (RSTIB)** principle, a theoretically-grounded generalization of RGIB that lifts the Markov assumption Z–X–Y while not impairing IB nature. RSTIB explicitly minimizes noisy information from both the input end (captured by I(X;Z|Y)) and the target end (captured by I(Z;Y|X)), introducing the interaction information I(X;Y;Z) to decouple mutual information terms[^src-rstib].

The RSTIB-MLP instantiation uses pure MLP networks with data reparameterization to derive upper bounds for input, target, and representation regularization (all KL divergences to unit Gaussian with analytical solutions). A key innovation is a **knowledge distillation module** in the training regime: a pre-trained teacher model quantifies noise impact per time series via a softmax-normalized distance metric (noise impact indicator α̂), which dynamically balances the informative terms in the learning objective[^src-rstib].

On six benchmark datasets (PEMS04/07/08, LargeST, Weather2K-R, Electricity) under noise ratios 0%–50%, RSTIB-MLP achieves better or comparable robustness vs. SOTA STGNNs (GWN, STG-NCDE, TrendGCN, BiTGraph) and MLP baselines (STID, FreTS), while being substantially more efficient than STGNNs (∼10× faster training per epoch on PEMS04)[^src-rstib].

## Key Contributions

1. First theoretical framework (RSTIB) for combating the dual noise effect in STF by generalizing RGIB with lifted Markov assumption[^src-rstib]
2. RSTIB-MLP instantiation with closed-form analytical regularization bounds and data reparameterization[^src-rstib]
3. Knowledge distillation module with noise impact indicator α̂ for dynamic balance of regularization terms per time series[^src-rstib]
4. Comprehensive evaluation showing superior robustness-efficiency trade-off across six datasets and 10+ robust baselines[^src-rstib]

## Limitations

RSTIB-MLP is slightly less efficient than the simplest MLP baseline STID (an acceptable trade-off). The teacher model for noise impact indicator requires pre-training. Clean-data improvements are marginal; the method is primarily designed for noisy scenarios[^src-rstib].

[^src-rstib]: [[source-rstib-mlp]]
