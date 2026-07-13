---
title: "FreDF: Learning to Forecast in Frequency Domain"
type: source-summary
tags:
  - time-series-forecasting
  - frequency-domain
  - direct-forecast
  - label-autocorrelation
  - learning-objective
  - iclr
created: 2026-07-13
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

## Summary

**FreDF (Frequency-enhanced Direct Forecast)** is a model-agnostic training paradigm for multi-step time series forecasting that aligns forecasts and labels in the frequency domain. Authored by Hao Wang, Licheng Pan, Zhichao Chen, Degui Yang, Sen Zhang, Yifei Yang, Xinggao Liu, Haoxuan Li, and Dacheng Tao (Zhejiang University, Central South University, Nanyang Technological University, Shanghai Jiao Tong University, Peking University); arXiv:2402.02399 (preprint Feb 2024; source file labeled ICLR 2025). Code: https://github.com/Master-PLC/FreDF.[^src-fredf]

## Core Arguments

**1. Label autocorrelation breaks the DF likelihood assumption.** Modern multi-step models use the direct forecast (DF) paradigm: a multi-output head predicts all future steps jointly under MSE. Theorem 3.1 shows that MSE matches the conditional negative log-likelihood only if label steps are conditionally independent given history ($Y_t \perp Y_{t'} \mid L$). Label sequences are autoregressively generated, so this assumption fails and training is biased relative to the true likelihood.[^src-fredf]

**2. Empirical verification via double machine learning (DML).** Treating history as a confounder, DML estimates causal strength $Y_t \to Y_{t'}$ on Weather ($T=192$). About 37.5% of off-diagonal entries exceed 0.3, confirming label autocorrelation; the matrix shows periodic structure. After Fourier transform, only ~3.6% of off-diagonal frequency-component causations exceed 0.1, indicating near-independence among frequency bases.[^src-fredf]

**3. Frequency-domain alignment as a simple DF upgrade.** FreDF keeps any backbone $g$ and adds frequency supervision: compute time-domain MSE $L^{(\mathrm{tmp})}$; FFT both forecast and label to get $F, \hat F$; define frequency loss $L^{(\mathrm{feq})}$ as the sum of complex moduli $|F-\hat F|$ (not squared, because frequency magnitudes span orders of magnitude); fuse $L_\alpha = \alpha L^{(\mathrm{feq})} + (1-\alpha) L^{(\mathrm{tmp})}$. Orthogonality of Fourier bases mitigates label autocorrelation while retaining DF's parallel multi-step inference.[^src-fredf]

## Experiments

Long-term forecast (ETT×4, ECL, Traffic, Weather; horizons 96/192/336/720) uses iTransformer+FreDF and reports average MSE/MAE SOTA vs iTransformer, FreTS, TimesNet, Crossformer, TiDE, DLinear, FEDformer, Autoformer, Transformer, TCN, LSTM (e.g., ETTm1 MSE 0.392 vs iTransformer 0.415). Short-term M4 improves FreTS (SMAPE 12.112 / MASE 1.648 / OWA 0.877). Imputation also improves iTransformer. Ablations: pure frequency loss already helps; joint time+frequency is often best near $\alpha \approx 0.8$–1.0; both amplitude and phase matter (phase especially). FreDF generalizes across iTransformer, DLinear, Autoformer, Transformer; 1D-time / 1D-feature / 2D FFT; and Legendre/Fourier/Chebyshev/Laguerre bases (orthogonal Legendre & Fourier best). With 30% training data, frequency learning can match full-data time-domain performance.[^src-fredf]

## Limitations

Fourier bases are fixed and may not adapt to data geometry; data-adaptive orthogonal transforms (e.g., PCA) are suggested. Label-structure correlations beyond 1D series (images, speech, point clouds) are left open. Later DistDF argues that Fourier/PCA only guarantee *marginal* decorrelation, not the *conditional* independence needed to fully remove MSE likelihood bias.[^src-fredf]

## Key Terminology

- **Label autocorrelation**: dependence among future steps of the multi-step label sequence given history
- **Direct forecast (DF)**: multi-output multi-step prediction under step-wise loss (vs iterative forecast)
- **Frequency loss $L^{(\mathrm{feq})}$**: modulus-based alignment of FFT(forecast) and FFT(label)
- **FreDF**: plug-and-play frequency-enhanced DF training with mix weight $\alpha$

---

[^src-fredf]: [[source-fredf]]
