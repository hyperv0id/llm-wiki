---
title: "A Novel Framework for Significant Wave Height Prediction based on Adaptive Feature Extraction Time-Frequency Network"
type: source-summary
tags:
  - time-series
  - forecasting
  - frequency-domain
  - wavelet-transform
  - significant-wave-height
  - encoder-decoder
  - lstm
  - ocean-engineering
created: 2026-04-28
last_updated: 2026-04-28
source_count: 0
confidence: high
status: active
---

# AFE-TFNet

**AFE-TFNet** (Adaptive Feature Extraction Time-Frequency Network) is an encoder-decoder rolling framework for significant wave height (Hs) prediction, proposed by Jianxin Zhang, Lianzi Jiang, Xinyu Han, and Xiangrong Wang (Shandong University of Science and Technology). It addresses the challenges of non-linear and non-stationary wave signals by combining wavelet transform and Fourier transform feature extraction with a novel Dominant Harmonic Sequence Energy Weighting (DHSEW) mechanism, all within a rolling window framework that prevents data leakage.

## Overview

Hybrid models combining decomposition preprocessing with machine learning often suffer from data leakage when decomposition is applied to entire datasets including test sets[^src-afe-tfnet]. Existing rolling decomposition models (e.g., VMD-LSTM-Rolling) address leakage but have limited feature extraction capabilities. AFE-TFNet introduces a parallel feature extraction stage that simultaneously captures local frequency features via Wavelet Transform (WT) and global periodic features via Fast Fourier Transform (FFT), within each rolling training window independently. A Frequency Inception Block (FIB) then performs multi-scale convolution on the extracted features, and the DHSEW mechanism dynamically weights time-domain vs. frequency-domain contributions based on the input's harmonic energy ratio.

## Key Method

AFE-TFNet's encoder-decoder architecture has two stages[^src-afe-tfnet]:

1. **Feature Extraction Stage**: 
   - **Wavelet Transform (Morlet wavelet)**: Extracts local time-frequency features using logarithmically spaced scales, capturing instantaneous signal variations.
   - **Fourier Transform + Frequency Reshape**: After FFT, the top-k dominant frequencies are identified. The signal is segmented and reshaped into 2D maps based on the corresponding periods, enabling analysis of periodic patterns at multiple scales.
   - **Frequency Inception Block (FIB)**: Inspired by Inception networks, applies parallel 1×1, 3×3, and 5×5 convolutions to the concatenated WT and reshaped FFT features, capturing both high-frequency (small kernel) and low-frequency (large kernel) information. A 3×3 max pooling path provides translation invariance.

2. **Feature Fusion Stage (DHSEW)**: Computes the energy ratio of dominant harmonic sequences (fundamental frequency plus harmonics) to total spectral energy. The weight w_f = E_h / E_f is used to scale frequency-domain features, while w_t = 1 − w_f scales time-domain features. Strongly periodic signals emphasize frequency features; aperiodic signals emphasize time features.

3. **Decoder**: An LSTM network processes the fused multi-dimensional features to generate the next time step prediction, operating within a rolling window that advances one step at a time.

## Results

AFE-TFNet was evaluated on hourly data from three NDBC buoy stations (No.41010, No.46025, No.46029) with input variables Ws, DPD, APD, and Hs[^src-afe-tfnet]. Prediction horizons of 1h, 3h, 6h, and 12h were tested against baselines: NaiveDrift, XGBoost, CatBoost, LightGBM, LSTM, TCN, and MWNet. AFE-TFNet achieved statistically significant improvements (Wilcoxon signed-rank test, p < 0.05) across all comparisons. Average reductions of 20.23% in RMSE, 21.16% in MAE, 28.7% in MAPE, and 5.86% improvement in R over the second-best model were reported. Performance gains were modest at 1h but grew substantially at 6h and 12h horizons, with the gap widening as prediction length increased. The model showed robustness to changes in rolling window size.

## Critique

AFE-TFNet's key innovation is the joint use of WT and FFT within a rolling framework that strictly prevents data leakage—a practical concern often overlooked in decomposition-based forecasting papers. The DHSEW mechanism parallels Dualformer's periodicity-aware weighting, suggesting convergent evolution in time-frequency fusion strategies. The Frequency Inception Block is a clever adaptation of computer vision architectures to frequency-domain features. However, the paper's experimental scope is limited to wave height prediction at three stations; generalizability to broader time series domains is unproven. The baseline comparison does not include recent Transformer-based forecasting models (PatchTST, FEDformer, etc.), making it hard to assess relative standing in the broader forecasting literature. The LSTM decoder, while effective, may limit the model's ability to capture very long-range dependencies compared to attention-based alternatives. The rolling window approach, while preventing leakage, is computationally more expensive than direct forecasting.

[^src-afe-tfnet]: [[source-afe-tfnet]]
