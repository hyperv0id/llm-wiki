# SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting

**Authors:** Hang Ding, Xue Wang, Tian Zhou, Tao Yao (Shanghai Jiao Tong University, Alibaba Group US, DAMO Academy Alibaba Group, Hupan Lab)

**Venue:** The Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)

**Code:** https://github.com/Dear-Sloth/SimDiff/tree/main

## Abstract

Diffusion models have recently shown promise in time series forecasting, particularly for probabilistic predictions. However, they often fail to achieve state-of-the-art point estimation performance compared to regression-based methods. This limitation stems from difficulties in providing sufficient contextual bias to track distribution shifts and in balancing output diversity with the stability and precision required for point forecasts. Existing diffusion-based approaches mainly focus on full-distribution modeling under probabilistic frameworks, often with likelihood maximization objectives, while paying little attention to dedicated strategies for high-accuracy point estimation. Moreover, other existing point prediction diffusion methods frequently rely on pre-trained or jointly trained mature models for contextual bias, sacrificing the generative flexibility of diffusion models.

To address these challenges, the authors propose SimDiff, a single-stage, end-to-end framework. SimDiff employs a single unified Transformer network carefully tailored to serve as both denoiser and predictor, eliminating the need for external pre-trained or jointly trained regressors. It achieves state-of-the-art point estimation performance by leveraging intrinsic output diversity and improving mean squared error accuracy through multiple inference ensembling. Key innovations, including normalization independence and the median-of-means estimator, further enhance adaptability and stability. Extensive experiments demonstrate that SimDiff significantly outperforms existing methods in time series point forecasting.

## 1. Introduction

Two inseparable questions hinder diffusion-based forecasting: (i) how to inject sufficient contextual bias from past observations to obtain a stable and faithful predictive distribution? and (ii) how to reconcile the inherent trade-off between output diversity and point-forecast accuracy?

Early likelihood-driven approaches (TimeGrad, CSDI) maximize log-probability and produce richly diverse samples, but real-world time series exhibit pronounced distribution drift between historical and future windows. These models neglect to make targeted adjustments to distribution drift, leading to unstable training, exploding sampling variance, and poor MSE/MAE scores.

To tame this instability, TimeDiff and mr-Diff prepend a pre-trained autoregressive predictor whose outputs serve as the initial trajectory. This fixes the diffusion process to a deterministic baseline, constraining the model's ability to explore the full range of possible distributions. TMDM goes further by jointly training a mature transformer predictor and a conditional diffusion model within a Bayesian ELBO framework, but still relies on an embedded regressor.

Two core questions the authors address:
1. Can the generative nature of diffusion models be harnessed to enhance point estimation through ensemble methods?
2. Is it possible to train a purely end-to-end diffusion model without relying on mature regression models to provide contextual bias?

## 2. SimDiff: Method

### Diffusion and Denoising For Time Series

The forward diffusion step for future values Y at step k: Y_k = sqrt(ᾱ_k) Y_0 + sqrt(1 - ᾱ_k) ε, where ε ~ N(0, I). The backward denoising process reconstructs the future time series through a denoising transformer backbone. The model initializes from Ŷ_K ~ N(0, I) and iteratively denoises.

### Normalization Independence (N.I.)

Past and future segments rarely share the same level or scale. Prior work normalizes both with past statistics (implicitly assuming stationarity). N.I. breaks this coupling: past samples are instance-normalised and rescaled by learnable (γ, β); future targets are normalised with their own statistics independently only during training. At test time, predictions start from standard Gaussian noise, then de-normalise using only past statistics and learned affine parameters. This adds only a lightweight affine layer with negligible cost but markedly improves robustness to distribution drift.

### Transformer Denoising Network

Key architectural choices:
- **Patch-based Tokenization**: Overlapping patches converted to tokens via dense MLP; diffusion timesteps processed into a time token and concatenated.
- **Rotary Position Embedding (RoPE)**: Encodes relative positional information through rotational transformations.
- **Channel Independence and No Skip Connections**: Skip connections amplify noise in time series, so SimDiff removes them. Uses channel independence (processing each channel separately) to enhance efficiency, increase data volume, and improve distribution learning.

### Median of Means (MoM) Ensemble

The MoM estimator divides n samples into K subsamples of size B, computes their means μ̂_1,...,μ̂_K, and takes the median. This is repeated R times with shuffled data. The final estimator is the average of the R medians: μ̂_MoM = (1/R) Σ median(μ̂_1^(r),...,μ̂_K^(r)). MoM effectively captures the true data distribution, retaining subtle temporal patterns rather than producing a smoothed trajectory. It is robust to outliers and heavy-tailed noise.

### Loss Function

Weighted MAE loss, with weights adjusted by the cumulative noise reduction over diffusion steps, ensuring the model focuses on periods with higher noise levels.

## 3. Experiments

SimDiff achieves the best performance on 6 out of 9 datasets (NorPool, Caiso, Electricity, ETTh1, ETTm1, Wind), with particularly notable improvements on challenging datasets. On large datasets like Traffic and Electricity, SimDiff achieves SOTA or comparable results. Quantitatively, SimDiff reduces MSE by an average of 8.3% across all datasets compared to other diffusion models like mr-Diff.

Key results:
- **Probabilistic performance**: Achieves competitive CRPS and CRPS-sum scores despite not being explicitly optimized for probabilistic tasks, leveraging the inherent generative capacity of diffusion.
- **Point forecasting**: Best rank (1.33 average) across 9 datasets, outperforming both regression-based (PatchTST 3.22, N-Hits 7.11) and diffusion-based (mr-Diff 4.00, TimeDiff 5.67) methods.
- **Sample diversity vs. accuracy**: SimDiff attains lower MSE with controlled variance compared to CSDI and TimeGrad, confirming samples capture temporal structure more faithfully.
- **N.I. ablation**: N.I. consistently improves performance across datasets, especially for those with severe OOD (e.g., Weather, NorPool).
- **MoM ablation**: MoM delivers the largest accuracy gain compared to simple averaging or single-sample inference.
- **Inference efficiency**: Over 90% improvement in inference speed compared to other diffusion models.

## 4. Related Works

Covers diffusion models for time series (TimeGrad, CSDI, SSSD, TimeDiff, mr-Diff), transformers and diffusion transformers (Informer, Autoformer, PatchTST, FEDformer, U-ViT, DiT), and other deep learning approaches (FiLM, NBeats, SCINet, TimesNet, DeepAR).

## 5. Conclusion

SimDiff addresses the twin challenges of providing sufficient contextual bias and balancing diversity with accuracy, integrating a tailored Transformer within a diffusion framework without relying on any external pre-trained or jointly trained models. Normalization independence and the Median-of-Means estimator improve robustness to noise and distribution shifts. SimDiff delivers faster inference than prior diffusion-based models.

## Key Contributions

1. First fully end-to-end diffusion model achieving stable SOTA results in time series point forecasting
2. Normalization Independence (N.I.) — diffusion-specific technique that better captures data distributions and mitigates temporal drift
3. Simple yet efficient transformer backbone with clear empirical validation
4. Matches leading probabilistic models (CRPS, CRPS-sum) without explicit probabilistic design
5. Median-of-Means (MoM) estimator that aggregates probabilistic samples for SOTA point forecasting
