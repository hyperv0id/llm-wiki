---
title: "GCGNet: Graph-Consistent Generative Network for Time Series Forecasting with Exogenous Variables"
type: source-summary
tags:
  - time-series-forecasting
  - exogenous
  - graph-neural-network
  - variational-autoencoder
  - joint-correlation
  - iclr-2026
created: 2026-07-13
last_updated: 2026-07-18
source_count: 1
confidence: medium
status: active
---

# GCGNet: Graph-Consistent Generative Network for Time Series Forecasting with Exogenous Variables

**Authors**: Zhengyu Li, Xiangfei Qiu, Yuhan Zhu, Xingjian Wu, Jilin Hu, Chenjuan Guo, Bin Yang (East China Normal University, Decision Intelligence).[^src-gcgnet]

**Venue**: ICLR 2026 | **arXiv**: 2603.08032v2 (May 2026) | **Code**: https://github.com/decisionintelligence/GCGNet

## Core Arguments

**1. Two-step exogenous forecasting interferes with itself.** Forecasting with exogenous variables requires both *temporal* correlations (past→future endogenous dynamics) and *channel* correlations (exogenous→endogenous influence). Prior deep models mostly use a two-step pipeline—either temporal-then-channel ([[source-timexer|TimeXer]], [[source-exotst|ExoTST]]) or channel-then-temporal ([[source-tft|TFT]], [[source-crosslinear|CrossLinear]])—which can cause mutual interference and suboptimal joint structure capture.[^src-gcgnet]

**2. Noise breaks naive correlation learning.** Sensor failure, transmission error, and recording mistakes inject noise so observed series no longer reflect true correlations; pointwise reconstruction overfits noise. Generative models that learn latent structure are more robust than models that read correlations directly from raw observations.[^src-gcgnet]

**3. GCGNet jointly models correlations as graphs under generative consistency.** [[gcgnet|GCGNet]] (Graph-Consistent Generative Network) combines three modules: (i) a [[variational-generator-exogenous|Variational Generator]] (VAE + instance norm) producing coarse future endogenous (and optional future exogenous) forecasts; (ii) a [[graph-structure-aligner|Graph Structure Aligner]] that patchifies full sequences, builds similarity graphs, denoises them with a Graph VAE, and minimizes L1 alignment \(L_{\mathrm{align}}=\|A-\hat A\|_1\) between ground-truth and generated joint graphs; (iii) a [[graph-refiner|Graph Refiner]] that top-k sparsifies \(\hat A\), runs multi-layer GCN message passing, and projects to the final \(\hat Y^{\mathrm{endo}}\), preventing Graph VAE degeneration while enabling joint temporal–channel information exchange.[^src-gcgnet]

**4. Problem form.** Given \(X^{\mathrm{endo}}\in\mathbb{R}^{N\times T}\), \(X^{\mathrm{exo}}\in\mathbb{R}^{D\times T}\), and optional \(Y^{\mathrm{exo}}\in\mathbb{R}^{D\times F}\), predict \(\hat Y^{\mathrm{endo}}=F_\theta(X^{\mathrm{endo}},X^{\mathrm{exo}},Y^{\mathrm{exo}})\). When future exogenous is unavailable, the generator substitutes \(\tilde Y^{\mathrm{exo}}\). Total loss: \(L_f+L_{\mathrm{align}}+L_{\mathrm{KL}}^{V}+L_{\mathrm{KL}}^{G}\).[^src-gcgnet]

## Experiments

Evaluated on **12** exogenous datasets: 5 EPF markets (NP, PJM, BE, FR, DE) plus Energy, Colbún, Rapel, and Longyuan wind sets Sdwpfh1/2, Sdwpfm1/2 (weather from ERA5). Short/long horizons: typically lookback 168→24 and 720→360 (Colbún/Rapel: 60→10, 180→30). Baselines include TimeXer, TFT, TiDE, DUET, CrossLinear, Amplifier, TimeKAN, xPatch, PatchTST (latter group extended with MLP fusion for future exo).[^src-gcgnet]

**Results.** GCGNet reports 18 first-place MSE and 20 first-place MAE rankings in Table 1. Ablations (Table 2) show large drops when removing Graph Refiner, \(L_{\mathrm{align}}\), or replacing Graph VAE / Variational Generator. Remains strong without future exo (Table 3) and under 10–50% zero/random exo masking (Table 4–5). Qualitative NP case: two-step models let one correlation type override the other; GCGNet tracks price under joint wind/load structure.[^src-gcgnet]

## Limitations / Scope

Focus is **numerical** multi-series exogenous forecasting (not image/text multimodal ST). Joint graphs are patch-level undirected similarities denoised by VAE, not causal graphs. Spatial topology of STG traffic is not the primary setting—datasets are mostly market/energy series with named covariates.[^src-gcgnet]

## Related Pages

- Entity: [[gcgnet]]
- Techniques: [[graph-structure-aligner]], [[graph-refiner]], [[variational-generator-exogenous]]
- Concept: [[joint-temporal-channel-correlation]]
- Related exogenous work: [[source-timexer]], [[source-crosslinear]], [[source-exotst]], [[source-exost]], [[source-kite]], [[source-dag]]

---

[^src-gcgnet]: [[source-gcgnet]]
