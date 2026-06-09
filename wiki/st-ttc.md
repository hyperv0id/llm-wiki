---
title: "ST-TTC"
type: entity
tags:
  - spatial-temporal
  - traffic-forecasting
  - test-time-computing
  - spectral-domain
  - distribution-shift
  - plug-and-play
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# ST-TTC

**ST-TTC** (full title: *Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting*) is a NeurIPS 2025 Spotlight method by Wei Chen and Yuxuan Liang (HKUST-Guangzhou) that corrects distribution shift in [[traffic-forecasting|spatio-temporal forecasting]] **at inference time, without retraining**[^src-st-ttc]. It frames the problem as [[test-time-computing-st|test-time computing]]: a lightweight, plug-and-play calibrator $g_\theta$ is appended after a frozen pre-trained backbone $f_\theta$ and continuously adapts to the evolving test distribution[^src-st-ttc].

## Motivation

Deployed ST models degrade because real-world periodic patterns (daily/weekly cycles in traffic flow, air quality, energy) are **non-stationary**: they drift via amplitude fluctuations (peaks rising/falling with seasons) and phase shifts (rush hours advancing/delaying with congestion)[^src-st-ttc]. Pre-trained models fit fixed periodic patterns and cannot track these gradual systematic biases[^src-st-ttc]. Prior remedies — OOD architectures, data augmentation, continual fine-tuning — are computationally expensive and assume training data captures all future invariance, which rarely holds[^src-st-ttc]. ST-TTC instead exploits a property unique to STF: **label autocorrelation**, where sliding-window construction makes the true labels of past test samples available for supervised calibration[^src-st-ttc].

## Architecture

ST-TTC composes two components, organized around *what to compute* and *how to compute it*[^src-st-ttc]:

### 1. Spectral Domain Calibrator (SD-Calibrator)

See [[spectral-domain-calibration]] for the full technique. The calibrator operates on the backbone's time-domain prediction $\hat y \in \mathbb{R}^{B\times N\times T}$[^src-st-ttc]:

- **Spatial-aware decomposition** — apply real-FFT per node along time: $Y_f = \text{rFFT}(\hat y) \in \mathbb{C}^{B\times N\times M}$ with $M=T/2+1$; decompose into amplitude $A=|Y_f|$ and phase $P=\angle Y_f$[^src-st-ttc].
- **Group-wise modulation** — split the $M$ bins into $G$ groups; learn per-group, per-node offsets $\lambda_\alpha,\lambda_\phi\in\mathbb{R}^{G\times N\times1}$ (both initialized to 0). Modulate: $A'_g = A_g\odot(1+\lambda_\alpha^g)$, $P'_g = P_g+\lambda_\phi^g$[^src-st-ttc].
- **Inverse transform** — reconstruct via inverse rFFT to get the calibrated signal $\hat y_{\text{cal}}$[^src-st-ttc].

This uses only $2NG$ parameters (vs $2NM$ for full-spectrum) and runs in $O(NT\log T)$ FFT time[^src-st-ttc]. Calibrating in the spectral domain — where periodicity is transparently expressed as amplitude/phase of frequency components — is more direct and robust than time-domain correction, and the group-wise design addresses both spatial heterogeneity of non-stationarity and the cost of full-spectrum parameterization[^src-st-ttc].

### 2. Flash Gradient Update with Streaming Memory Queue

See [[flash-gradient-update]]. A FIFO queue of size equal to the prediction horizon $T_f$ buffers $(X_t, Y_t)$ pairs[^src-st-ttc]. When full, the *dequeued (oldest)* pair is used for a **single-sample, single-step** gradient descent on the calibrator parameters $\lambda \leftarrow \lambda - \eta\nabla_\lambda L$, with the backbone frozen[^src-st-ttc]. Using the dequeued sample (rather than the just-arrived one) avoids the information-leakage problem identified by Lau et al. (2025)[^src-st-ttc].

## Theoretical guarantees

- **Theorem 1 (output-perturbation bound):** under $|\lambda_\alpha^g|\le\epsilon_\alpha$, $|\lambda_\phi^g|\le\epsilon_\phi$, the calibration error satisfies $\|y'-y\|_2 \le (\epsilon_\alpha+\epsilon_\phi)\|Y\|_2$ (via Parseval's theorem), ensuring controlled deviation from the original prediction[^src-st-ttc].
- **Proposition 2 (controlled descent):** a single update step strictly reduces the loss on the dequeued sample when the learning rate $\eta < 2/L_c$, with bounded parameter change $\|\lambda_{k+1}-\lambda_k\|_2 \le \eta G_{\max}$[^src-st-ttc].

## Results

| Setting | Backbone(s) | Result |
|---|---|---|
| Default (6 datasets) | STAEformer, STTN, [[gwnet|GWNet]], [[stgcn|STGCN]], STID, ST-Norm | Consistent ~1–2% MAE/RMSE gains[^src-st-ttc] |
| METR-LA | GWNet | RMSE 7.43 → 7.21; beats TTT-MAE, TENT, CompFormer, DOST[^src-st-ttc] |
| Large-scale | [[patchstg|PatchSTG]] on LargeST (≤8,600 nodes) | Gains across SD/GBA/GLA/CA; +≤3.82 min inference vs 14 h training[^src-st-ttc] |
| OOD learning | STONE | Larger gains as shift increases (all & new nodes)[^src-st-ttc] |
| Continual learning | [[eac|EAC]], STKEC | Up to 32.6% MAE reduction on Energy-Stream[^src-st-ttc] |
| Efficiency | GWNet on METR-LA | 4.64× faster, 37.12% less GPU memory vs least-efficient TTA baseline[^src-st-ttc] |

Ablations show frequency-domain calibration far outperforms time-domain, with **amplitude modulation the primary contributor**; node-sharing hurts (spatial heterogeneity); and the single-sample/single-step flash update is near-optimal — more samples or steps add cost with <1% gain[^src-st-ttc].

## Limitations

ST-TTC calibrates only the *outputs*; it does not enhance the backbone's internal computation, and full-shot gains are modest[^src-st-ttc]. It relies on STF's label-autocorrelation property, which does not transfer to vision/NLP settings[^src-st-ttc]. Future work targets test-time enhancement of the internal capacity of ST foundation models[^src-st-ttc].

## Related pages

- [[source-st-ttc]] — source summary
- [[spectral-domain-calibration]] — SD-Calibrator technique
- [[flash-gradient-update]] — streaming-queue single-step update
- [[test-time-computing-st]] — the test-time computing paradigm
- [[test-time-adaptation-st]] — UrbanMind's masked-reconstruction TTA (related, label-free)
- [[traffic-forecasting]] — application domain
- [[patchstg]], [[gwnet]], [[stgcn]] — backbones used in experiments

[^src-st-ttc]: [[source-st-ttc]]
