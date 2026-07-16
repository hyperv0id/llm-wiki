---
title: "Spectral Domain Calibration (SD-Calibrator)"
type: technique
tags:
  - spectral-domain
  - test-time-computing
  - phase-amplitude-modulation
  - distribution-shift
  - traffic-forecasting
  - plug-and-play
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Spectral Domain Calibration (SD-Calibrator)

**Spectral Domain Calibration** is the "what to compute" component of [[st-ttc|ST-TTC]] (NeurIPS 2025): a lightweight plug-and-play module that corrects a frozen backbone's prediction by modulating **amplitude and phase** of its frequency components, rather than correcting in the time domain[^src-st-ttc]. The motivation is that non-stationary drift in periodic spatio-temporal signals is expressed transparently in the spectral domain — amplitude fluctuations (changing peak magnitudes) and phase shifts (peak hours advancing/delaying) map directly to changes in specific frequency bins[^src-st-ttc].

## Why the spectral domain

Time-domain correction requires extensive parameterization, increases model complexity, and struggles to capture evolving periodic structure; coupled structural/branching modules also overfit random noise[^src-st-ttc]. Spectral correction is more direct and robust, but raises two challenges the SD-Calibrator must solve: (1) the degree of non-stationarity varies across spatial nodes, and (2) full-spectrum parameterization is expensive[^src-st-ttc]. The answer is a **lightweight, spatial-aware** calibrator[^src-st-ttc].

## Three steps

Given backbone prediction $\hat y\in\mathbb{R}^{B\times N\times T}$[^src-st-ttc]:

1. **Spatial-aware decomposition.** Apply a real-to-complex FFT (rFFT) along time, *per spatial node*: $Y_f=\text{rFFT}(\hat y)\in\mathbb{C}^{B\times N\times M}$, where $M=T/2+1$. Decompose into amplitude $A=|Y_f|$ and phase $P=\angle Y_f$[^src-st-ttc].

2. **Group-wise modulation.** Partition the $M$ bins into $G$ contiguous groups of size $\lfloor M/G\rfloor$. Learn per-group, per-node offsets $\lambda_\alpha,\lambda_\phi\in\mathbb{R}^{G\times N\times1}$, **both initialized to 0** so the calibrator is an identity map before any learning (avoiding incorrect early calibration)[^src-st-ttc]. For each group $g$:
   $$A'_g = A_g\odot(1+\lambda_\alpha^g), \qquad P'_g = P_g+\lambda_\phi^g, \qquad Y'_{f,g}=A'_g\odot e^{j P'_g}$$

3. **Inverse transform.** Reconstruct the calibrated time-domain signal via inverse rFFT: $\hat y_{\text{cal}}=\text{irFFT}(Y'_f)\in\mathbb{R}^{B\times N\times T}$[^src-st-ttc].

## Complexity

The full-spectrum alternative would learn independent amplitude/phase offsets for every bin and node ($2NM$ parameters). The $G$-group design needs only $2NG$ parameters; since $G$ is a constant and $M$ grows linearly with horizon $T$, $G\ll M$, sharply cutting memory and gradient-update cost while keeping interpretable per-band calibration[^src-st-ttc]. The dominant runtime cost is the $O(NT\log T)$ rFFT/irFFT[^src-st-ttc].

## Theoretical guarantee

Under bounded modulation $|\lambda_\alpha^g|\le\epsilon_\alpha$, $|\lambda_\phi^g|\le\epsilon_\phi$, the calibration error is bounded by $\|y'-y\|_2 \le (\epsilon_\alpha+\epsilon_\phi)\|Y\|_2$ (proved via a first-order expansion and Parseval's theorem)[^src-st-ttc]. This sub-linear bound, plus the group-wise reduction of degrees of freedom from $O(NM)$ to $O(NG)$, inherently limits overfitting to transient noise[^src-st-ttc].

## Empirical findings

In ST-TTC's ablations, **frequency-domain calibration significantly outperforms time-domain calibration, with amplitude modulation being the primary contributor**; learning phase alone is weaker[^src-st-ttc]. Sharing offsets across nodes (instead of per-node) degrades performance due to spatial heterogeneity[^src-st-ttc].

## Relation to other spectral methods

SD-Calibrator differs from spectral *forecasting* models — e.g. [[fedformer|FEDformer]]'s [[frequency-enhanced-block|FEB]]/[[frequency-enhanced-attention|FEA]] blocks, [[source-frets|FreTS]], or [[specstg|SpecSTG]]'s spectral diffusion — which embed frequency-domain operators *inside* the trained model. SD-Calibrator instead operates *after* a frozen backbone, learning only minor per-band corrections at test time[^src-st-ttc].

## Related pages

- [[st-ttc]] — the parent method
- [[flash-gradient-update]] — how the calibrator parameters are updated at test time
- [[test-time-computing-st]] — the broader paradigm
- [[source-st-ttc]] — source summary

[^src-st-ttc]: [[source-st-ttc]]
