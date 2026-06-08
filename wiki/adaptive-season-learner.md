---
title: "Adaptive Season Learner (ASL)"
type: technique
tags:
  - temporal-decomposition
  - multi-scale-learning
  - spatiotemporal-forecasting
  - frequency-domain
  - graph-neural-networks
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Adaptive Season Learner (ASL)

**ASL** (Adaptive Season Learner) is a spatiotemporal pattern learning module from [[dpgnet|DPGNet]] (ICLR 2026, under review) that models **complex temporal dynamics** of nodes and constructs **graph structures across different temporal patterns and time scales**[^src-dpgnet].

## Three-Stage Design

ASL decomposes temporal processing into three coordinated mechanisms[^src-dpgnet]:

### Stage 1: Multi-Scale View Construction

Average pooling downsamples the embedded input $H \in \mathbb{R}^{N \times L \times h}$ into m scales[^src-dpgnet]:

$$\tilde{H} = \{H^0, H^1, \ldots, H^m\},\quad H^i \in \mathbb{R}^{N \times \lfloor L/2^i \rfloor \times h}$$

- $H^0$ = finest scale (original input), captures detailed temporal information
- $H^m$ = coarsest scale, contains macro-level trends
- Each scale processed independently before fusion

### Stage 2: Temporal Decomposition

For each scale $i$, a moving average separates trend and seasonal components[^src-dpgnet]:

$$H^i_T = \text{AvgPool}(\text{Padding}(H^i)),\quad H^i_S = H^i - H^i_T$$

Where $\text{AvgPool}(\cdot)$ performs a moving average to smooth periodic fluctuations. $\text{Padding}(\cdot)$ copies the last time step to preserve sequence length[^src-dpgnet].

**Why decompose?** Complex temporal patterns in spatiotemporal data are driven by intertwined trend and seasonal components[^src-autoformer]. Separating them simplifies the modeling task — the model can learn specialized representations for each component type[^src-dpgnet].

### Stage 3: Pattern-Specific Feature Extraction

**Trend features** — processed via TCN (two stacked dilated convolution layers)[^src-dpgnet]:
$$\hat{H}^i_T = \text{TCN}(H^i_T)$$

**Seasonal features** — processed via a frequency-domain approach inspired by FilterNet[^src-dpgnet]:
$$R^i, I^i = \text{FFT}(H^i_S),\quad \hat{H}^i_S = \text{iFFT}(\text{Linear}(R^i), \text{Linear}(I^i))$$

The Fourier transform decomposes seasonal components into real ($R^i$) and imaginary ($I^i$) frequency components, which are then linearly transformed and reconstructed via inverse FFT[^src-dpgnet].

## Pattern-Specific Graph Construction

After feature extraction, ASL builds separate adjacency matrices for trend and seasonal patterns at each scale[^src-dpgnet]:

$$\tilde{A}^i = \sigma\left(\beta_1 (\sum_l \hat{H}^i_{:,l,:}) M^i + \beta_2 (\sum_l \hat{H}^i_{:,l,:})(\sum_l \hat{H}^i_{:,l,:})^\top\right)$$

Where $\tilde{A}^i \in \mathbb{R}^{N \times N}$, $M^i \in \mathbb{R}^{N \times h}$, and $\beta_1, \beta_2 \in \mathbb{R}$ are trainable[^src-dpgnet]. This produces:
- $\tilde{A}^i_S$ — seasonal relationships (nodes with similar periodic patterns are connected)
- $\tilde{A}^i_T$ — trend relationships (nodes with similar macro-trend behavior are connected)

## Information Aggregation via GCN

At each scale, three adjacency matrices guide information flow[^src-dpgnet]:

$$Z^i_T = \text{Linear}([\text{GCN}(H^i_T, \tilde{A}^i_T), \text{GCN}(H^i, A_L)])$$
$$Z^i_S = \text{Linear}([\text{GCN}(H^i_S, \tilde{A}^i_S), \text{GCN}(H^i, A_L)])$$

Where $A_L$ is the dynamic adjacency from [[adaptive-graph-learner|AGL]]. The GCN uses $\hat{A} = \tilde{D}^{-\frac{1}{2}}(A + I_N)\tilde{D}^{-\frac{1}{2}}$ with a learnable weight $W \in \mathbb{R}^{h \times h}$[^src-dpgnet].

## Multi-Scale Fusion

Different fusion strategies are used for different pattern types[^src-dpgnet]:

**Bottom-up (seasonal, fine→coarse)**: $Z^i_S = Z^i_S + \text{Linear}(Z^{i-1}_S)$ for $i = 1 \to m$

Detailed seasonal information from fine scales enriches coarse-scale seasonal modeling[^src-dpgnet].

**Top-down (trend, coarse→fine)**: $Z^i_T = Z^i_T + \text{Linear}(Z^{i+1}_T)$ for $i = m-1 \to 0$

Macro-level trend knowledge from coarser scales guides finer-scale trend prediction. Detailed trend information may introduce noise when modeling macro trends[^src-dpgnet].

**Final fusion**: $Z^i = Z^i_T + Z^i_S$

## Relationship to Prior Work

| Method | Decomposition | Multi-Scale | Pattern-Specific Graph |
|--------|:---:|:---:|:---:|
| [[autoformer|Autoformer]] | ✓ | ✗ | ✗ |
| [[timemixer|TimeMixer]] | ✓ | ✓ | ✗ |
| [[dst-mamba|DST-Mamba]] | ✓ | ✓ (trend only) | ✗ |
| **ASL (DPGNet)** | ✓ | ✓ | ✓ |

ASL is the first module to combine all three dimensions — temporal decomposition, multi-scale processing, and pattern-specific graph construction — within a unified spatiotemporal architecture[^src-dpgnet].

## Ablation Significance

Removing seasonal features degrades performance more than removing trend features, confirming that seasonal patterns carry more discriminative spatial information in spatiotemporal data[^src-dpgnet].

## Limitations

- Multi-scale parameter m is manually set — no automatic selection mechanism[^src-dpgnet]
- Single-source evidence (under review)
- The FFT approach assumes stationary frequency components, which may not hold for rapidly changing traffic conditions

[^src-dpgnet]: [[source-dpgnet]]
[^src-autoformer]: [[source-autoformer]]
