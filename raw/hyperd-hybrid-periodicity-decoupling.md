# HyperD: Hybrid Periodicity Decoupling Framework for Traffic Forecasting

**Authors:** Hao Wen, Nan Feng (Jilin University)

**arXiv:** 2511.09275 (2025)

---

## Abstract

Traffic forecasting is crucial for intelligent transportation systems but remains challenging due to complex spatio-temporal dynamics. Existing methods primarily model either short-term local patterns or long-term global dependencies but rarely handle both concurrently with explicit periodicity decomposition. This paper proposes HyperD, a Hybrid Periodicity Decoupling framework that explicitly disentangles short-term and long-term periodicity from traffic signals. HyperD comprises a **Frequency-Aware Residual Representation (FR) module** that decomposes traffic signals via Fourier transform and learns periodic embeddings coupled with residual patterns, a **Spatial-Temporal Attentive Encoder (STAE)** with dual encoders (STFE and FR-embedded STFE) to model short-term and long-term periodicity respectively, and a **two-stage coarse-to-fine decoder (fDMLP, decoupled MLP)** that gradually recovers fine-grained predictions while filtering out trend information. A **Dual-View Alignment (DVA) loss** aligns representations across time scales and periodicity views. Experiments on four real-world traffic benchmarks (PEMS03, PEMS04, PEMS07, PEMS08) demonstrate that HyperD achieves state-of-the-art performance across all forecasting horizons, with particularly strong improvements on long-term predictions (e.g., 12-step ahead).

---

## 1. Introduction

Traffic forecasting predicts future traffic states (e.g., speed, flow, occupancy) based on historical observations. Accurate forecasting is essential for route planning, traffic control, and congestion management in Intelligent Transportation Systems (ITS). The core challenge lies in modeling complex spatial-temporal dependencies: **(1) spatial correlations** among sensors (road topology, proximity, similarity of patterns) and **(2) temporal dynamics** including trends, seasonality, and abrupt changes.

Traditional methods (ARIMA, VAR, SVR) fail to capture non-linear spatial correlations. Deep learning approaches have dominated recent research. Early DNN models (FC-LSTM, GRU) treat traffic as grid-based or sequence data, losing topological information. Graph Neural Networks (GNNs) combined with temporal models (TCN, RNN, attention) have become the standard paradigm: DCRNN (2018) introduced diffusion convolution for directed graphs; STGCN (2018) used pure convolutions; ASTGCN (2019) added attention mechanisms; STSGCN (2020) localized spatial-temporal graphs; STFGNN (2020) generated spatial-temporal fusion graphs. Transformer architectures (STTN, Traffic Transformer, PDFormer) further improved performance by leveraging global attention.

**Limitations of existing methods:**

- Most methods implicitly model periodicity without explicit decomposition, making it difficult to distinguish short-term (recent trends, abrupt changes) from long-term (daily/weekly rhythms) periodic patterns.
- Fourier-based approaches (e.g., FEDformer, FreTS) treat frequency components uniformly, failing to separate periodic vs. residual signals.
- Multi-scale temporal modeling exists (MTGNN, StemGNN), but typically lacks a principled alignment mechanism between different periodicity views.

**Key insight:** Traffic signals can be decomposed into **short-term periodicity** (recent patterns, e.g., last few hours), **long-term periodicity** (daily/weekly cycles), and **residual components** (trends, noise). Explicitly decoupling these components and modeling them with specialized architectures can improve forecasting accuracy, especially for long horizons.

## 2. Related Work

### 2.1 Spatial-Temporal Graph Neural Networks

The dominant approach for traffic forecasting. DCRNN (Li et al., 2018) introduced diffusion convolution with GRU-based sequence-to-sequence architecture. STGCN (Yu et al., 2018) used Chebyshev graph convolutions with gated temporal convolutions. GWNet (Wu et al., 2019) proposed adaptive adjacency matrices. AGCRN (Bai et al., 2020) learned node-specific patterns. STGODE (Fang et al., 2021) used neural ODEs for continuous spatial-temporal modeling. D2STGNN (Shao et al., 2022) applied diffusion and inherent graph convolutions.

### 2.2 Attention and Transformer Methods

STTN (Xu et al., 2020) proposed spatial-temporal Transformer with positional encoding. GMAN (Zheng et al., 2020) used spatial and temporal attention with gated fusion. PDFormer (Jiang et al., 2023) introduced propagation delay-aware graph construction. STAEformer (Liu et al., 2024) demonstrated that lightweight spatial-temporal attentive embeddings can outperform complex graph convolutions.

### 2.3 Frequency and Periodicity Methods

FEDformer (Zhou et al., 2022) introduced Fourier-enhanced attention for time series. FreTS (Yi et al., 2023) applied Fourier graph neural operators for traffic. StemGNN (Cao et al., 2020) modeled multi-scale temporal patterns in spectral domain. FormerTime (Zhang et al., 2024) proposed hierarchical multi-scale representations. These approaches use frequency transformations but do not explicitly decouple periodic and residual components.

### 2.4 Periodicity Decomposition

Autoformer (Wu et al., 2021) introduced progressive decomposition with auto-correlation. FEDformer used seasonal-trend decomposition in frequency domain. DLinear (Zeng et al., 2023) showed that simple linear models with decomposition can be competitive. However, explicit **hybrid** decomposition that separates short-term and long-term periodicity with dedicated modeling remains underexplored in traffic forecasting.

## 3. Problem Formulation

**Traffic network:** A graph G = (V, E, A) where V is the set of N sensors, E represents road connections, and A ∈ ℝ^{N×N} is the adjacency matrix. Input window of T historical steps produces X_{t-T+1:t} ∈ ℝ^{T×N×C} where C is the number of traffic features (typically 1, e.g., speed). The goal is to predict future T' steps: Ŷ ∈ ℝ^{T'×N×C}.

**Periodicity decomposition:** Traffic signal x ∈ ℝ^{T×N×C} is decomposed into:
- **Short-term periodicity** (P_short): captures intra-day and recent temporal dependencies (e.g., preceding 1–6 hours)
- **Long-term periodicity** (P_long): captures daily, weekly, and seasonal cycles
- **Residual** (R): captures remaining trends, noise, and aperiodic components

## 4. HyperD Architecture

HyperD consists of four components:

1. **Frequency-Aware Residual Representation (FR)** — decomposes input into periodic and residual components
2. **Spatial-Temporal Attentive Encoder (STAE)** — dual-pathway encoder for short-term and long-term periodicity
3. **Decoupled MLP Decoder (fDMLP)** — two-stage coarse-to-fine decoder
4. **Dual-View Alignment (DVA) Loss** — aligns representations across periodicity views

### 4.1 Frequency-Aware Residual Representation (FR)

The FR module transforms historical traffic data into the frequency domain via **Real Fast Fourier Transform (RFFT)**:

Let X ∈ ℝ^{T×N×C} be the input. Apply RFFT along the temporal dimension:

**F = RFFT(X, dim=temporal) ∈ ℂ^{K×N×C}** where K = ⌊T/2⌋ + 1

The frequency components are clustered into three bands using learned thresholds:
- **High-frequency band:** corresponds to rapid fluctuations, noise → **Residual component**
- **Mid-frequency band:** corresponds to short-term periodic patterns → **Short-term periodicity embedding (H_s)**
- **Low-frequency band:** corresponds to daily/weekly cycles → **Long-term periodicity embedding (H_l)**

The residual signal R is obtained via inverse RFFT (IRFFT) from high-frequency components: **R = IRFFT(F_high)**.

The short-term periodic embedding H_s and long-term periodic embedding H_l are learned representations in the frequency domain that will guide the dual-pathway encoder.

Key design: Unlike FreTS which treats all frequency components uniformly via graph convolution in frequency domain, FR explicitly **separates** frequency bands and generates distinct embeddings for short-term and long-term periodicity. This enables targeted modeling in subsequent encoder stages.

### 4.2 Spatial-Temporal Attentive Encoder (STAE)

STAE consists of two parallel pathways, each built on the STAEformer-style architecture (spatial-temporal embedding with attention):

#### Pathway 1: Short-Term Encoder (STFE)
Input: Original traffic signal X + Short-term periodic embedding H_s

This pathway focuses on **local temporal dynamics** (recent hours, abrupt changes). It uses a spatial-temporal attentive block where:
- **Temporal Embedding** captures time-of-day and day-of-week patterns
- **Spatial Embedding** captures sensor identity
- **Spatial-Temporal Attention** models interactions between all (N × T) tokens

Output: Short-term representation **Z_s ∈ ℝ^{T'×N×D}**

#### Pathway 2: Long-Term Encoder (FR-STFE)
Input: Residual signal R (from FR module) + Long-term periodic embedding H_l

This pathway focuses on **global cyclical patterns** (daily/weekly rhythms). By inputting the residual signal (with short-term fluctuations removed via FR), the long-term encoder can focus on modeling broader periodic structures without interference from local noise.

The architecture mirrors Pathway 1 but operates on the residual signal guided by long-term frequency embeddings.

Output: Long-term representation **Z_l ∈ ℝ^{T'×N×D}**

**Design rationale:** The dual-pathway design with explicit input separation (original signal → short-term encoder, residual → long-term encoder) ensures that each encoder specializes in its target periodicity scale, avoiding the information mixing problem present in single-stream architectures.

### 4.3 Decoupled MLP Decoder (fDMLP)

The decoder operates in two stages, inspired by DLinear's decomposition philosophy but adapted for multi-scale periodicity fusion:

#### Stage 1: De-trending MLP (DMLP)
- Projects the concatenated representations [Z_s || Z_l] to a coarse prediction Ŷ_coarse
- Simultaneously extracts a **trend component** T that captures monotonic directional bias in the predictions
- The trend is **removed** from the coarse prediction to prevent over-smoothing toward the mean

#### Stage 2: Fine-grained MLP (fDMLP)
- Takes the de-trended coarse prediction as input
- Applies channel-wise MLP layers with residual connections to recover fine-grained spatial-temporal details
- Produces the final prediction Ŷ

**Decoupling strategy:** By explicitly extracting and removing trend information, fDMLP prevents the decoder from collapsing to a simple averaging solution. The two-stage process (coarse → fine) mirrors the multi-scale philosophy of the entire HyperD framework.

### 4.4 Dual-View Alignment (DVA) Loss

The total loss function combines prediction error with representation alignment:

**L = L_mae(Ŷ, Y) + λ * L_DVA**

Where:
- **L_mae(Ŷ, Y)** = MAE between prediction and ground truth
- **L_DVA** is the Dual-View Alignment loss:

**L_DVA = α * L_temporal + β * L_periodic**

- **L_temporal:** Aligns short-term and long-term representations across time scales. Uses **Soft-DTW (Differentiable Dynamic Time Warping)** to measure temporal alignment between Z_s and Z_l, promoting consistency in how the two encoders represent the same underlying traffic state at different time scales.

- **L_periodic:** Aligns the short-term and long-term **periodicity embeddings** (H_s and H_l) in the frequency domain. Uses a cosine similarity-based loss to ensure that the embeddings capture complementary rather than redundant periodic information.

**Why alignment matters:** Without explicit alignment, the dual-pathway encoder may learn representations that contradict each other for the same time step, degrading overall prediction accuracy. The DVA loss acts as a regularizer that encourages the two pathways to produce **consistent yet complementary** representations.

## 5. Experiments

### 5.1 Datasets
Four California highway datasets (PeMS, Caltrans Performance Measurement System):
- **PEMS03:** 358 sensors, 26,208 time steps
- **PEMS04:** 307 sensors, 16,992 time steps
- **PEMS07:** 883 sensors, 28,224 time steps
- **PEMS08:** 170 sensors, 17,856 time steps

Input: 12 historical steps (1 hour). Predict: 12 future steps (1 hour).

### 5.2 Baselines
Compared against 15 methods including:
- Classical: HA, VAR
- RNN-based: DCRNN, AGCRN
- GNN-based: STGCN, GWNet, STGODE, D2STGNN, DGCRN
- Attention-based: STTN, GMAN, PDFormer, STAEformer
- Frequency-based: FEDformer, FreTS, StemGNN
- MLP-based: DLinear, STID

### 5.3 Main Results

HyperD achieves **state-of-the-art MAE, RMSE, and MAPE** on all four datasets. Key findings:

1. **Overall dominance:** HyperD outperforms all 15 baselines on all 4 datasets across all 3 metrics (MAE, RMSE, MAPE), with average improvements of 2–5% over the second-best method (typically STAEformer or D2STGNN).

2. **Long-term advantage:** The performance gap widens at longer horizons (9-step, 12-step), demonstrating that explicit hybrid periodicity decoupling is particularly beneficial for long-range forecasting where periodic patterns dominate.

3. **Ablation studies confirm each component's contribution:**
   - Removing FR module → +3.2% MAE increase
   - Removing dual-pathway (single encoder) → +2.1% MAE increase
   - Removing DVA loss → +1.5% MAE increase
   - Removing fDMLP (simple MLP decoder) → +2.8% MAE increase

4. **Frequency band analysis:** HyperD is robust to the choice of frequency band cutoffs; performance degrades gracefully even with manually set (non-learned) thresholds.

5. **Efficiency:** Despite the dual-pathway architecture, HyperD maintains competitive training and inference time (roughly 1.3× STAEformer's cost), significantly more efficient than heavy Transformer-based methods like GMAN or PDFormer.

### 5.4 Supplementary Experiments
- **Robustness to missing data:** HyperD degrades gracefully under random sensor dropout (5%–30%).
- **Transfer learning:** Pre-training on PEMS04 and fine-tuning on PEMS08 yields positive transfer.
- **Visualization:** The FR module's learned frequency cutoffs align with interpretable periods (daily cycle at 288 steps/day for 5-min data, weekly cycle at 2016 steps/week).

## 6. Conclusion

HyperD proposes a principled framework for traffic forecasting that explicitly decouples short-term and long-term periodicity using frequency-domain decomposition (FR), models each scale with specialized encoders (STAE), and aligns representations across periodicity views (DVA loss). The two-stage fDMLP decoder prevents trend-induced smoothing. State-of-the-art results on four benchmarks with controlled computational cost demonstrate the effectiveness of hybrid periodicity decoupling.

**Limitations and future work:**
- Current FR module uses fixed learned cutoffs per dataset; adaptive per-node or per-time-window cutoffs could improve personalization.
- Spatial graph structure is used only in baselines but not fully integrated into HyperD's current design (STAE uses spatial embedding but not explicit graph convolutions).
- External factors (weather, holidays, accidents) are not incorporated; multi-modal fusion is left for future work.

---

**Source:** arXiv preprint 2511.09275 (2025).
