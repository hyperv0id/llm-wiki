---
title: "Traffic Forecasting"
type: concept
tags:
  - time-series
  - spatial-temporal
  - intelligent-transportation
created: 2026-04-27
last_updated: 2026-05-13
source_count: 16
confidence: high
status: active
---

# Traffic Forecasting

Traffic forecasting is the task of predicting future traffic states (speed, flow, occupancy) based on historical observations from sensor networks[^src-hyperd-hybrid-periodicity-decoupling]. It is a core component of Intelligent Transportation Systems (ITS), enabling route planning, traffic control, and congestion management.

## Core Challenges

Two main dependencies must be modeled simultaneously[^src-hyperd-hybrid-periodicity-decoupling]:

- **Spatial correlations** — dependencies among sensors, including road topology, geographic proximity, and similarity of usage patterns
- **Temporal dynamics** — trends, seasonality, daily/weekly rhythms, and abrupt changes over time

## Methods

### Classical
ARIMA, VAR, and SVR fail to capture non-linear spatial correlations[^src-hyperd-hybrid-periodicity-decoupling].

### Deep Graph-Based
The dominant paradigm since DCRNN (2018): Graph Neural Networks (GNNs) combined with temporal models (TCN, RNN, attention). Key milestones include STGCN (2018), ASTGCN (2019), GWNet (2019), STSGCN (2020), STFGNN (2020), D2STGNN (2022), and STGODE (2021)[^src-hyperd-hybrid-periodicity-decoupling].

### Transformer-Based
STTN (2020), GMAN (2020), PDFormer (2023), and STAEformer (2024) use attention mechanisms to model global spatial-temporal dependencies[^src-hyperd-hybrid-periodicity-decoupling].

### Frequency-Domain
[[fedformer|FEDformer]] (ICML 2022) applies Fourier and Wavelet transforms in its [[frequency-enhanced-block|FEB]]/[[frequency-enhanced-attention|FEA]] blocks and [[moe-decomposition|MOEDecomp]] for adaptive seasonal-trend decomposition, but treats frequency components uniformly without separating periodic from residual signals. FreTS (NeurIPS 2023) and StemGNN (2020) follow similar uniform processing in the frequency domain[^src-hyperd-hybrid-periodicity-decoupling].

### Periodicity-Decoupled
[[hyperd|HyperD]] (2025) explicitly decouples short-term and long-term periodicity via hybrid frequency-domain decomposition[^src-hyperd-hybrid-periodicity-decoupling].

### Accident-Aware
Traditional models assume stationary traffic patterns but fail during accidents which create non-stationary perturbations with directional shockwaves. ConFormer (KDD 2026) addresses this through accident-aware graph propagation and Guided Layer Normalization (GLN), achieving up to 10.7% improvement in accident scenarios[^src-conformer].

### Incident-Guided
Extending beyond accidents, [[igstgnn|IGSTGNN]] (KDD 2026) explicitly models the impact of broader non-recurrent incidents (accidents, weather, hazards, breakdowns, etc.) through two plug-and-play modules: [[incident-context-spatial-fusion|ICSF]] captures heterogeneous spatial influence via attention + spatial relationship tensor, and [[temporal-incident-impact-decay|TIID]] models temporal decay via Gaussian function. Achieves 5.65% average MAE improvement on Alameda dataset[^src-incident-guided-st-forecasting].

### Large-Scale Long-Horizon
FaST (KDD 2026) addresses computational bottlenecks in large-scale graphs (8,600+ nodes) with long-horizon predictions (672 steps = 1 week) using [[adaptive-graph-agent-attention|AGA-Att]] for O(N·a) spatial complexity and [[mixture-of-experts|Dense MoE]] for efficient feature extraction. Achieves 4.4%-18.4% MAE improvement over SOTA with 1.3x-2.2x faster inference[^src-fast-long-horizon-forecasting].

### Pre-training & Masked Autoencoder

[[std-mae|STD-MAE]] (IJCAI-2024) proposes a spatial-temporal-decoupled masked pre-training framework for traffic forecasting. It addresses a critical limitation of end-to-end models: short input horizons (typically 12 steps = 1 hour) that cause a **[[spatiotemporal-mirage|spatiotemporal mirage]]** — similar input sequences leading to dissimilar future values and vice versa. STD-MAE pre-trains two decoupled masked autoencoders (S-MAE for spatial, T-MAE for temporal) on long sequences (e.g., 864 steps = 3 days), learning clear spatiotemporal heterogeneity representations that can enhance any downstream predictor. On six PEMS benchmarks, STD-MAE achieves SOTA performance with 22.6%-72.5% faster pre-training than comparable methods[^src-2312-00516-std-mae].

### Regularized Adaptive Graph Convolution
[[ragc|RAGC]] (arXiv 2026) tackles two limitations of adaptive graph learning for large-scale networks: O(N²) graph convolution complexity and lack of node embedding regularization. It proposes [[efficient-cosine-operator|ECO]] for O(N) graph convolution via cosine similarity decomposition, and integrates [[stochastic-shared-embedding|SSE]] with adaptive graph convolution through a [[residual-difference-mechanism|residual difference mechanism]] that suppresses SSE-induced noise while retaining regularization benefits. On four LargeST datasets (716–8,600 nodes), RAGC consistently achieves the best prediction accuracy with competitive training/inference speed[^src-ragc-efficient-traffic-forecasting].

### Flow Matching / Frequency-Domain
[[freqflow-ts|FrèqFlow/SpectFlow]] (NeurIPS 2025) is the first framework to apply conditional flow matching in the frequency domain for deterministic MTS traffic forecasting. With only 89k parameters, it uses a complex-valued linear layer for frequency interpolation and a flow matching head for residual learning. On Brussels, PEMS08, and PEMS04 datasets, it achieves 7% average RMSE improvement over Moirai-MoE and diffusion baselines (GCRDD, DiffSTG, PriSTI, SpecSTG)[^src-2511-16426].

### Probabilistic / Diffusion-Based
Deterministic models only output point estimates, lacking uncertainty quantification. Probabilistic methods address this gap:
- **[[specstg|SpecSTG]]** (arXiv 2024) is the first spectral diffusion framework for probabilistic STG forecasting. It generates the graph Fourier representation of future time series instead of raw sequences, naturally embedding spatial dependencies into the diffusion process. With [[fast-spectral-graph-convolution|Fast Spectral Graph Convolution]] reducing graph convolution complexity from $O(N^2)$ to $O(N)$, SpecSTG achieves up to 8% RMSE improvement and 3.33× training speedup over GCRDD (the most efficient existing diffusion method)[^src-2401-08119-specstg].
- Other diffusion methods (TimeGrad, GCRDD, DiffSTG, PriSTI) operate in the original domain and treat sensors independently during probabilistic generation, limiting spatial information usage[^src-2401-08119-specstg].

### Spatial-Temporal Imputation
时空数据填补与预测紧密相关——填补缺失值是许多预测管道的前置步骤。GSLI（AAAI 2025）提出多尺度图结构学习框架，通过节点尺度学习解决特征异质性问题，通过特征尺度学习捕获跨特征空间依赖，在 6 个真实数据集上取得最优填补性能[^src-yang-gsli-2025]。ImputeFormer（KDD 2024）则通过低秩归纳偏置实现线性复杂度的 Transformer 填补[^src-2312-01728]。CoFILL（arXiv 2025）使用条件扩散模型进行时空填补[^src-cofill-spatiotemporal-imputation]。

### Mamba / SSM-Based
Mamba 的选择性状态空间模型也被应用于交通预测。Han et al. (NeurIPS 2024) 的统一框架揭示了 Mamba 的遗忘门 $\widetilde{A}_i$ 在交通场景中对应于空间衰减模式——附近传感器的相关性更强，这与遗忘门的局部偏置特性一致。该框架表明，交通特定的位置编码（如道路距离、转向关系）可替代遗忘门的循环计算，在保持并行性的同时捕获局部空间结构[^src-demystify-mamba-linear-attention-2024]。

### Foundation Model
[[most|MoST]] (KDD 2026) is the first multi-modality spatio-temporal foundation model for traffic prediction, enabling zero-shot cross-city generalization using satellite imagery, POI text, and location as background context[^src-most]. Unlike task-specific models, MoST uses an SNR-based [[multi-modality-refinement|modality selector]] to adaptively filter noisy modalities and [[multi-modality-guided-spatial-expert|multi-modality-guided spatial experts]] to capture region-specific local spatial patterns[^src-most]. Its zero-shot performance surpasses most full-shot end-to-end models and the OpenCity foundation model across five datasets[^src-most].

## Key Models

Several influential models span the development of traffic and spatial-temporal forecasting:

- **[[source-st-resnet|ST-ResNet]]** (AAAI 2017) — one of the first deep learning approaches for citywide crowd flow prediction, using residual convolutional units to model spatial-temporal dependencies[^src-st-resnet].
- **[[source-astgcn|ASTGCN]]** (AAAI 2019) — combines attention mechanisms with graph convolution to jointly capture spatial and temporal patterns in traffic flow[^src-astgcn].
- **[[source-prnet|PRNet]]** — introduces periodic residual learning to explicitly model recurring temporal patterns in crowd flow forecasting[^src-prnet].
- **[[source-penguin|PENGUIN]]** (AISTATS 2026) — proposes periodic-nested group attention for long-sequence time-series forecasting, with applicability to traffic domains[^src-penguin].

For a comprehensive overview of deep learning methods for time series, including traffic forecasting, the [[source-deep-time-series-survey|TSLib survey]] provides systematic benchmarking across multiple domains[^src-deep-time-series-survey].

## Benchmarks

The standard benchmarks are the PeMS (Caltrans Performance Measurement System) datasets from California highways: PEMS03, PEMS04, PEMS07, PEMS08. Standard setup: 12 input steps (1 hour) → 12 output steps (1 hour)[^src-hyperd-hybrid-periodicity-decoupling].

The XTraffic benchmark provides incident-aligned traffic datasets for California (2023), with 521-990 sensor nodes and 5,587-18,700 incident records[^src-incident-guided-st-forecasting].

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-st-resnet]: [[source-st-resnet]]
[^src-astgcn]: [[source-astgcn]]
[^src-prnet]: [[source-prnet]]
[^src-penguin]: [[source-penguin]]
[^src-deep-time-series-survey]: [[source-deep-time-series-survey]]
[^src-conformer]: [[source-conformer]]
[^src-fast-long-horizon-forecasting]: [[source-fast-long-horizon-forecasting]]
[^src-incident-guided-st-forecasting]: [[source-incident-guided-st-forecasting]]
[^src-most]: [[source-most]]
[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
[^src-2401-08119-specstg]: [[source-2401-08119-specstg]]
[^src-yang-gsli-2025]: [[source-yang-gsli-2025]]
[^src-demystify-mamba-linear-attention-2024]: [[source-demystify-mamba-linear-attention-2024]]
[^src-2312-00516-std-mae]: [[source-2312-00516-std-mae]]
[^src-2511-16426]: [[source-2511-16426]]
