---
title: "Traffic Forecasting"
type: concept
tags:
  - time-series
  - spatial-temporal
  - intelligent-transportation
created: 2026-04-27
last_updated: 2026-06-01
source_count: 27
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
The dominant paradigm since [[dcrnn|DCRNN]] (2018): Graph Neural Networks (GNNs) combined with temporal models (TCN, RNN, attention). Key milestones include [[dcrnn|DCRNN]], [[stgcn|STGCN]] (2018), [[astgcn|ASTGCN]] (2019), [[gwnet|GWNet]] (2019), STSGCN (2020), STFGNN (2020), D2STGNN (2022), and STGODE (2021)[^src-hyperd-hybrid-periodicity-decoupling].

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

Before STD-MAE, [[gpt-st|GPT-ST]] (NeurIPS 2023) pioneered the MAE pre-training paradigm for spatio-temporal graphs. GPT-ST is a plug-and-play pre-training framework that uses a customized temporal hypergraph encoder, a hierarchical spatial capsule clustering network, and a cluster-aware adaptive mask strategy. It seamlessly integrates with 13 diverse downstream STGNN baselines (STGCN, GWN, MTGNN, MSDR, etc.) without modifying their architectures, achieving universal performance improvement across 4 datasets[^src-gpt-st].

[[std-mae|STD-MAE]] (IJCAI-2024) proposes a spatial-temporal-decoupled masked pre-training framework for traffic forecasting. It addresses a critical limitation of end-to-end models: short input horizons (typically 12 steps = 1 hour) that cause a **[[spatiotemporal-mirage|spatiotemporal mirage]]** — similar input sequences leading to dissimilar future values and vice versa. STD-MAE pre-trains two decoupled masked autoencoders (S-MAE for spatial, T-MAE for temporal) on long sequences (e.g., 864 steps = 3 days), learning clear spatiotemporal heterogeneity representations that can enhance any downstream predictor. On six PEMS benchmarks, STD-MAE achieves SOTA performance with 22.6%-72.5% faster pre-training than comparable methods[^src-2312-00516-std-mae].

### Regularized Adaptive Graph Convolution
[[ragc|RAGC]] (arXiv 2026) tackles two limitations of adaptive graph learning for large-scale networks: O(N²) graph convolution complexity and lack of node embedding regularization. It proposes [[efficient-cosine-operator|ECO]] for O(N) graph convolution via cosine similarity decomposition, and integrates [[stochastic-shared-embedding|SSE]] with adaptive graph convolution through a [[residual-difference-mechanism|residual difference mechanism]] that suppresses SSE-induced noise while retaining regularization benefits. On four LargeST datasets (716–8,600 nodes), RAGC consistently achieves the best prediction accuracy with competitive training/inference speed[^src-ragc-efficient-traffic-forecasting].

### Flow Matching / Frequency-Domain
[[freqflow-ts|FrèqFlow/SpectFlow]] (NeurIPS 2025) is the first framework to apply conditional flow matching in the frequency domain for deterministic MTS traffic forecasting. With only 89k parameters, it uses a complex-valued linear layer for frequency interpolation and a flow matching head for residual learning. On Brussels, PEMS08, and PEMS04 datasets, it achieves 7% average RMSE improvement over Moirai-MoE and diffusion baselines ([[d3vae|GCRDD]], [[diffstg|DiffSTG]], [[pristi|PriSTI]], SpecSTG)[^src-2511-16426].

### Probabilistic / Diffusion-Based
Deterministic models only output point estimates, lacking uncertainty quantification. Probabilistic methods address this gap:
- **[[specstg|SpecSTG]]** (arXiv 2024) is the first spectral diffusion framework for probabilistic STG forecasting. It generates the graph Fourier representation of future time series instead of raw sequences, naturally embedding spatial dependencies into the diffusion process. With [[fast-spectral-graph-convolution|Fast Spectral Graph Convolution]] reducing graph convolution complexity from $O(N^2)$ to $O(N)$, SpecSTG achieves up to 8% RMSE improvement and 3.33× training speedup over [[d3vae|GCRDD]] (the most efficient existing diffusion method)[^src-2401-08119-specstg].
- **[[ustd|USTD]]** (SIGSPATIAL 2024) unifies forecasting and kriging into a single diffusion framework. Key innovation: pre-trained GWNet-style encoder (with graph sampling + 75% masking) separately from task-specific gated attention denoisers (TGA for forecasting, SGA for kriging). This decoupled training strategy enables USTD to become the first diffusion STG model to surpass deterministic baselines on forecasting (CRPS ↓12% on PEMS-BAY). Other diffusion methods ([[timegrad|TimeGrad]], [[d3vae|GCRDD]], [[diffstg|DiffSTG]], [[pristi|PriSTI]]) operate in the original domain and treat sensors independently during probabilistic generation, limiting spatial information usage[^src-2401-08119-specstg].

### Spatial-Temporal Imputation
时空数据填补与预测紧密相关——填补缺失值是许多预测管道的前置步骤。GSLI（AAAI 2025）提出多尺度图结构学习框架，通过节点尺度学习解决特征异质性问题，通过特征尺度学习捕获跨特征空间依赖，在 6 个真实数据集上取得最优填补性能[^src-yang-gsli-2025]。ImputeFormer（KDD 2024）则通过低秩归纳偏置实现线性复杂度的 Transformer 填补[^src-2312-01728]。CoFILL（arXiv 2025）使用条件扩散模型进行时空填补[^src-cofill-spatiotemporal-imputation]。

### Mamba / SSM-Based
Mamba 的选择性状态空间模型也被应用于交通预测。Han et al. (NeurIPS 2024) 的统一框架揭示了 Mamba 的遗忘门 $\widetilde{A}_i$ 在交通场景中对应于空间衰减模式——附近传感器的相关性更强，这与遗忘门的局部偏置特性一致。该框架表明，交通特定的位置编码（如道路距离、转向关系）可替代遗忘门的循环计算，在保持并行性的同时捕获局部空间结构[^src-demystify-mamba-linear-attention-2024]。

### Foundation Model

**[[unist|UniST]]** (KDD 2024) is the first one-for-all spatio-temporal foundation model, using MAE pre-training with four complementary masking strategies and knowledge-guided memory-based prompt learning. A single 6.71M-parameter model covers 20+ datasets across multiple cities and domains with zero-shot prediction surpassing few-shot baselines — e.g., Crowd zero-shot RMSE 14.67 vs ACFM 1%-shot 21.17[^src-unist].

**[[urbanpg|UrbanPG]]** (AAAI 2026) approaches foundation model capability from a different angle: prompt-backbone decoupling with STCA linear attention (O(N·d²)). Unifies three paradigms — large-scale prediction (SOTA on CA 8600 nodes), few-shot generalization (fine-tune only prompts), and continual learning (expand spatial prompts, zero forgetting). Key trade-off: cannot support multi-task parallel training like UrbanFM[^src-urbanpg].

**[[uniflow|UniFlow]]** (arXiv 2024) is the first foundation model to unify both grid-based and graph-based flow prediction. Using Transformer + ST-MRA (4 learnable memory pools for time/frequency/spatial patterns), it achieves SOTA on 9 datasets (6 grid + 3 graph, >10K nodes) with 9.1% average RMSE improvement over baselines. Zero-shot performance surpasses most trained baselines. Same Tsinghua FIB Lab as UrbanDiT and UniST[^src-uniflow].

**[[urbangpt|UrbanGPT]]** (KDD 2024) is the first spatio-temporal LLM, using Vicuna-7b + multi-level gated dilated convolution encoder + instruction-tuning paradigm. It replaces graph-based spatial modeling with textual POI descriptions that the LLM interprets semantically, enabling cross-region and cross-city zero-shot prediction — NYC-taxi inflow MAE=6.16 vs best baseline ASTGCN 9.75 (↓36.8%). Key limitation: 7B parameters, 174s inference per sensor, impractical for large-scale deployment[^src-urbangpt].

[[most|MoST]] (KDD 2026) is the first multi-modality spatio-temporal foundation model for traffic prediction, enabling zero-shot cross-city generalization using satellite imagery, POI text, and location as background context[^src-most]. Unlike task-specific models, MoST uses an SNR-based [[multi-modality-refinement|modality selector]] to adaptively filter noisy modalities and [[multi-modality-guided-spatial-expert|multi-modality-guided spatial experts]] to capture region-specific local spatial patterns[^src-most]. Its zero-shot performance surpasses most full-shot end-to-end models and the OpenCity foundation model across five datasets[^src-most].

[[opencity|OpenCity]] (2024) is an early pure-numerical ST foundation model using instance normalization + patch embedding + TimeShift Transformer (PTTM+DTP dual attention) + GCN spatial aggregation. Pre-trained on 21 datasets (151M observations), it achieves zero-shot performance surpassing full-shot baselines on 4/6 test datasets including unseen data types (NYC-BIKE), with <3s inference latency per city[^src-opencity].

[[urbanfm|UrbanFM]] (arXiv 2026) advances the foundation model paradigm by adopting scaling as the central design principle. It introduces WorldST (100+ cities, 1B+ data points — 33-145× more than predecessors), MiniST (KD-Tree clustering to unify sensor/grid tokens), and a minimalist factorized attention architecture with ST-RoPE. UrbanFM achieves 39-70.2% zero-shot improvement over existing foundation models, surpasses full-shot expert models, and supports imputation without imputation training[^src-urbanfm].

[[factost|FactoST]] (NeurIPS 2025 / arXiv 2026) proposes a fundamentally different approach — the Pattern Factorization Hypothesis — arguing that effective ST generalization requires decoupling universal temporal dynamics from domain-specific spatial contexts. FactoST-v2 factorizes training into two stages: UTP (encoder-only Transformer, 11B+ time points, 8 domains) learns graph-agnostic temporal patterns with linear O(N) complexity, then STA (lightweight adapter with STMF+STF+DSPA+CMR) injects spatial awareness. Achieves SOTA on few-shot/full-shot/zero-shot across all PEMS benchmarks, with 4.4M default parameters and 11s inference. Same HKUST(GZ) group as UrbanFM (Yuxuan Liang as corresponding author)[^src-factost].

**[[bigcity|BIGCity]]** (arXiv 2024) is the first MTMD (Multi-Task, Multi-Data modality) spatio-temporal model, unifying individual-level trajectory data and population-level traffic state data within a single GPT-2+LoRA framework with task-oriented prompts. It covers 8 heterogeneous tasks across 3 cities, surpassing 18 independently trained baselines. BIGCity represents a fundamental expansion of ST foundation model scope — from traffic-only (MTSD) to trajectory+traffic (MTMD)[^src-bigcity].

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
[^src-uniflow]: [[source-uniflow]]
[^src-2312-01728]: [[source-2312-01728]]
[^src-cofill-spatiotemporal-imputation]: [[source-cofill-spatiotemporal-imputation]]
[^src-gpt-st]: [[source-gpt-st]]
[^src-urbangpt]: [[source-urbangpt]]
[^src-opencity]: [[source-opencity]]
[^src-unist]: [[source-unist]]
[^src-urbanfm]: [[source-urbanfm]]
[^src-urbanpg]: [[source-urbanpg]]
[^src-factost]: [[source-factost]]
[^src-bigcity]: [[source-bigcity]]
