---
title: "Traffic Forecasting"
type: concept
tags:
  - time-series
  - spatial-temporal
  - intelligent-transportation
created: 2026-04-27
last_updated: 2026-07-16
source_count: 46
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

### Road Network Representation Learning
While traffic forecasting predicts future values at sensor locations, **road network representation learning** is a related but distinct task: learning reusable vector embeddings for road segments that capture both spatial structure and functional semantics. These embeddings generalize across diverse downstream tasks (next-location prediction, label classification, destination prediction, route planning) rather than being task-specific[^src-hifinet].

[[hifinet|HiFiNet]] (AAAI 2026) introduces hierarchical frequency-decomposition GNNs for this task, constructing a three-level graph (segment→locality→region) where learnable cross-attention assignment matrices enable localized graph frequency decomposition. The model separates low-frequency (smooth global commuting patterns) and high-frequency (local city-center variations) signals, processes them through a [[topology-aware-graph-transformer|Topology-Aware Graph Transformer]], and fuses the enriched components. On Beijing/Chengdu/Xi'an datasets, HiFiNet achieves SOTA across all four tasks, with ablation confirming both hierarchy and frequency decomposition as critical[^src-hifinet]. See [[road-network-representation-learning]] and [[graph-frequency-decomposition]] for detailed treatment.

### Flow Matching / Frequency-Domain
[[freqflow-ts|FrèqFlow/SpectFlow]] (NeurIPS 2025) is the first framework to apply conditional flow matching in the frequency domain for deterministic MTS traffic forecasting. With only 89k parameters, it uses a complex-valued linear layer for frequency interpolation and a flow matching head for residual learning. On Brussels, PEMS08, and PEMS04 datasets, it achieves 7% average RMSE improvement over Moirai-MoE and diffusion baselines ([[d3vae|GCRDD]], [[diffstg|DiffSTG]], [[pristi|PriSTI]], SpecSTG)[^src-2511-16426].

### Dynamic Graph with Meta-Parameters
[[metadg|MetaDG]] (AAAI 2026) proposes a new paradigm: extending dynamics from only generating dynamic adjacency matrices to also generating **meta-parameters** (node-wise model weights) at each time step. This pushes the field from ST-isolated modeling (where spatial and temporal dimensions use separate base model structures) toward [[st-unification|ST-unification]]. Key components: (1) Dynamic Node Generation (DNG) — time-gated fusion of static embeddings and hidden states; (2) Spatio-Temporal Correlation Enhancement (STCE) — spatial cross-attention + temporal smoothing; (3) [[dynamic-graph-qualification|Dynamic Graph Qualification (DGQ)]] — qualifying edge reliability via cross-time-step similarity. SOTA on PEMS03/04/07/08 across all metrics, with particular advantage in long-term prediction[^src-metadg].

### Probabilistic / Diffusion-Based
Deterministic models only output point estimates, lacking uncertainty quantification. Probabilistic methods address this gap:
- **[[specstg|SpecSTG]]** (arXiv 2024) is the first spectral diffusion framework for probabilistic STG forecasting. It generates the graph Fourier representation of future time series instead of raw sequences, naturally embedding spatial dependencies into the diffusion process. With [[fast-spectral-graph-convolution|Fast Spectral Graph Convolution]] reducing graph convolution complexity from $O(N^2)$ to $O(N)$, SpecSTG achieves up to 8% RMSE improvement and 3.33× training speedup over [[d3vae|GCRDD]] (the most efficient existing diffusion method)[^src-2401-08119-specstg].
- **[[ustd|USTD]]** (SIGSPATIAL 2024) unifies forecasting and kriging into a single diffusion framework. Key innovation: pre-trained GWNet-style encoder (with graph sampling + 75% masking) separately from task-specific gated attention denoisers (TGA for forecasting, SGA for kriging). This decoupled training strategy enables USTD to become the first diffusion STG model to surpass deterministic baselines on forecasting (CRPS ↓12% on PEMS-BAY). Other diffusion methods ([[timegrad|TimeGrad]], [[d3vae|GCRDD]], [[diffstg|DiffSTG]], [[pristi|PriSTI]]) operate in the original domain and treat sensors independently during probabilistic generation, limiting spatial information usage[^src-2401-08119-specstg].

### Spatial-Temporal Imputation
时空数据填补与预测紧密相关——填补缺失值是许多预测管道的前置步骤。GSLI（AAAI 2025）提出多尺度图结构学习框架，通过节点尺度学习解决特征异质性问题，通过特征尺度学习捕获跨特征空间依赖，在 6 个真实数据集上取得最优填补性能[^src-yang-gsli-2025]。ImputeFormer（KDD 2024）则通过低秩归纳偏置实现线性复杂度的 Transformer 填补[^src-2312-01728]。CoFILL（arXiv 2025）使用条件扩散模型进行时空填补[^src-cofill-spatiotemporal-imputation]。

### Mamba / SSM-Based
Mamba 的选择性状态空间模型也被应用于交通预测。Han et al. (NeurIPS 2024) 的统一框架揭示了 Mamba 的遗忘门 $\widetilde{A}_i$ 在交通场景中对应于空间衰减模式——附近传感器的相关性更强，这与遗忘门的局部偏置特性一致。该框架表明，交通特定的位置编码（如道路距离、转向关系）可替代遗忘门的循环计算，在保持并行性的同时捕获局部空间结构[^src-demystify-mamba-linear-attention-2024]。

### Spectral / Topological Methods

A growing line of work applies spectral and topological techniques to address fundamental GNN limitations in traffic forecasting:

**[[hifinet|HiFiNet]]** (AAAI 2026) introduces hierarchical frequency-decomposition GNNs, explicitly separating low-frequency (smooth global) and high-frequency (local variation) graph signals to mitigate over-smoothing[^src-hifinet].

**[[ssf|SSF (Spectral Sheaf Filtering)]]** (ICLR 2026, under review) is the first framework to model spatio-temporal data using **[[cellular-sheaf|cellular sheaves]]** from algebraic topology. Rather than uniformly propagating information along edges like standard GNNs, SSF assigns learnable **restriction maps** per edge that encode context-dependent transformation dynamics. It then applies a **heat kernel spectral filter** over the [[sheaf-laplacian|sheaf Laplacian]] — a generalization of the graph Laplacian that accounts for both topology and edge-specific transformation semantics. The sheaf Laplacian's eigendecomposition enables frequency-aware decomposition, with the heat kernel $e^{-\alpha\lambda}$ suppressing high-frequency noise while preserving low-frequency structural patterns. SSF achieves SOTA on METR-LA, PEMS-BAY, PEMS04, PEMS08, and NAVER-Seoul, with particularly dramatic long-horizon gains — e.g., NAVER-Seoul MAPE 1.03% at 15min vs. best baseline 8.32%. The sheaf structure naturally mitigates [[over-smoothing-in-gnns|over-smoothing]] because restriction maps prevent node representations from converging[^src-ssf].

### Koopman / Micro-Macro Coupled

A fundamentally different paradigm: modeling traffic at two scales simultaneously — microscopic vehicle trajectories and macroscopic flow density — unified under [[micro-macro-coupled-koopman-modeling|Koopman operator theory]] that lifts nonlinear dynamics to linear observation spaces[^src-mmckm].

**[[mmckm|MMCKM]]** (ICLR 2026 Poster) is the first framework to achieve this bidirectional coupling on dynamic vehicle graphs. Key innovations[^src-mmckm]:

- **[[vehicle-centric-graph-traffic-pde|Vehicle-Centric Graph PDE]]**: Discretizes the LWR advection-diffusion equation directly onto vehicles as Lagrangian graph nodes, preserving high-frequency perturbations that Eulerian grid methods lose. Advection operator $C^{\text{adv}}$ is skew-symmetric (energy-preserving), diffusion operator $L^{\text{diff}}$ is PSD (entropy-producing), both parameterized with constructive physical guarantees[^src-mmckm].
- **History-Free Koopman Evolution**: Both macro (density) and micro (trajectory) dynamics are evolved by time-invariant linear Koopman operators from a single snapshot — eliminating the trajectory tracking overhead of sequence-based methods. Spectral alignment couples Koopman eigenvalues to PDE operator spectra for stability[^src-mmckm].
- **[[intent-discriminator-koopman|Intent Discriminator]] (MoE)**: Selects among 5 parameter-bounded Koopman operators (free flow, car-following, lane changing, merging, emergency) with distinct spectral radii, oscillation frequencies, and actuation bounds. Koopman control via CrossAttention injects macro flow into micro dynamics with ISS stability guarantees[^src-mmckm].

On NGSIM and HighD, MMCKM achieves history-free trajectory prediction matching history-dependent SOTA methods (BAT, MS-STGCN, Vit-Traj) while outperforming CV at all horizons. Operator interval creates a trade-off: 0.1s excels short-term (RMSE=0.33 at 1s), 1.0s excels long-term via fewer iterations. Ablation: diffusion term critical (removal degrades macro 2.9–4.6%); Intent Discriminator contributes 29% at short horizon; Koopman control reduces error 37% at 5s[^src-mmckm].

### Mixture of Experts / Adaptive Routing
[[testam|TESTAM]] (ICLR 2024) is the first MoE-based spatio-temporal attention model for traffic forecasting. It uses three heterogeneous experts — identity (temporal-only), learnable static graph, and spatial attention — adaptively routed via [[memory-augmented-gating|memory-augmented gating]] with two classification losses that solve the MoE routing freeze problem in regression. With only 224K params, TESTAM achieves SOTA on METR-LA, PEMS-BAY, and EXPY-TKY, excelling on large-scale graphs (1,843-node EXPY-TKY) and non-recurring conditions through in-situ spatial modeling[^src-testam]. The [[time-enhanced-attention|time-enhanced attention]] mechanism eliminates autoregressive error propagation by directly attending from source to target time steps.

### Neuron-Level Analysis / Pattern Neuron Fine-Tuning

A fundamentally different approach operates at the **neuron level** rather than the architecture level. [[pn-train|PN-Train]] (ICLR 2025) discovers that specific neurons in transformer-based UTSMs — called [[pattern-neuron|pattern neurons]] — are stably associated with low-frequency patterns (holidays, extreme weather). By fine-tuning only the detected pattern neurons (<10% of total parameters) while freezing the rest, PN-Train significantly improves holiday forecasting without degrading non-holiday performance[^src-pn-train].

Key findings[^src-pn-train]:
- Pattern neurons concentrate in attention query/key components, confirming the attention mechanism's role in pattern capture
- Shallow layers capture general patterns, middle layers refine low-level features (hierarchical distribution)
- Perturbation-based detection (directly measuring prediction impact) outperforms gradient-based methods
- Only ~10 fine-tuning samples (R=10) needed — making it a low-cost enhancement to existing UTSMs

PN-Train uses [[staeformer|STAEformer]] as its backbone and outperforms 9 baselines including [[testam|TESTAM]] on Metro-Traffic, Pedestrian, and GBAP datasets[^src-pn-train].

### Lightweight / MLP-Based

A growing direction challenges the dominance of heavy Transformer-based STGNNs, prioritizing efficiency without sacrificing accuracy:

**[[bist|BiST]]** (PVLDB 2025) is the first model to break the input-label spatiotemporal consistency assumption via a **bidirectional learning paradigm**[^src-bist]. The forward process uses pure MLP layers with temporal decomposition and spatiotemporal embedding prompts to generate base predictions. The backward process explicitly models [[spatiotemporal-deviation|spatiotemporal deviation]] between input and label representations through a residual decoupling module (context features via virtual clusters + personalized features) and adaptive diffusion smoothing. On 13 datasets (up to 16,972-node XTraffic, 20-year XXLTraffic) vs 26 baselines, BiST achieves **8.13% improvement over SOTA** while using only **1.86% of training time** and **7.36% of memory**[^src-bist]. The GMRF-based spatiotemporal dynamics theory proves that optimal prediction = base prediction + diffusion-smoothed correction term[^src-bist].

STID (Shao et al., CIKM 2022) uses learnable node embeddings to characterize spatiotemporal structure, assisting pure MLP in learning — a strong yet simple baseline for large-scale data.

**[[graphsparsenet|GraphSparseNet (GSNet)]]** (PVLDB 2025) addresses GNN scalability from a different angle: it observes that well-trained adaptive adjacency matrices are highly sparse, so learning the full N×N matrix is wasteful[^src-graphsparsenet]. GSNet replaces it with two small matrices — K (C×C low-dimensional adjacency) and U (combination coefficients) — performing all graph operations in a compressed C-dimensional space where C ≪ N. Two O(N) modules (Feature Extractor + Relational Compressor) achieve linear complexity while theoretically preserving the expressiveness of full-rank adjacency via Theorem 3.1[^src-graphsparsenet]. On CA (8,600 nodes), GSNet achieves SOTA MAE 19.76 with 3.51× faster training than BigST and 64–70× faster than GWNet/AGCRN[^src-graphsparsenet]. See [[low-dimensional-graph-adjacency]] for the compressed adjacency concept.
### Mixed-Graph Algorithm Unrolling

A new paradigm that bridges model-based optimization and data-driven learning: instead of designing larger attention mechanisms, **unroll a mixed-graph optimization algorithm into a lightweight, interpretable Transformer**[^src-lightweight-mixed-graph-unrolling].

Qi et al. (ICML 2026) propose constructing two graphs — an undirected graph $G^u$ for spatial correlations and a directed acyclic graph $G^d$ for temporal sequential relationships — and minimizing an optimization objective combining GLR (for $G^u$), [[directed-graph-laplacian-regularizer|DGLR]] ($\ell_2$), and [[directed-graph-total-variation|DGTV]] ($\ell_1$) via ADMM. Each ADMM iteration (CG solve, soft-thresholding, multiplier update) is unrolled into a neural layer, with periodically inserted graph learning modules that serve as [[graph-learning-as-self-attention|self-attention substitutes]]. The unrolled network achieves competitive performance with only **38K parameters** — 7.2% of transformer-based PDFormer (1,404K) and 4.9% of its inference cost[^src-lightweight-mixed-graph-unrolling].

Key advantages[^src-lightweight-mixed-graph-unrolling]:
- **Interpretability**: each layer corresponds to an optimization step (low-pass graph filter on $G^u$ and $G^d$)
- **Parameter efficiency**: graph learning modules replace Q/K/V matrices with compact Mahalanobis distances
- **Data efficiency**: maintains stable performance under limited training data where larger models degrade
- **Directed temporal modeling**: $G^d$ naturally captures time's arrow, outperforming undirected temporal graphs

See [[mixed-graph-spatiotemporal-modeling]], [[directed-graph-laplacian-regularizer|DGLR]], [[directed-graph-total-variation|DGTV]], and [[graph-learning-as-self-attention]] for detailed mechanism pages.

### Fine-Grained / Multi-Granularity Prediction

**[[fine-grained-traffic-prediction|Fine-grained traffic prediction]]** encompasses both road-level and lane-level forecasting within a unified framework[^src-minitraffic]. Unlike traditional large-scale urban traffic prediction that treats road segments as atomic units, fine-grained prediction models the internal lane structure of each road segment — critical for lane-change guidance in autonomous vehicles, dynamic lane system control, and precise signal optimization[^src-minitraffic].

The core challenge is **data imbalance**: road-level data is abundant (METR-LA, PeMS-Bay), but lane-level annotations are scarce and expensive to collect. [[mcgvae|McgVAE]] (CIKM 2024) was the first model to jointly handle both granularities using an ensemble VAE architecture, but it lacks a pre-training mechanism[^src-minitraffic].

**[[minitraffic|MiniTraffic]]** (ICML 2026) is the first lightweight pre-trained model (~119K params) specifically designed for this task. It uses [[frequency-domain-stability-augmentation|FDA]] to augment road-level data with frequency-domain perturbations that simulate lane-level variability, paired with contrastive clustering to construct small-scale semantic graphs for efficient attention[^src-minitraffic]. On 6 fine-grained datasets vs 29 baselines, MiniTraffic reduces lane-level MAE by 7%–39% and road-level FLOPs by ~85% compared to [[gpt-st|GPT-ST]][^src-minitraffic].


### Spatial Patching / Efficient Dynamic Spatial Modeling

Dynamic spatial attention (dot-product between all node pairs) has quadratic complexity O(N²d), making it intractable for large-scale networks. Three approaches have emerged to reduce this cost:

- **Linear-based** (e.g., BigST): O(Nd²) complexity by computing Q(K^T V) instead of (QK^T)V. Fast but loses interpretability — spatial correlations cannot be explicitly shown[^src-patchstg].
- **Low-rank-based** (e.g., STWave, AirFormer): Projects to reduced rank R ≪ N, achieving O(NRd). Loses fidelity — critical information is not guaranteed to survive the low-rank compression[^src-patchstg].
- **Patching-based** ([[patchstg|PatchSTG]]): O(NRd) complexity but retains both interpretability and fidelity. Borrows the patching idea from vision Transformers (ViT) and adapts it to irregular traffic points via [[leaf-kdtree|leaf KDTree]] spatial partitioning[^src-patchstg].

[[patchstg|PatchSTG]] (KDD 2025) is the first framework to bridge KDTree spatial data management and Transformer patching. It uses [[irregular-spatial-patching|irregular spatial patching]] (leaf KDTree → BFS → cosine-similarity padding → subtree backtracking) to create balanced, non-overlapping patches, then applies interleaved depth (within-patch local) and breadth (cross-patch global) attention. On LargeST (up to 8,600 nodes), PatchSTG achieves SOTA with **10× training speedup** and **4× memory reduction** vs D2STGNN/DSTAGNN[^src-patchstg]. Ablation confirms leaf KDTree is the most critical component — spatial message passing is only beneficial between geographically adjacent points[^src-patchstg].

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

### LLM-Augmented Few-Shot
[[fstllm|FSTLLM]] (ICML 2025) targets the data-scarce regime where STGNNs and TSFMs fail: it encodes node-specific text (e.g., car-park descriptions, reviews) with a frozen LLaMA-2-7B to build a semantically meaningful adjacency matrix via α-Entmax, runs a swappable graph-diffusion-convolution STGNN backbone, then QLoRA-fine-tunes an LLM to calibrate the numerical predictions with six-part prompts (task, node description, node pattern, history, prediction token, future token). It is plug-and-play — wrapping GPT4TS and [[itransformer|iTransformer]] improves them without retraining — and FSTLLM trained on 3 days of data beats baselines trained on 30 days, with ~30% MAPE reduction on Nottingham parking and >50% on ECL[^src-fstllm].

### Continual Spatio-Temporal Learning

While foundation models aim for zero-shot cross-city generalization, **[[continual-spatio-temporal-forecasting|CSTF]]** addresses a complementary challenge: sequentially learning from streaming, evolving data within a single domain without catastrophic forgetting. This is critical for real-world deployments where traffic networks continuously expand (new sensors added) and distributions shift over time[^src-stbp].

**[[team|TEAM]]** (PVLDB 2024) is the first framework to address traffic forecasting on **evolving RNs** — where nodes and edges can be both added and removed over time[^src-team]. TEAM formalizes the problem as a graph snapshot sequence, uses a hybrid Conv+Attention architecture (CAST) for efficient learning on small-scale incremental data, and introduces a continual learning module based on the [[wasserstein-metric|Wasserstein metric]] (EMD). The module measures per-node stability by comparing data histograms before/after evolution: stable nodes (low EMD) go to a consolidation buffer for rehearsal, unstable nodes (high EMD) go to an update buffer for re-training. With elastic weight consolidation (EWC) regularization, TEAM achieves 4× faster training than full retraining while maintaining competitive accuracy[^src-team]. See [[evolving-rn-traffic-forecasting]] for the problem formulation.

Key methods in this paradigm:

- **[[trafficstream|TrafficStream]]** (Chen et al., IJCAI 2021): First CSTF framework, using historical data replay and parameter smoothing[^src-stbp].
- **STKEC** (Wang et al., 2023): Influence-based knowledge expansion and memory-augmented consolidation for expanding graphs[^src-stbp].
- **[[pecpm|PECPM]]** (Wang et al., KDD 2023): Pattern-matching-based representative pattern bank with conflict detection and traceability mechanisms[^src-stbp].
- **STRAP** (Zhang et al., NeurIPS 2025): Retrieval-augmented multi-dimensional pattern library for OOD generalization[^src-stbp].
- **[[eac|EAC]]** (Chen & Liang, ICLR 2025): Dynamic prompt pool with expand-and-compress operations, lightweight parameter-efficient CSTF[^src-stbp].
- **[[stbp|STBP]]** (Liu & Zhang, ICLR 2026): Fixed general backbone + incrementally expanding [[contextual-pattern-bank|contextual pattern bank]]. Freezes backbone to prevent forgetting, expands only parametric bank for adaptation. Achieves 21.44% MAE reduction over EAC on PEMS-Stream via frequency-domain processing (FreNet) and dual-stream linear graph attention (DLGA)[^src-stbp].

### Out-of-Distribution Generalization

Recent work argues the node-to-node message-passing core of STGNNs is itself a source of out-of-distribution fragility: [[stop|STOP]] (ICML 2025) blocks node-to-node messages and routes all interaction through a small set of shared Context-Aware Units, improving OOD generalization by up to 17.01% and inductive performance on new sensors by up to 18.44%[^src-stop].

### Test-Time Calibration / Distribution Shift
[[st-ttc|ST-TTC]] (NeurIPS 2025 Spotlight) corrects non-stationary distribution shift at inference time without retraining: it appends a lightweight [[spectral-domain-calibration|spectral-domain calibrator]] (per-node amplitude/phase modulation) after a frozen backbone and updates it via a leakage-free [[flash-gradient-update|flash gradient update]] on historical labels, yielding consistent ~1–2% MAE/RMSE gains across 6 backbones on PEMS03/04/07/08, KnowAir, and UrbanEV (METR-LA RMSE 7.43→7.21 with GWNet), and complementing OOD and continual learning methods[^src-st-ttc].

## Key Models

Several influential models span the development of traffic and spatial-temporal forecasting:

- **[[source-st-resnet|ST-ResNet]]** (AAAI 2017) — one of the first deep learning approaches for citywide crowd flow prediction, using residual convolutional units to model spatial-temporal dependencies[^src-st-resnet].
- **[[source-astgcn|ASTGCN]]** (AAAI 2019) — combines attention mechanisms with graph convolution to jointly capture spatial and temporal patterns in traffic flow[^src-astgcn].
- **[[source-prnet|PRNet]]** — introduces periodic residual learning to explicitly model recurring temporal patterns in crowd flow forecasting[^src-prnet].
- **[[source-penguin|PENGUIN]]** (AISTATS 2026) — proposes periodic-nested group attention for long-sequence time-series forecasting, with applicability to traffic domains[^src-penguin].

For a comprehensive overview of deep learning methods for time series, including traffic forecasting, the [[source-deep-time-series-survey|TSLib survey]] provides systematic benchmarking across multiple domains[^src-deep-time-series-survey].

## Related Tasks

[[multimodal-traffic-profiling|Multimodal Traffic Profiling]] — Unlike forecasting (predicting future values), profiling is a **classification** task that identifies traffic states (smooth/slow/congested) or events (accidents/construction). [[mtp|MTP]] (AAAI 2026) augments numerical time series into visual and textual modalities, processing all three in the frequency domain with hierarchical contrastive fusion for SOTA classification results on 6 traffic datasets[^src-mtp].

### Cross-City Traffic Flow Generation

While traffic forecasting predicts future values from historical data, **traffic flow generation** synthesizes realistic flow data from static geographic features — critical for cities with limited or no historical records[^src-craft]. The task has evolved through three stages: physics-based models (gravity/radiation), static flow generation (DeepGravity, DeepFlowGen), and dynamic flow generation (GANs, diffusion models)[^src-craft].

[[craft|CRAFT]] (NeurIPS 2025) is the first method explicitly designed for **zero-shot cross-city traffic flow generation**. It uses a DDPM backbone with two lightweight plug-in modules: [[geographic-feature-alignment|Geographic Feature Alignment (GFA)]] to solve cross-city domain shift via optimal transport, and [[retrieval-based-condition-augmentation|Retrieval-based Condition Augmentation (RCA)]] to enrich diffusion conditions by retrieving similar flow patterns from source cities[^src-craft]. On four bicycle-sharing datasets (Chicago, DC, Toronto, NYC), CRAFT achieves 59.7% improvement over baseline average and only 10.4% degradation vs. training on real target city data[^src-craft]. See [[cross-city-traffic-flow-generation]] for the problem domain overview.

### Mobile Network Traffic

While the above sections address **vehicle traffic** (road sensors), mobile traffic forecasting addresses **wireless network traffic** — predicting data volumes at cellular base stations. [[uomo|UoMo]] (KDD 2025) is the first universal foundation model for this domain, unifying short-term prediction, long-term prediction, and zero-history generation under a single transformer-based diffusion model with task-oriented masking and contrastive context alignment[^src-uomo]. Deployed on China Mobile's [[jiutian-platform|Jiutian platform]], UoMo achieves +25.3% served users in BS deployment and -40.7% equipment depreciation in BS sleep control[^src-uomo]. See [[mobile-traffic-forecasting]] for the domain page and [[masked-diffusion-pre-training]] for the pre-training technique.

### Large-Scale Linear-Complexity Modeling

As road networks grow to tens of thousands of nodes, the $O(N^2)$ adaptive-adjacency cost of [[gwnet|GWNet]]-style STGNNs becomes prohibitive. [[bigst|BigST]] (PVLDB 2024) decouples long-sequence modeling into a cached [[long-sequence-feature-extractor|feature extractor]] and replaces dense graph convolution with [[linearized-spatial-convolution|linearized spatial convolution]] (PRF-kernel factorization), achieving $O(N)$ complexity and scaling to ~100K nodes (Beijing, 99,716 segments) while beating GWNet/AGCRN/DCRNN by 6–9% MAE[^src-bigst]. Related scalable approaches include [[ragc|RAGC]] (cosine-similarity $O(N)$ graph conv) and [[patchstg|PatchSTG]] (KDTree spatial patching); see [[large-scale-spatial-temporal-graph]] for the full landscape[^src-bigst].

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
[^src-mtp]: [[source-mtp]]
[^src-hifinet]: [[source-hifinet]]
[^src-metadg]: [[source-metadg]]
[^src-testam]: [[source-testam]]
[^src-uomo]: [[source-uomo]]
[^src-craft]: [[source-craft]]
[^src-patchstg]: [[source-patchstg]]
[^src-stbp]: [[source-stbp]]
[^src-ssf]: [[source-ssf]]
[^src-mmckm]: [[source-mmckm]]
[^src-stop]: [[source-stop]]
[^src-fstllm]: [[source-fstllm]]
[^src-st-ttc]: [[source-st-ttc]]
[^src-bist]: [[source-bist]]
[^src-graphsparsenet]: [[source-graphsparsenet]]
[^src-bigst]: [[source-bigst]]
[^src-lightweight-mixed-graph-unrolling]: [[source-lightweight-mixed-graph-unrolling]]
[^src-minitraffic]: [[source-minitraffic]]
[^src-pn-train]: [[source-pn-train]]
