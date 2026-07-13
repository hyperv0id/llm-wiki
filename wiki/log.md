## [2026-07-13] ingest | SRSNet: Enhancing Time Series Forecasting through Selective Representation Spaces
创建的页面：[[source-srsnet]], [[srsnet]], [[selective-representation-space]], [[selective-patching]], [[dynamic-reassembly]], [[adaptive-fusion]]
更新的页面：[[index]], [[patchtst]], [[patch-based-tokenization]], [[crossformer]]
核心贡献：批评固定 adjacent patching 的表示空间；提出可微 Selective Patching + Dynamic Reassembly + Adaptive Fusion 的即插即用 SRS 模块；SRSNet=SRS+MLP 在多域 LTSF 上 SOTA，并可提升 PatchTST/Crossformer 等 patch 骨干。NeurIPS 2025 / arXiv:2510.14510。
源文件：外部 PDF（不可变 raw/ 策略；路径见 ingest-reports/srsnet-why.md）
---

---
title: Log
type: concept
created: 2026-04-26
last_updated: 2026-07-13
## [2026-07-13] ingest | QDF: Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models
创建的页面：[[source-qdf]], [[qdf]], [[quadratic-form-weighted-objective]], [[heterogeneous-task-weights]]
更新的页面：[[index]], [[source-fredf]], [[source-distdf]], [[fredf]], [[label-autocorrelation]], [[direct-forecast]], [[autocorrelation-bias]], [[joint-distribution-wasserstein-alignment]]
核心贡献：在 DF 学习目标中同时处理标签自相关（Σ^{-1} 非对角）与异质任务权重（Σ^{-1} 非均匀对角）；双层优化学习 PSD 加权矩阵后以二次型 NLL 训练，模型无关；相对 FreDF/Time-o1/DistDF 的同组兄弟工作。arXiv:2511.00053（ICLR 2026 preprint）。
源文件：外部 PDF（不可变 raw/ 策略；路径见 ingest-reports/qdf-why.md）
---

## [2026-07-13] ingest | GCGNet: Graph-Consistent Generative Network for Time Series Forecasting with Exogenous Variables
创建的页面：[[source-gcgnet]], [[gcgnet]], [[joint-temporal-channel-correlation]], [[graph-structure-aligner]], [[graph-refiner]], [[variational-generator-exogenous]]
更新的页面：[[index]], [[source-timexer]], [[source-kite]], [[kite]], [[dag]]
核心贡献：批评外生预测的两步时间/通道串行建模，提出 VAE 粗生成 + Graph Structure Aligner（patch 图 VAE + L1 对齐）+ Graph Refiner（top-k GCN）联合建模相关并抗噪声；12 个外生数据集 ICLR 2026 SOTA。
源文件：外部 PDF（不可变 raw/ 策略；路径见 ingest-reports/gcgnet-why.md）
---

## [2026-07-13] ingest | FreDF: Learning to Forecast in Frequency Domain
创建的页面：[[source-fredf]], [[fredf]], [[label-autocorrelation]], [[direct-forecast]], [[frequency-enhanced-direct-forecast]]
更新的页面：[[index]], [[autocorrelation-bias]]
核心贡献：指出 DF 多步预测忽略标签自相关（Theorem 3.1），提出在频域对齐预测与标签的模型无关训练范式 L_α = α·L_feq + (1-α)·L_tmp；在 ETT/ECL/Traffic/Weather 与 M4 上提升 iTransformer/FreTS 等骨干。arXiv:2402.02399（源文件标注 ICLR 2025）。
源文件：raw/FreDF_Wang_2025_ICLR.pdf（不可变，仅读取外部 PDF 路径）
---


## [2026-07-12] query | ICML 2026 Spotlight 与多模态外生信息引导长期时空预测的相关分析
从 OpenReview 抓取 ICML 2026 全部 538 篇 spotlight 论文的标题+摘要，按方向“多模态外生信息引导的长期时空预测”分三层筛选：直接相关、方法可迁移、基础设施。
创建的页面：[[source-icml-2026-spotlight-papers]], [[icml-2026-spotlight-vs-multimodal-exogenous-spatiotemporal]]
更新的页面：[[index]]
源文件：raw/icml-2026-spotlight-papers.md（538 篇）
---

## [2026-07-08] ingest | ST Foundation Models Survey (2501.09045)
创建的页面：[[source-st-foundation-models-survey]]
更新的页面：[[index]]
核心贡献：提出 STFMs 系统愿景与 4 维泛化能力框架（领域/空间/时间/尺度），定性评估 6 个现有 STFM（UniST/OpenCity/UrbanGPT/ClimaX/Pangu/W-MAE），指出碎片化与空间偏差问题，展望统一架构/跨域协同/多模态训练/分布偏移适应。A*STAR 2025。

## [2026-07-08] ingest | Multi-modal Time Series Survey (2503.13709)
创建的页面：[[source-multimodal-ts-survey]]
更新的页面：[[index]]
核心贡献：系统综述多模态时间序列分析，提出统一跨模态交互框架（融合/对齐/迁移），覆盖 40+ 方法和多领域数据集（Terra/Time-MMD/MIMIC/FNSPID），讨论推理/决策/泛化/偏见的未来方向。arXiv 2025。


## [2026-07-07] ingest | STFM Pipeline Review — Unraveling Spatio-Temporal Foundation Models via the Pipeline Lens
创建的页面：[[source-stfm-pipeline-review]]
更新的页面：[[index]]
核心贡献：Pipeline 视角 STFM 综述，覆盖数据协调→模型设计（原始/迁移分类法）→训练目标→迁移适配→应用，涵盖 UniST/UrbanDiT/ClimaX/Moirai/Chronos。arXiv 2025 / TKDE。
---
## [2026-07-07] ingest | Multimodal PINN for Mean Radiant Temperature Modeling
创建的页面：[[source-multimodal-pinn]]
更新的页面：[[index]]
核心贡献：多模态 PINN 框架融合数值特征 + 鱼眼图像（ResNet-50）+ 辐射传输物理损失函数，MaRTy 数据集上 RMSE 3.50 / R² 0.88，阴影预测准确率 94%。arXiv 2025。
---
## [2026-07-07] ingest | What If TSF (WIT) — Scenario-Guided Multimodal Forecasting Benchmark
创建的页面：[[source-whatif-tsf]]
更新的页面：[[index]]
核心贡献：首个场景引导多模态时序预测基准，5352 样本覆盖 4 领域，含可能/反事实场景，方向准确率评估，LLM 利用场景引导显著优于纯时序方法。arXiv 2026。
---
## [2026-07-07] ingest | CaST: Deciphering Spatio-Temporal Graph Forecasting — A Causal Lens and Treatment
创建的页面：[[source-cast]]
更新的页面：[[index]]
核心贡献：首次将 SCM + do-calculus 系统应用于 STG 预测，后门调整处理时序 OoD（环境解缠 + VQ codebook），前门调整 + Hodge-Laplacian 边卷积建模动态空间因果关系的涟漪效应。NeurIPS 2023。
## [2026-07-07] ingest | ClimaX: A Foundation Model for Weather and Climate
创建的页面：[[source-climax]]
更新的页面：[[index]]
核心贡献：首个气象/气候基础模型，variable tokenization + cross-attention aggregation 处理异构数据，CMIP6 预训练 + randomized forecasting objective，WeatherBench 与 IFS 竞争，ClimateBench SOTA。ICML 2023。
## [2026-07-07] ingest | DP-GPT4MTS: Dual-Prompt LLM for Textual-Numerical Time Series Forecasting
创建的页面：[[source-gpt4mts]]
更新的页面：[[index]]
核心贡献：双提示机制（explicit-hard prompt + textual-soft prompt via BERT）解耦任务指令与文本上下文，frozen GPT-2 骨干仅微调位置/层归一化，GDELT/Time-MMD 上 SOTA。

## [2026-06-22] ingest | Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)

## [2026-06-22] ingest | DDIM (ODE 视角) + 苏剑林博客
创建的页面：[[source-ddim]], [[source-ddim-ode-spaces-ac-cn]], [[ddim]]
更新的页面：[[probability-flow-ode]], [[diffusion-model]], [[ddpm]], [[index]]
核心视角：从 Fokker-Planck 方程的等价变换推导概率流 ODE，DDIM 本质 = PF-ODE 在 VP SDE 下的一阶 Euler 离散化。加速采样 = ODE 大步长离散化，一致性 = ODE 确定性，可逆性 = ODE 双向求解。

Ingested the original classifier-free guidance paper by Ho & Salimans (Google Research Brain, arXiv:2207.12598, July 2022). This is the foundational work that introduced CFG — the dominant conditional generation technique for diffusion models, used in Stable Diffusion, DALL-E 2, Imagen, and virtually all modern diffusion pipelines.

Created: [[source-classifier-free-diffusion-guidance]]
Updated: [[classifier-free-guidance]], [[classifier-guidance]], [[index]]

Key insights captured:
- Original CFG formulation: $\tilde{\epsilon}_\theta = (1+w)\epsilon_\theta(z_\lambda,c) - w\epsilon_\theta(z_\lambda)$
- Joint training with $p_\text{uncond}$ random conditioning dropout; $p_\text{uncond} \in \{0.1, 0.2\}$ optimal
- Implicit classifier motivation: $p^i(c|z) \propto p(z|c)/p(z)$, but actual scores need not be conservative → CFG is not an adversarial attack
- Continuous-time framework (log SNR $\lambda$, hyperbolic secant distribution)
- 128×128 ImageNet: FID=2.43 at $w=0.3$ beats classifier-guided ADM-G (FID=2.97); IS=421 at $w=4.0$
- Intuitive explanation: decreases unconditional likelihood while increasing conditional likelihood
- Updated [[classifier-guidance]] with the paper's critique of classifier guidance (adversarial attack concerns, GAN similarity)

## [2026-06-22] ingest | RoFormer: Rotary Position Embedding (Su et al., 2021/2023)

Ingested the original RoPE paper by Su Jianlin (苏剑林) et al. from Zhuiyi Technology (arXiv:2104.09864v5, Nov 2023). This is the foundational work that introduced Rotary Position Embedding — the multiplicative position encoding scheme now used by virtually all major LLMs (LLaMA, Mistral, Qwen, DeepSeek, etc.).

Created: [[source-roformer]], [[rope]], [[roformer]]
Updated: [[siren-rope]], [[temporal-rotation]], [[learnable-frequency-scaling]], [[index]]

Key insights captured:
- Core formulation: encode absolute position with rotation matrices → self-attention inner product depends only on relative position
- θ_i = 10000^{-2(i-1)/d} frequency schedule inherited from sinusoidal PE
- Three key properties: long-term decay, sequence length flexibility, linear attention compatibility
- Experiments: WMT EN-DE, BERT pre-training (faster convergence), GLUE, Performer+RoPE, Chinese long text
- First relative position encoding compatible with linear (O(N)) self-attention

## [2026-06-22] ingest | mHC: Manifold-Constrained Hyper-Connections (Xie et al., 2026)

Ingested arXiv 2512.24880v2 from DeepSeek-AI — proposes Manifold-Constrained Hyper-Connections, a framework that constrains Hyper-Connections' residual mapping $H_l^{res}$ to the Birkhoff polytope of doubly stochastic matrices via Sinkhorn-Knopp projection, restoring the identity-mapping property and enabling stable large-scale LLM pretraining.

Created: [[source-mhc-manifold-constrained-hyper-connections]], [[manifold-constrained-hyper-connections]], [[hyper-connections]], [[birkhoff-polytope]], [[identity-mapping-property]]
Updated: [[sinkhorn-algorithm]], [[residual-connections-as-diffusion]], [[index]]

Key insights captured:
- HC expands residual stream width but breaks identity mapping; composite $H_l^{res}$ gain peaks near 3000 in 27B models
- mHC constrains $H_l^{res}$ to doubly stochastic matrices: spectral norm $\leq 1$, closed under multiplication, preserves mean/norm
- Infrastructure: TileLang kernel fusion, selective recomputation with $L_r^* \approx \sqrt{nL/(n+2)}$, DualPipe communication overlap
- 27B MoE results: loss -0.021 vs baseline, 6.7% time overhead for expansion rate $n=4$

## [2026-06-18] ingest | Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks (Zheng et al.)

Ingested arXiv 2510.25760 — a comprehensive survey covering multimodal spatial reasoning with MLLMs across 2D, 3D, embodied AI, and novel modalities (video, audio). The paper provides a systematic taxonomy, evaluates methods across four dimensions (test-time scaling, post-training, architecture, explainability), and introduces open benchmarks.

Created: [[source-2510-25760]], [[multimodal-spatial-reasoning]], [[multimodal-large-language-model]], [[3d-visual-grounding]], [[vision-language-action]], [[vision-language-navigation]], [[embodied-question-answering]]
Updated: [[index]]

Key insights captured:
- 10 types of spatial reasoning + 6 evaluation dimensions
- Three 3D grounding approaches: 3D input, multi-view, hybrid
- Four VLA enhancement directions: spatial modalities, multi-task co-training, explicit reasoning, backbone evaluation
- VLN taxonomy: environment understanding → intention interpretation → path planning
- Root causes of MLLM spatial failures: representation imbalance, attention bias (only 15-20% on spatial relations), lack of geometric priors

## [2026-06-18] ingest | Bulk Ingest: 13 Exogenous Multimodal Spatio-Temporal Papers

批量收录 13 篇外生变量驱动/多模态时空预测与轨迹建模论文：

Exogenous Variables & Causal:
- [[source-select-then-balance]] — Select, then Balance: Exogenous Variable Modeling for ST Forecasting
- [[source-causal-st-prediction]] — E²-CSTP: Causal Spatio-Temporal Prediction via Multi-Modal Approach (NeurIPS 2025)
- [[source-causal-llm]] — Causal-LLM: Spatiotemporal Foundation Model

LLM + Geolocation & Mobility:
- [[source-geolocation-llm-st]] — LLMGeovec: Geolocation Representation from LLMs for ST Learning (AAAI 2025)
- [[source-beyond-imitation-mobility]] — Beyond Imitation: Human Mobility from LLM Reasoning (MobiGeaR)

Prediction Applications:
- [[source-jstc]] — JSTC: Travel Time Prediction with Joint Spatial-Temporal Correlation
- [[source-flightdiff]] — FlightDiff: Dual-Constraint Guided Diffusion for Flight Prediction
- [[source-deepfec]] — DeepFEC: Energy Consumption Prediction for Smart Cities

Language & Text Integration:
- [[source-language-flow-time]] — Language in the Flow of Time: TaTS Framework (ICLR 2026)

Surveys:
- [[source-multimodal-spatial-reasoning-survey]] — Multimodal Spatial Reasoning Survey (Zheng et al., 2025)
- [[source-trajectory-dl-survey]] — Deep Learning for Trajectory Data Management Survey

Representation & Retrieval:
- [[source-clmtr]] — CLMTR: Contrastive Multi-modal Trajectory Representation
- [[source-retrieval-augmented-st-traffic]] — RAST: Retrieval Augmented Spatio-Temporal Framework for Traffic

已创建页面：[[source-select-then-balance]], [[source-causal-st-prediction]], [[source-causal-llm]], [[source-geolocation-llm-st]], [[source-beyond-imitation-mobility]], [[source-jstc]], [[source-flightdiff]], [[source-deepfec]], [[source-language-flow-time]], [[source-multimodal-spatial-reasoning-survey]], [[source-trajectory-dl-survey]], [[source-clmtr]], [[source-retrieval-augmented-st-traffic]]
已更新页面：[[index]]
修复 frontmatter：日期修正（3 页 2025→2026）、confidence high→medium（9 页）、补 status: active（1 页）

## [2026-06-18] maintenance | 交叉链接补充：multimodal-spatial-reasoning ↔ spatio-temporal-reasoning

收录后 lint 发现 [[multimodal-spatial-reasoning]] 与既有 [[spatio-temporal-reasoning]] 应为双向交叉链接——二者分属同一空间推理概念谱系的不同层次（前者是 MLLM 通用框架，后者是图结构时序数据的特化）。已为双方添加反向链接。
更新的页面：[[multimodal-spatial-reasoning]], [[spatio-temporal-reasoning]]

## [2026-06-16] maintenance | SB ingest 补充：交叉链接、frontmatter 修复与重复页面处理

修复 SB ingest 的多项遗漏：
- 为 [[diffusion-models]], [[optimal-transport]], [[flow-matching]], [[score-based-generative-modeling]] 添加 SB 交叉链接
- 将 [[stochastic-optimal-control]] 标记为 superseded → [[stochastic-optimal-control-sb]]
- 修复 index.md 中 hopf-cole-transform 双分类重复
- 统一修正所有单 source 页面的 confidence: high → medium，补全缺失的 frontmatter 字段（confidence, status, last_updated）
- 更新受影响页面的 last_updated 和 source_count

## [2026-06-16] maintenance | Rewrote 5 SB pages to full Chinese with all formulas

重写了 5 个 Schrödinger bridge 相关页面，将正文改为中文，保留所有技术公式，加入 `> [!note]` callout。所有页面格式完整：frontmatter、wikilinks、公式、[[cross-links]]、脚注。

更新的页面：[[stochastic-optimal-control-sb]], [[girsanov-theorem]], [[diffusion-schrodinger-bridge-matching]], [[adjoint-matching]], [[gaussian-schrodinger-bridge]]

## [2026-06-16] ingest | Foundations of Schrödinger Bridges for Generative Modeling (Tang, 2026)

Ingested arXiv:2603.18992, a 220-page comprehensive tutorial unifying SB theory for generative modeling. Paper was fully read (all 9 sections + appendices). Created source-summary + 22 concept/technique pages in Chinese (正文中文，技术名词英文)，each with complete mathematical formulations.

创建的页面：[[source-schrodinger-bridges-generative-modeling]], [[schrodinger-bridge]], [[entropic-optimal-transport]], [[sinkhorn-algorithm]], [[hopf-cole-transform]], [[girsanov-theorem]], [[building-schrodinger-bridges]], [[stochastic-optimal-control-sb]], [[doob-h-transform]], [[iterative-markovian-fitting]], [[diffusion-schrodinger-bridge-matching]], [[adjoint-matching]], [[adjoint-schrodinger-bridge-sampler]], [[conditional-score-flow-matching]], [[gaussian-schrodinger-bridge]], [[discrete-schrodinger-bridge]], [[fractional-schrodinger-bridge]], [[multi-marginal-schrodinger-bridge]], [[unbalanced-schrodinger-bridge]], [[generalized-schrodinger-bridge]], [[branched-schrodinger-bridge]]
更新的页面：[[index]]


## [2026-06-16] creation | SB Generative Modeling Algorithms and Variations

Created four detailed technique/concept pages with complete mathematical formulations from the source "Foundations of Schrödinger Bridges for Generative Modeling" (Tang, 2026). These pages provide all key formulas for the generative modeling algorithms and problem variations in the SB framework.

创建的页面：[[diffusion-schrodinger-bridge-matching]] (rewritten with full DSBM loss formulas), [[adjoint-matching]] (lean AM, SB-AM, corrector matching, adjoint sampling), [[gaussian-schrodinger-bridge]] (closed-form Gaussian SB with Bures-Wasserstein geometry), [[fractional-schrodinger-bridge]] (fBM, OU approximation, Doob's h-transform in augmented space)
更新的页面：[[index]], [[log]]

## [2026-06-16] creation | Schrödinger Bridge Technique Pages

Created four technique/concept pages from the source "Foundations of Schrödinger Bridges for Generative Modeling". These cover the core mathematical machinery of the SB framework: the Hopf-Cole linearization, stochastic optimal control formulation, Doob's h-transform construction, and Iterative Markovian Fitting algorithm.

创建的页面：[[hopf-cole-transform]], [[stochastic-optimal-control]], [[doob-h-transform]], [[iterative-markovian-fitting]]
更新的页面：[[index]], [[log]]

## [2026-06-16] ingest | Benamou-Brenier 算法（博客笔记）

Ingest Better_Yu 的博客园笔记《最优传输算法——Benamou Brenier算法》(2021-05-09). The blog introduces the Benamou-Brenier algorithm — a continuous numerical method for optimal transport that reformulates the OT problem as a convex variational problem in (d+1)-dimensional space-time. Key ideas: (1) Dynamic OT formulation — find Wasserstein geodesics μ_t instead of static map T; (2) Variable substitution E_t = v_t ϱ_t to convert a non-convex, nonlinearly-constrained problem into a convex optimization with linear constraints; (3) Numerical solution via augmented Lagrangian — three-step iteration (Laplace solve O(N log N) + pointwise projection O(N) + dual ascent). Strengths: handles vanishing densities without special assumptions, works with general convex costs, supports density constraints and multi-population extensions. 与 Flow Matching 中的 OT 路径有深层联系（连续性方程 → Fokker-Planck 的无噪声版）。

创建的页面：[[source-benamou-brenier-blog]], [[benamou-brenier-algorithm]]
更新的页面：[[optimal-transport]], [[index]], [[log]]

## [2026-06-16] ingest | DiffusionBlocks — Block-wise Neural Network Training via Diffusion Interpretation (ICLR 2026)

Ingest DiffusionBlocks paper (Makoto Shing, Masanori Koyama, Takuya Akiba; Sakana AI, University of Tokyo; ICLR 2026; arXiv:2506.14202v4; code: github.com/SakanaAI/DiffusionBlocks). DiffusionBlocks provides the first theoretically grounded framework for block-wise neural network training by interpreting residual connections as discretized steps of continuous-time diffusion processes. Core insight: residual updates z_ℓ = z_{ℓ-1} + f(z_{ℓ-1}) naturally correspond to Euler discretization of the probability flow ODE in diffusion models, enabling partitioning networks into blocks that each handle specific noise-level ranges and train independently via score matching. Key innovations: (1) Equi-probability partitioning — divide noise range by equal cumulative probability mass (not uniform spacing) to ensure balanced learning difficulty across blocks; (2) Independent block training — each block trains with gradients for only L/B layers, achieving B× memory reduction for ALL components (parameters, gradients, optimizer states, activations); (3) Systematic conversion recipe for transformer architectures — partition layers, assign noise ranges, add noise conditioning. Experimental results across diverse architectures while matching/exceeding end-to-end performance: ViT (CIFAR-100: 59.30% vs 60.25%, 3× memory, vastly outperforms Forward-Forward 7.85%), DiT (ImageNet FID 9.00 vs 9.01, 3× training+inference memory), Masked Diffusion (text8: 1.45 vs 1.56 BPC — better), Autoregressive LMs (comparable MAUVE, 4× memory), Recurrent-depth (Huginn: 0.70 vs 0.49 MAUVE while eliminating 32 iterations). Surprising finding: moderate partitioning (B=2-3) sometimes outperforms end-to-end due to specialization-induced curriculum learning. Wall-time overhead minimal (~7% from noise conditioning). Composable with activation checkpointing. First method achieving continuous-time + block-wise + competitive performance. 高质量论文，理论扎实，实验全面，影响深远。

创建的页面：[[source-diffusionblocks]], [[diffusionblocks]], [[block-wise-training]], [[residual-connections-as-diffusion]], [[equi-probability-noise-partitioning]], [[memory-efficient-training]], [[activation-checkpointing]]
更新的页面：[[edm]], [[dit]], [[index]], [[log]]

## [2026-06-15] ingest | PAST — Primary-Auxiliary Spatio-Temporal Network for Traffic Time Series Imputation (PVLDB)

Ingest PAST paper (Hanwen Hu, Zimo Wen, Shiyou Qian, Jian Cao; Shanghai Jiao Tong University; PVLDB, arXiv:2511.13414, 2025; code: github.com/Hanwen-Hu/PAST). PAST is a traffic time series imputation model that disentangles spatio-temporal patterns into primary (internal data relationships) and auxiliary (external features like timestamps and node attributes). Core innovations: (1) Graph-Integrated Module (GIM) — pure GNN with dynamic directed temporal graphs, interval-aware dropout, and multi-order spatial convolutions for primary pattern extraction; (2) Cross-Gated Module (CGM) — bidirectional sigmoid+tanh gating on temporal and spatial embeddings for auxiliary pattern extraction; (3) Ensemble self-supervised training framework (GBDT-inspired residual fitting). Evaluated on METR-LA, PeMS-Bay, LargeST-SD across 27 missing conditions vs 7 baselines: up to 26.2% RMSE reduction and 31.6% MAE reduction in block missing scenarios.

创建的页面：[[source-past]], [[past]], [[primary-auxiliary-patterns]], [[interval-aware-dropout]], [[cross-gated-mechanism]]
更新的页面：[[index]], [[log]]

## [2026-06-15] ingest | STD-PLM — Understanding Both Spatial and Temporal Properties of Spatial-Temporal Data with PLM (AAAI 2025)

Ingest STD-PLM paper (Yiheng Huang, Xiaowei Mao, Shengnan Guo, Yubin Chen, Junfeng Shen, Tiankuo Li, Youfang Lin, Huaiyu Wan; Beijing Jiaotong University; AAAI 2025; code: github.com/Hyheng/STD-PLM). STD-PLM is a unified PLM-based framework for both spatial-temporal forecasting and imputation, using GPT-2 backbone (first 3 layers) with LoRA fine-tuning. Core innovations: (1) Spatial-Temporal Tokenizer — generates tokens from both spatial (node-level) and temporal (system-level) dimensions, a first among PLM-based ST models; (2) Topology-Aware Node Embedding — uses Laplacian eigenvector embeddings for inductive cross-graph transfer; (3) Sandglass Attention (SGA) — precoder-decoder attention module that aggregates N node tokens into M region tokens (M<N) then recovers, capturing high-order correlations while reducing compute; (4) Unified forecasting+imputation in one model with mask tokens for missing awareness. On PEMS03/04/07/08: SOTA or runner-up for forecasting, SOTA for imputation (RM 70% + CM 70%). Remarkable few-shot: 5% training data matches full LSTM, 20% surpasses full ASTGCN. Zero-shot cross-dataset transfer maintains acceptable accuracy.

创建的页面：[[source-std-plm]], [[std-plm]], [[sandglass-attention]], [[spatial-temporal-tokenizer]], [[topology-aware-node-embedding]]
更新的页面：[[index]], [[log]], [[time-llm]], [[nuwats]]

## [2026-06-10] ingest | TEAM — Topological Evolution-aware Framework for Traffic Forecasting (PVLDB 2024)

Ingest TEAM paper (Duc Kieu, Tung Kieu, Peng Han, Bin Yang, Christian S. Jensen, Bac Le; U. of Science HCM/Aalborg U/UESTC/ECNU; PVLDB 18(2): 265-278, 2024; code: github.com/kvmduc/TEAM-topo-evo-traffic-forecasting). TEAM is the first framework for traffic forecasting on evolving road networks. Core innovations: (1) Problem formalization — traffic forecasting as graph snapshot sequence with add/remove nodes and edges; (2) CAST — hybrid Conv+Attention architecture (ChebNetII + GAT + dilated TCN + temporal attention) with doubly residual stacks for efficient learning on small-scale incremental data; (3) Continual learning module — Wasserstein/EMD-based node stability measurement, dual buffers (consolidation Bc + update Bu), elastic weight consolidation. On PEMS03-Evolve and PEMS04-Evolve (7-month evolution), TEAM achieves 4× faster training than full retraining while maintaining competitive accuracy. First work to handle both RN expansion and shrinkage with rehearsal-based continual learning for regression. PVLDB 2024 published paper — confidence: high.

Created pages: [[source-team]], [[team]], [[evolving-rn-traffic-forecasting]]
Updated pages: [[traffic-forecasting]], [[index]], [[log]]

## [2026-06-10] ingest | GraphSparseNet — A Novel Method for Large Scale Traffic Flow Prediction (PVLDB 2025)

Ingest GraphSparseNet paper (Weiyang Kong, Kaiqi Wu, Sen Zhang, Yubao Liu; Sun Yat-Sen University; PVLDB 18(7): 2295-2307, 2025; code: github.com/PolynomeK/GSNet). GSNet is a scalable GNN framework that replaces full N×N adjacency matrix learning with low-dimensional (C×C) compressed graph operations. Core innovations: (1) Theoretical proof (Theorem 3.1) that rank-C adjacency can be equivalently expressed by two small matrices K and U; (2) Feature Extractor — compress-decompress pipeline based on node embeddings for node feature learning; (3) Relational Compressor — compress input to C-dim → concatenate coefficient U → feature fusion via K → decompress, all O(N). Evaluated on 4 datasets (up to 8,600-node CA) vs 13 baselines. On CA: SOTA MAE 19.76, 3.51× faster training than BigST, 64-70× faster than GWNet/AGCRN. Ablation confirms RC module and K matrix are most critical. PVLDB 2025 published paper — confidence: high.

Created pages: [[source-graphsparsenet]], [[graphsparsenet]], [[low-dimensional-graph-adjacency]]
Updated pages: [[traffic-forecasting]], [[index]], [[log]]

## [2026-06-10] ingest | BiST — A Lightweight and Efficient Bi-Directional Model for Spatiotemporal Prediction (PVLDB 2025)

Ingest BiST paper (Jiaming Ma, Binwu Wang, Pengkun Wang, Zhengyang Zhou, Xu Wang, Yang Wang; USTC; PVLDB 18(6): 1663-1676, 2025; code: github.com/PoorOtterBob/BiST). BiST is a lightweight bidirectional spatiotemporal prediction model that challenges the input-label consistency assumption of existing STGNNs. Core innovations: (1) Spatiotemporal dynamics theory via GMRF proving optimal prediction = base prediction + diffusion-smoothed correction term; (2) Bidirectional architecture — forward MLP-only process for base prediction + backward residual correction process with label representations; (3) Residual decoupling module decomposing features into context (virtual clusters) and personalized features; (4) Adaptive diffusion kernel for residual smoothing. Evaluated on 13 datasets (up to 16,972-node XTraffic, 20-year XXLTraffic) vs 26 baselines — 8.13% improvement over SOTA with only 1.86% training time and 7.36% memory. Excels at handling sudden data surges and plummets (spatiotemporal deviation). PVLDB 2025 published paper — confidence: high.

Created pages: [[source-bist]], [[bist]], [[spatiotemporal-deviation]], [[bidirectional-spatiotemporal-prediction]]
Updated pages: [[traffic-forecasting]], [[index]], [[log]]

## [2026-06-08] ingest | Swift — An Autoregressive Consistency Model for Efficient Weather Forecasting (arXiv 2025)

Ingest Swift paper (Jason Stock, Troy Arcomano, Rao Kotamarthi; Argonne National Laboratory; arXiv:2509.25631, Sep 2025; code: github.com/stockeh/swift). Swift is the **first autoregressive consistency model for probabilistic weather forecasting**, achieving 39× faster inference than diffusion baselines with 75-day stable forecasts. Core innovations: (1) Temporal consistency model — applies TrigFlow-based continuous-time CM to autoregressive weather rollouts, reducing NFE from 20–40 to 1 per step; (2) CRPS autoregressive finetuning — first use of continuous ranked probability score loss through multi-step (K=1–8) rollouts to calibrate ensemble forecasts from a consistency model, balancing accuracy and ensemble dispersion; (3) 225M Swin Transformer with shifted windows, adaLN modulation, and dynamic time intervals (δi ∼ U{6, 12, 24}) for training regularization. Two-stage training: Muon optimizer for consistency pretraining (15M images) + AdamW for CRPS finetuning (5M images). Evaluated on ERA5 (WeatherBench 2) at 1.40625° resolution: 4 surface + 5 atmospheric variables × 13 pressure levels. Results: competitive with IFS ENS, underdispersive but stable to 75 days, captures realistic tropical cyclone tracks (Hurricane Laura), reproduces equatorial wave modes in Hovmöller diagrams, models seasonal cycles correctly. arxiv preprint — confidence: medium.

Created pages: [[source-swift]], [[swift]], [[autoregressive-consistency-models]], [[crps-autoregressive-finetuning]]
Updated pages: [[consistency-models]], [[generative-time-series-forecasting]], [[diffusion-models]], [[probability-flow-ode]], [[index]], [[log]]

## [2026-06-08] ingest | MMCKM — Micro-Macro Coupled Koopman Modeling on Graph for Traffic Flow Prediction (ICLR 2026 Poster)

Ingest MMCKM paper (ICLR 2026 Poster, accepted). MMCKM is the first framework to unify microscopic vehicle trajectory prediction and macroscopic traffic density evolution within a single Koopman operator-based architecture on dynamic vehicle graphs. Core innovations: (1) Vehicle-Centric Graph PDE — Lagrangian discretization of LWR advection-diffusion traffic flow equation directly onto vehicles as graph nodes, with constructive parameterization guaranteeing skew-symmetric advection (energy-preserving) and PSD diffusion (entropy-producing). This preserves high-frequency vehicle-level perturbations that Eulerian grid methods inherently lose. (2) Unified History-Free Koopman Modeling — both macro density and micro trajectory dynamics are lifted to linear observation spaces and evolved by time-invariant Koopman operators from a single snapshot, eliminating trajectory tracking overhead. Spectral alignment loss couples Koopman eigenvalues to PDE operator spectra (diffusion ↔ magnitude, advection ↔ rotation). (3) Physics-Guided Multi-Regime Micro Dynamics — Intent Discriminator (MoE) selects among 5 parameter-bounded Koopman operators (free flow, car-following, lane changing, merging, emergency) with distinct spectral radii, oscillation frequencies, and actuation bounds. CrossAttention-based Koopman control injects macro flow with ISS stability guarantees (no unbounded error growth). Evaluated on NGSIM and HighD: history-free trajectory prediction matches history-dependent SOTA (BAT, MS-STGCN, Vit-Traj), outperforms CV at all horizons (1–5s). Optimal operator interval 0.4s on HighD (ADE=1.65). Ablation: diffusion term critical (removal degrades macro 2.9–4.6%), Intent Discriminator 29% improvement at 1s, Koopman control 37% error reduction at 5s. KDE bandwidth sensitivity analysis reveals diffusion is only beneficial with physically meaningful density gradients. ICLR 2026 accepted paper — confidence: high.

Created pages: [[source-mmckm]], [[mmckm]], [[vehicle-centric-graph-traffic-pde]], [[micro-macro-coupled-koopman-modeling]], [[intent-discriminator-koopman]]
Updated pages: [[traffic-forecasting]], [[index]], [[log]]

## [2026-06-08] ingest | SSF — Spectral Sheaf Filtering: A Topological Approach to Spatio-Temporal Modeling (ICLR 2026 under review)

Ingest SSF paper (Anonymous authors, under double-blind review at ICLR 2026; code: github.com/anonymous-submisssion/SSF). SSF is the first framework to bridge cellular sheaf theory (algebraic topology) with spectral graph filtering for spatio-temporal traffic forecasting. Core innovations: (1) Cellular Sheaf Construction — assigns stalk vector spaces and learnable restriction maps to graph edges, encoding context-dependent, non-uniform information propagation that addresses fundamental GNN limitations; (2) Sheaf Laplacian — generalizes the combinatorial graph Laplacian, incorporating edge-specific transformation semantics; (3) Heat Kernel Spectral Filtering — eigendecomposition of sheaf Laplacian ($L_F = U\Lambda U^T$) followed by $e^{-\alpha\lambda}$ filtering that suppresses high-frequency noise while preserving low-frequency structural patterns. Evaluated on 5 benchmarks (METR-LA, PEMS-BAY, PEMS04, PEMS08, NAVER-Seoul) across 15/30/60min horizons, achieving SOTA on all, with dramatic long-horizon gains — NAVER-Seoul MAPE 1.03% at 15min vs. best baseline 8.32%. Ablation: spectral filtering is critical (removing it causes RMSE 20.72 vs 3.89 at 60min on NAVER-Seoul); optimal $k=3$ eigenvalues; stalk dimension $d$ trades accuracy for computation. Theoretical contribution: Theorem 1 proves sheaf Laplacian eigendecomposition properties, extending spectral graph theory to sheaf setting. Under review — confidence: medium.

Created pages: [[source-ssf]], [[ssf]], [[cellular-sheaf]], [[sheaf-laplacian]]
Updated pages: [[traffic-forecasting]], [[over-smoothing-in-gnns]], [[index]], [[log]]

## [2026-06-08] ingest | STBP — A General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank for Urban Continual Forecasting (ICLR 2026)

Ingest STBP paper (Aoyu Liu, Yaying Zhang; Tongji University; ICLR 2026 Poster; code: github.com/Aoyu-Liu/STBP). STBP proposes a continual spatio-temporal forecasting (CSTF) framework that pairs a frozen general-purpose backbone with a dynamically expanding contextual pattern bank. Core innovations: (1) Contextual Pattern Bank — three-component parametric memory (gating/scaling/attention key) that autonomously distinguishes node relevance and heterogeneity, expanding via concatenation as new nodes arrive; (2) Dual-Stream Linear Graph Attention (DLGA) — random-feature linear attention reducing O(N²)→O(N) with pattern bank parameters as additional keys; (3) FreNet — frequency-domain network extracting stable low-frequency components via FFT+learnable embedding. After initial joint training, backbone is frozen permanently; only pattern bank expands and fine-tunes. Evaluated on PEMS-Stream (7 periods), CA-Stream (4 periods, +254% node explosion), AIR-Stream (4 periods) vs 9 baselines. SOTA: 21.44%/21.93%/2.35% MAE reduction over EAC on PEMS/CA/AIR. Few-shot (10% data): MAE 13.58/17.11 on PEMS/CA. Efficiency comparable to lightweight CSTF methods despite richer backbone. Explicitly positions fixed-backbone+expandable-bank as stepping stone toward cross-domain ST foundation models.

Created pages: [[source-stbp]], [[stbp]], [[contextual-pattern-bank]], [[continual-spatio-temporal-forecasting]]
Updated pages: [[traffic-forecasting]], [[spatio-temporal-foundation-model]], [[index]], [[log]]

## [2026-06-08] ingest | PatchSTG — Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective (KDD 2025)

Ingest PatchSTG paper (Yuchen Fang, Yuxuan Liang, Bo Hui, Zezhi Shao, Liwei Deng, Xu Liu, Xinke Jiang, Kai Zheng; UESTC/HKUST-GZ/Auburn/ICT-CAS/NUS/PKU; KDD 2025; code: github.com/LMissher/PatchSTG). PatchSTG is the first framework to bridge KDTree spatial data management with Transformer patching for irregularly distributed traffic points. Core innovations: (1) Leaf KDTree — a novel KDTree variant ensuring all points land in leaf nodes, enabling balanced, non-overlapping spatial partitioning via BFS traversal; (2) Irregular Spatial Patching — three-stage pipeline: leaf KDTree → cosine-similarity padding → subtree backtracking, producing equal-occupancy patches with preserved spatial locality; (3) Dual Attention Encoder — interleaved depth attention (within-patch local) and breadth attention (cross-patch global) for efficient dynamic spatial modeling without information loss. Complexity O(max(P,R)·M·d) vs O(N²d) for dot-product methods. Evaluated on LargeST (SD 716 / GBA 2,352 / GLA 3,834 / CA 8,600 nodes, 12→12 forecasting) vs 10 baselines. SOTA on all four datasets. Up to 10× training speedup and 4× memory reduction on CA vs D2STGNN/DSTAGNN. Leaf KDTree is the most critical component; METIS and KMeans alternatives fail. PatchSTG is the only efficient dynamic spatial paradigm that simultaneously achieves no information loss, interpretability, and domain knowledge incorporation.

Created pages: [[source-patchstg]], [[patchstg]], [[irregular-spatial-patching]], [[leaf-kdtree]]
Updated pages: [[traffic-forecasting]], [[large-scale-spatial-temporal-graph]], [[index]], [[log]]

## [2026-06-08] ingest | CRAFT — Cross City Traffic Flow Generation via Retrieval Augmented Diffusion Model (NeurIPS 2025)

Ingest CRAFT paper (Yudong Li, Jingyuan Wang*, Xie Yu, Peiyu Wang; Beihang University; Qian Huang; Huawei; NeurIPS 2025; code: github.com/lyd1881310/CRAFT). CRAFT is a DDPM-based diffusion model for zero-shot cross-city traffic flow generation, the first work explicitly targeting this task. Core innovations: (1) Geographic Feature Alignment (GFA) — two complementary losses: Traffic Flow Alignment (TFA) aligns geographic representations with flow patterns, and Cross-City Alignment (CCA) uses optimal transport to project similar regions across cities into proximity without requiring explicit correspondence labels; (2) Retrieval-based Condition Augmentation (RCA) — retrieves similar historical flow patterns from source cities based on geo-representation similarity + temporal matching (month/day/hour), then aggregates via self-attention to augment diffusion conditions. Both modules are lightweight plug-ins requiring no backbone modifications. Evaluated on 4 real-world bicycle-sharing datasets (Chicago, DC, Toronto, NYC) in leave-one-out cross-city setup vs 8 baselines (GMEL, DFG, KSTDiff, CGAN, Diffwave, DiT, DDPM, CVAE). SOTA zero-shot: 59.7% improvement over baseline average, 22.5% over second-best GMEL, 61.5% over ordinary DDPM. Downstream utility: only 10.4% avg degradation vs. training on real data. Ablation: GFA most critical component (domain shift is dominant challenge), temporal embedding largest contributor within RCA. TFA+CCA t-SNE analysis confirms alignment. Sensitivity robust (max 8.8% fluctuation). Long-horizon (up to T=168): CRAFT most stable. Authors from Jingyuan Wang's lab at Beihang (same group as BIGCity, GTG, PDformer, HiFiNet).

Created pages: [[source-craft]], [[craft]], [[geographic-feature-alignment]], [[retrieval-based-condition-augmentation]], [[cross-city-traffic-flow-generation]]
Updated pages: [[traffic-forecasting]], [[spatio-temporal-foundation-model]], [[diffusion-models]], [[index]], [[log]]

Ingest UoMo paper (Chai, Zhang, Qi, Qiu, Li; Tsinghua University + China Mobile; KDD 2025 ADS Track, arXiv:2410.15322; code: github.com/tsinghua-fib-lab/UoMo). UoMo is the first universal foundation model for mobile traffic forecasting, unifying short-term prediction, long-term prediction, and distribution generation under a transformer-based diffusion model with task-oriented masking and contrastive context alignment. Evaluated on 9 real-world datasets (7 Chinese cities + Munich, Germany + Hangzhou) vs 13 baselines; avg 27.85% RMSE improvement in long-term prediction, 18.57% in short-term, 15.6% in generation. Deployed in production on China Mobile's Jiutian platform (Nanning, Guangxi): +25.3% served users in BS deployment, -40.7% equipment depreciation in BS sleep control. Scaling analysis: 5M-200M parameter variants, diminishing returns beyond 100M at fixed data size.

Created pages: [[source-uomo]], [[uomo]], [[jiutian-platform]], [[mobile-traffic-forecasting]], [[masked-diffusion-pre-training]], [[contrastive-diffusion-alignment]]
Updated pages: [[traffic-forecasting]], [[spatio-temporal-foundation-model]], [[index]], [[log]]

## [2026-06-08] ingest | UrbanMind — Urban Dynamics Prediction with Multifaceted Spatial-Temporal Large Language Models (KDD 2025)

Ingest UrbanMind paper (Anonymous, KDD 2025, August 3-7, Toronto; code: github.com/Yliu1111/UrbanMind). UrbanMind is a multifaceted spatio-temporal LLM for urban dynamics prediction that integrates three key innovations: (1) Muffin-MAE — a dual-encoder masked autoencoder with temporal (p_t=0.33), spatial (p_s=0.25), and global masking that captures inter-correlated dependencies across multiple urban dynamics (speed, inflow, demand); (2) Semantic-aware prompting with selective LLaMA3 fine-tuning — frozen early layers, query-only (W_q) updates in later layers, preserving pretrained LLM knowledge; (3) Test-time adaptation — a masked reconstruction mechanism with a shared-weight reconstructor G that adapts to distributional shifts at inference by recovering masked LLM embeddings through few-epoch updates. Evaluated on 9 datasets (3 cities — Shenzhen/Xi'an/Chengdu × 3 dynamics — traffic speed/taxi inflow/travel demand) vs 11 baselines (DYffusion, TGC-LSTM, GCRN, GAGCN, GATGPT, GCNGPT, ST-LLM, TPLLM, LLaMA3, STG-LLM, UrbanGPT). SOTA in both zero-shot and standard prediction. Key results: 8.5% lower MAE than UrbanGPT in Shenzhen→Xi'an cross-city transfer; ablation confirms Muffin-MAE is the most critical component; temporal masking ratio 0.33 and spatial masking ratio 0.25 are optimal; more multifaceted dynamics consistently improve RMSE. Training: 70.9s/epoch; test-time adaptation: 16.5s/epoch. Support: Yanhua Li (NSF IIS-1942680/CNS-1952085/DGE-2021871), Jun Luo (ITF ITP/012/25LP).

创建的页面：[[source-urbanmind]], [[urbanmind]], [[muffin-mae]], [[test-time-adaptation-st]]
更新的页面：[[urbangpt]], [[urbanpg]], [[spatio-temporal-foundation-model]], [[index]], [[log]]

## [2026-06-08] ingest | CoGenCast — A Coupled Autoregressive–Flow Generative Framework for Time Series Forecasting (ICML 2026)

Ingest CoGenCast paper (Yaguo Liu, Mingyue Cheng, Daoyu Wang, Xiaoyu Tao, Qi Liu; USTC State Key Laboratory of Cognitive Intelligence; ICML 2026, arXiv:2602.03564). CoGenCast is the first hybrid generative framework that couples pre-trained LLMs (Qwen3) with flow-matching mechanism for time series forecasting. Core innovations: (1) Reconfigures decoder-only LLMs into encoder-decoder backbone via attention topology modification only — bidirectional encoder for context understanding + causal decoder for autoregressive generation; (2) Continuous flow-matching mechanism conditioned on LLM-generated autoregressive representations — denoising decoder predicts interval-conditioned average velocity; (3) JVP-corrected optimization objective explicitly penalizes velocity curvature, enabling one-step generation (NFE=1). Evaluated on 10 benchmarks vs 8 baselines (LLM4TS, Time-LLM, FlowTS, CDPM, CSDI, TimeDART, PatchTST, Autoformer). ~11% MSE reduction vs LLM baselines, ~7% vs transformer baselines. Linear noise scheduler significantly outperforms cosine. Code: github.com/liuyaguo/_CoGenCast.

创建的页面：[[source-cogencast]], [[cogencast]], [[hybrid-llm-flow-matching-forecasting]], [[one-step-flow-generation]], [[average-velocity-modeling]]
更新的页面：[[flow-matching]], [[generative-time-series-forecasting]], [[sundial]], [[time-llm]], [[index]], [[log]]

## [2026-06-08] ingest | Sundial — A Family of Highly Capable Time Series Foundation Models (ICML 2025)

Ingest Sundial paper (Yong Liu\*, Guo Qin\*, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang, Jianmin Wang, Mingsheng Long; Tsinghua University, BNRist; ICML 2025, arXiv:2502.00816v4). Sundial is the first family of native and flexible time series foundation models, using Flow Matching-based TimeFlow Loss for generative pre-training on continuous-valued time series without discrete tokenization or parametric priors. Core innovations: (1) TimeFlow Loss — a parameterized flow-matching training objective for autoregressive Transformers to learn per-patch predictive distributions and sample flexibly; (2) Sundial model family (32M/128M/444M) with enhanced Transformer (RoPE, Pre-LN, FlashAttention, KV Cache, multi-patch prediction); (3) TimeBench — a 1.032 trillion time point pretraining corpus spanning finance, IoT, meteorology, and healthcare. Key results: SOTA zero-shot on TSLib (8 wins, avg MSE 7.57% better than Time-MoE), GIFT-Eval MASE #1 CRPS #2 across 23 datasets, FEV Leaderboard outperforms 70% of supervised methods, 35× faster inference than Chronos. Key findings: continuous patch tokenization > discrete tokenization; TimeFlow > diffusion > MSE for probabilistic forecasting; generative modeling mitigates mode collapse causing over-smooth predictions from MSE; test-time calibration via more samples/steps without retraining. Code: github.com/thuml/Sundial. Checkpoints: huggingface.co/thuml/sundial-base-128m.

创建的页面：[[source-sundial]], [[sundial]], [[timeflow-loss]], [[timebench]]
更新的页面：[[flow-matching]], [[generative-time-series-forecasting]], [[timesfm]], [[chronos]], [[patch-based-tokenization]], [[index]], [[log]]

## [2026-06-08] ingest | FlowTS — Time Series Generation via Rectified Flow (arXiv 2025)

Ingest FlowTS paper (Yang Hu, Xiao Wang, Zezhen Ding et al.; Westlake University, UW, HKUST, USTC, KTH, UNC Chapel Hill; arXiv:2411.07506v3, Feb 2025). FlowTS is the first model to apply rectified flow to time series generation, replacing diffusion models' iterative ODE/SDE solvers with straight-line transport in probability space. Core innovations: (1) ODE-based rectified flow learning geodesic paths with exact linear trajectory simulation; (2) adaptive sampling strategy inspired by exploration-exploitation trade-off, using $t^k$ scaling factor to balance noise adaptation and precision; (3) trend-seasonal decomposition (Trend Synthetic Layers + Fourier Synthetic Layers) for explicit periodic/long-term pattern modeling; (4) attention register tokens for global context aggregation; (5) Rotary Position Embedding (RoPE) for temporal position encoding; (6) seamless unconditional-to-conditional adaptation without retraining. Architecture: encoder-decoder Transformer with N encoder + M decoder blocks. Evaluated on 6 datasets: Stocks, ETTh, Energy, fMRI, Sines, MuJoCo (unconditional) + Solar, MuJoCo (conditional). Key results: unconditional Context-FID 0.019 (Stocks) and 0.011 (ETTh) vs previous best 0.067/0.061; solar forecasting MSE 213 (43.2% improvement over previous best 375); MuJoCo imputation MSE 7e-5 at 70% missing (74.1% reduction vs Diffusion-TS). Only 30 sampling steps (N=30) and 2,500 training iterations to surpass Diffusion-TS at 200 steps + 10,000 iterations. Ablation confirms RoPE (50k freq) + 128 attention registers as optimal combination. Code: github.com/UNITES-Lab/FlowTS.

创建的页面：[[source-flowts]], [[flowts]], [[rectified-flow-for-time-series]], [[adaptive-sampling-flow-matching]]
更新的页面：[[flow-matching]], [[generative-time-series-forecasting]], [[tsflow]], [[rectified-flow]], [[index]], [[log]]

## [2026-06-08] ingest | TSFlow — Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting (ICLR 2025)

Ingest TSFlow paper (Marcel Kollovieh, Marten Lienen, Leo Schwinn, David Lüdke, Stephan Günnemann; TU Munich; ICLR 2025, arXiv:2410.03024v2). TSFlow is the first Conditional Flow Matching (CFM) model for probabilistic time series forecasting. Core innovations: (1) Gaussian Process priors (SE/OU/PE kernels) instead of isotropic Gaussian to align prior distribution with temporal structure; (2) mini-batch optimal transport couplings for straighter probability paths; (3) dual conditioning strategies: unconditional model conditioned via Langevin dynamics (CPS) + guidance at inference, and conditional model with GP regression prior trained directly. Architecture: DiffWave-style with S4 layers, 3 residual blocks, ~176k params, Euler ODE 32 steps. Evaluated on 8 univariate datasets (Electricity, Exchange, KDDCup, M4-Hourly, Solar, Traffic, UberTLC, Wikipedia). Key results: SOTA CRPS on 6/8 datasets, GP priors outperform isotropic prior even at 4 NFE vs 16 NFE, PE kernel best for unconditional (LPS 6/8 datasets), OU kernel best for conditional forecasting. Code: github.com/marcelkollovieh/TSFlow.

创建的页面：[[source-tsflow]], [[tsflow]], [[gaussian-process-prior-flow-matching]], [[conditional-prior-sampling]]
更新的页面：[[flow-matching]], [[generative-time-series-forecasting]], [[index]], [[log]]

## [2026-06-08] ingest | SADI — Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios (AAAI 2025)

Ingest SADI paper (Islam, Tadepalli & Fern; Oregon State University; AAAI 2025, arXiv:2503.01737). SADI is a two-stage self-attention diffusion model for multivariate time series imputation that introduces the "partial blackout" missingness framework — where a subset of features is missing for consecutive time steps — unifying random missing, interpolation, complete blackout, and forecasting as special cases. Core innovations: (1) FDE (Feature Dependency Encoder) — 1-D dilated convolution + self-attention on feature dimension to capture time-aware feature correlations; (2) GTA (Gated Temporal Attention) — self-attention-based residual blocks (inspired by DiffWave/WaveNet) with GLU activation to model non-local temporal dependencies; (3) Two-stage imputation with learnable weighted combination — second GTA block refines first-stage imputation with noise data reintroduced as grounding signal, combined via attention-weight-derived dynamic coefficients. Two training strategies: SADI-RM (random missing) and SADI-MPB (mixed partial blackout: RM pre-training → alternating RM+PB fine-tuning). SOTA on all 4 datasets (AgAID, Air Quality, Electricity, NACSE) across all missing-feature counts in MSE and CRPS, outperforming CSDI, BRITS, SAITS, and MICE. Key finding: SADI handles high-dimensional datasets (370 Electricity features, 352 NACSE features) without reducing channels, while CSDI fails due to GPU memory constraints. Ablation confirms FDE critical for high-feature-correlation datasets (NACSE, Electricity), two-stage weighted combination essential. Code not public (anonymous.4open.science).

创建的页面：[[source-sadi]], [[sadi]], [[partial-blackout]], [[feature-dependency-encoder]], [[gated-temporal-attention]], [[two-stage-imputation]], [[mixed-partial-blackout-training]]
更新的页面：[[csdi]], [[cofill]], [[index]], [[log]]

## [2026-06-08] ingest | SSD-TS — Exploring the Potential of Linear State Space Models for Diffusion Models in Time Series Imputation (KDD 2025)

Ingest SSD-TS paper (Hongfan Gao et al., ECNU; KDD 2025, arXiv:2410.13338). SSD-TS is the first work to use the Mamba selective state space model as the denoising backbone in a conditional diffusion model for probabilistic time series imputation. Core innovations: (1) Replaces Transformer/S4 attention backbones with Mamba-based PNM (Parallel Mamba Block) as the core computation unit in DDPM noise prediction; (2) BAM (Bidirectional Attention Mamba) — bidirectional Mamba with integrated temporal attention for intra-channel dependency modeling; (3) CMB (Channel Mamba Block) — unidirectional Mamba on the channel dimension for inter-channel dependency modeling, shown superior to SENet-style channel attention; (4) SMM (Sequential Mamba Module) stacking BAM+CMB pairs. Two theoretical advantages over attention backbones: content-independent parameter updates (avoiding misleading attention weights from noisy input) and controllable frequency response via SSM transfer function. Linear complexity O(NCL) vs Transformer O(CL²). SOTA on MuJoCo (65.8% MSE improvement over SSSD at 90% missing), PhysioNet (best RMSE at all 3 missing rates), AQI (2nd best), and ETTm1 forecasting. Best CRPS in 3/4 probabilistic tasks. 87.57M params but 1.6× faster inference and 2.5× lower GPU memory than Transformer-backbone variant. Ablation confirms all three components essential; temporal attention has largest impact; CMB with Mamba >> Channel Attention. Code: github.com/decisionintelligence/SSD-TS.

创建的页面：[[source-ssdts]], [[ssd-ts]], [[bam]], [[cmb]]
更新的页面：[[mamba]], [[s-mamba]], [[csdi]], [[index]], [[log]]

## [2026-06-08] ingest | DiTS — Multimodal Diffusion Transformers Are Time Series Forecasters (arXiv 2602)

Ingest DiTS paper (Haoran Zhang*, Haixuan Liu*, Yong Liu*, Yunzhong Qiu, Yuxuan Wang, Jianmin Wang, Mingsheng Long; Tsinghua University; arXiv:2602.06597, Feb 6 2026). DiTS is a flow-matching-based probabilistic time series forecasting framework that adapts the Multimodal Diffusion Transformer (MM-DiT) architecture to multivariate time series. Core innovations: (1) Dual-stream MM-DiT backbone treating endogenous and exogenous variates as distinct modalities with independent processing streams and joint attention interaction; (2) Orthogonal dependency decomposition into Time Attention (intra-variate temporal dynamics, shared across streams) and Variate Attention (inter-variate cross-covariate modeling via joint attention with per-stream QKV projections); (3) Adaptive modulation via global conditioning embedding Z_y = covariate mean pooling + sinusoidal timestep, modulating all sub-layers through AdaLN scale/shift/gate parameters; (4) Flow matching paradigm with rectified flow (5-step Log-Normal sampling) replacing DDPM for efficient probabilistic generation. SOTA on FEV-Bench (WQL 0.070, MASE 0.601 — #1 among 11 models including zero-shot foundation models Chronos-2, Moirai-2.0, Sundial-Base), EPF deterministic forecasting (avg MSE 0.274, 10%+ improvement over TimeXer, especially dominant at 360-step horizon), and univariate LTSF (beats PatchTST on all 6 benchmarks). Key ablations: (a) DiTS attention (Joint + AdaLN) > TimeXer-style > iTransformer-style > Timer-XL-style; (b) DiTS conditioning (Joint attention + AdaLN) > Joint-only > Cross-only > AdaLN-only — the MM-DiT-style concurrent use of both is essential; (c) metric misalignment discovered: more inference steps (>5) increase MSE while CRPS stays flat, attributed to time series' low information density. Implemented PyTorch, single RTX 4090 (24GB). ⚠️ arXiv preprint — not yet peer-reviewed; code not public.

创建的页面：[[source-dits]], [[dits]], [[mm-dit-for-time-series]], [[dual-stream-attention-time-series]], [[flow-matching-forecasting]]
更新的页面：[[index]], [[log]]

## [2026-06-08] ingest | MiDDiR — Mixed Channel Dependency Diffusion Model with Retrieval Guidance for Time Series Forecasting (ICLR 2026 under review)

Ingest MiDDiR paper (Anonymous authors, under review at ICLR 2026). MiDDiR is a mixed channel dependency diffusion model that proposes two core innovations for probabilistic multivariate time series forecasting: (1) Mixed Channel Dependency — encoding historical time series in a channel-dependent manner (FC + multi-head attention) for informative cross-channel representation, while denoising in a channel-independent manner (DiT-like blocks with AdaLN) to decrease modeling complexity; (2) Retrieval Guidance — at inference time, retrieves similar historical patterns from the training set and analytically tilts the diffusion score estimation via exponential tilted distribution to enhance conditional sampling quality, especially for low-density regions of the data manifold. SOTA on 7 datasets (ETTh1/ETTh2/ETTm1/ETTm2/Electricity/Traffic/Weather) across 4 prediction lengths — CRPS avg 0.243 (surpassing NsDiff by ~21.9%), QICE avg 2.322 (surpassing TMDM by ~41.0%), MAE avg 0.336 (best among generative models). GIFT-Eval: MAPE #1 among non-fundamental models, MSE/NMRSE #3 of 39 total models. Ablation confirms CD encoding critical for high-dim (Traffic 862 channels: +10.8% MAE without CD), retrieval guidance most beneficial in high-dimensional settings. Retrieval overhead minimal: 0.054-0.176 ms per variable, 0.51%-0.86% extra sampling step time. ⚠️ Paper under review — not yet accepted; code not public.

创建的页面：[[source-middir]], [[middir]], [[mixed-channel-dependency]], [[retrieval-guidance]]
更新的页面：[[index]], [[log]]

Ingest TimeDiT paper (Defu Cao, Wen Ye, Yizhou Zhang, Yan Liu; USC; KDD 2025, arXiv:2409.02322). TimeDiT is the first model to unify DiT-style transformer backbone with diffusion probabilistic sampling as a time series proto-foundation model. Core innovations: (1) unified masking mechanism (random/block/stride/reconstruction) that enables a single model to handle forecasting, imputation, anomaly detection, and data generation without task-specific architecture changes; (2) physics-informed sampling via energy-based prior — PDE knowledge injected at inference time through Langevin dynamics with closed-form Boltzmann distribution solution (Theorem 3.1), requiring no model retraining; (3) AdaLN condition injection adapted for time series. Four model sizes (S 33M / B 130M / L 460M / XL 680M), pre-trained on Chronos dataset (~5B time points). SOTA zero-shot forecasting vs Moirai, SOTA uncertainty quantification with missing values/multi-resolution, zero-shot physics sampling surpasses fully trained baselines. Code not yet public.

创建的页面：[[source-timedit]], [[timedit]], [[timedit-masking]], [[timedit-physics-informed]]
更新的页面：[[diffusion-models]], [[dit]], [[timesfm]], [[index]], [[log]]

## [2026-06-08] ingest | StaTS — Spectral Trajectory Schedule Learning for Adaptive Time Series Forecasting with Frequency Guided Denoiser (arXiv 2603)

Ingest StaTS paper (Anonymous, arXiv 2603.00037, submitted concurrently with PAFM to ICML). StaTS proposes a diffusion forecasting framework that jointly optimizes noise scheduling and denoising through two components: STS (Spectral Trajectory Scheduler) learns data-adaptive noise schedules via frequency-domain regularized projected gradient descent; FGD (Frequency Guided Denoiser) estimates schedule-induced spectral distortion to modulate denoising strength. Core findings: (1) Learned schedules depart from standard monotonic templates, following a nonlinear pattern (sharp early rise → flat middle → steep end) that avoids over-compressing critical spectral information. (2) StaTS achieves SOTA CRPS (10.67%–17.43% improvement vs best baseline) and MAE on all 8 benchmarks (ECL, ILI, ETTh1/m1/m2, Traffic, SolarEnergy), with only 27 MB training memory (vs CSDI 3512 MB). (3) STS yield largest gains with small diffusion steps — T=10 reduces CRPS by 55.4% vs linear schedule on ETTm1. (4) Two theorems: PGD monotonic convergence to first-order stationarity (Thm 3.1), Lipschitz-stable forward drift under schedule updates (Thm 3.2). (5) STS converges to consistent schedule pattern regardless of initialization (linear/cosine/quadratic). SOTA in probabilistic time series forecasting with extremely low compute. Code: github.com/zjt-gpu/StaTS.

创建的页面：[[source-stats]], [[stats]], [[spectral-trajectory-scheduler]], [[frequency-guided-denoiser]]
更新的页面：[[diffusion-models]], [[nsdiff]], [[timegrad]], [[index]], [[log]]

## [2026-06-08] ingest | VisiFold — Long-Term Traffic Forecasting via Temporal Folding Graph and Node Visibility (arXiv 2026)

Ingest VisiFold paper (Zhiwei Zhang, Xinyi Du, Weihao Wang, Xuanchi Guo, Wenjuan Han; Beijing Jiaotong University / Beijing Normal University; arXiv 2603.11816). VisiFold addresses two critical bottlenecks in long-term traffic forecasting: snapshot-stacking inflation (resource overhead grows with time steps T) and cross-step fragmentation (temporal dependencies partitioned across separate snapshots). Core innovations: (1) Temporal Folding Graph (TFG) — collapses all attributes across T snapshots into a single token per node, compressing N×T×C input to N×T, eliminating the temporal module and cross-step message passing entirely; (2) Node Visibility — node-level masking (randomly hiding nodes from the encoder, following MAE design) + subgraph sampling (randomly partitioning remaining nodes into fixed-size subgraphs) serves as both efficiency mechanism and implicit regularizer; (3) VisiFold architecture — TFG → embedding fusion (spatial + time-of-day + day-of-week) → node visibility → Transformer encoder → MLP prediction head with Huber loss. SOTA on PEMS04, PEMS08, and SEATTLE across 24/36/48-step horizons vs. 12 baselines. Remarkable efficiency: ~7× training speedup and ~4× GPU memory saving vs STAEformer, inference <1s. Key finding: model maintains performance with up to 80% nodes masked (r=0.8), revealing substantial data redundancy. Spatial embeddings are the dominant accuracy driver; TFG >> spatial folding (SF). Node-level masking outperforms alternative strategies (AllZero/PartialZero/RandomValue). Subgraph interaction (leader tokens) and node-specific temporal embeddings provide no additional benefit. Code available at github.com/PlanckChang/VisiFold.

创建的页面：[[source-visifold]], [[visifold]], [[temporal-folding-graph]], [[node-visibility]]
更新的页面：[[ragc]], [[index]], [[log]]

## [2026-06-08] ingest | GAMMA-Net — GAT + Multi-Axis Mamba Interleaved for Traffic Forecasting (arXiv 2026)

Ingest GAMMA-Net paper (First Author et al., arXiv 2604.16859). GAMMA-Net introduces a novel interleaved GAT + multi-axis Mamba architecture for spatio-temporal traffic forecasting that addresses a persistent trilemma: no existing approach simultaneously offers efficient long-horizon memory, fully adaptive graph reasoning, and a lightweight footprint suitable for real-time ITS deployment. Core innovations: (1) Interleaved GAT-Mamba architecture — (GAT → Mamba_Temporal) × L → (GAT → Mamba_Spatial) × L with L=3, forming a closed-loop information flow where temporal understanding is distilled before each graph topology update and vice versa; (2) Dual-axis Mamba scans — temporal Mamba captures long-range dependencies with linear complexity while spatial Mamba disperses context-rich signals over the graph; (3) Dynamic graph re-weighting via GAT — suppresses obsolete influences (e.g., closed ramps) and amplifies emergent patterns (e.g., spill-back links) at each time step. SOTA on 6 benchmarks (METR-LA, PEMS-BAY, PEMS03/04/07/08) with up to 16.25% MAE reduction vs baselines. Ablation: removing both Mamba axes causes MAE to surge 44% (METR-LA) and 45% (PEMS-BAY) at 60min horizon. SVD analysis of state transition matrices confirms spatial component captures local dependencies while temporal component captures broader patterns. Code not yet public. Created source-summary, entity, and technique pages. Updated s-mamba and stgcn with cross-references.

创建的页面：[[source-gamma-net]], [[gamma-net]], [[interleaved-gat-mamba]]
更新的页面：[[s-mamba]], [[stgcn]], [[index]], [[log]]

## [2026-06-08] ingest | RSTIB-MLP — Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting (ICML 2025)

Ingest RSTIB-MLP paper (Min Chen, Guansong Pang, Wenjun Wang, Cheng Yan; Tianjin University & SMU; ICML 2025). The paper addresses whether simple MLP networks can achieve robust spatial-temporal forecasting under noise while maintaining efficiency. Key contributions: (1) Identifies the dual noise effect — under the sliding window mechanism, noise harms both input and target ends simultaneously, causing faster feature variance degradation; (2) Proposes the Robust Spatial-Temporal Information Bottleneck (RSTIB) principle, theoretically generalizing RGIB by lifting the Z–X–Y Markov assumption, explicitly minimizing noisy information from both ends through interaction information I(X;Y;Z) decomposition; (3) Instantiates RSTIB on pure MLP networks (RSTIB-MLP) with analytical KL-divergence regularization bounds and data reparameterization; (4) Introduces a knowledge distillation training regime with noise impact indicator α̂_i to dynamically balance regularization per time series; (5) Comprehensive evaluation on six datasets (PEMS04/07/08, LargeST, Weather2K-R, Electricity) under 0%–50% noise ratios, achieving superior robustness-efficiency trade-off vs 10+ baselines including STGNNs (GWN, STG-NCDE, TrendGCN) and MLP models (STID, FreTS). Created source-summary, entity (rstib-mlp), concept (rstib), and technique (noise-impact-indicator) pages. Updated ltsf-linear and timemixer with RSTIB cross-references.

创建的页面：[[source-rstib-mlp]], [[rstib-mlp]], [[rstib]], [[noise-impact-indicator]]
更新的页面：[[ltsf-linear]], [[timemixer]], [[index]], [[log]]

## [2026-06-08] ingest | TESTAM — A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts (ICLR 2024)

Ingest TESTAM paper (Hyunwook Lee, Seungmin Jin, Hyeshin Chu, Hongkyu Lim, Sungahn Ko; UNIST; ICLR 2024 Poster, accepted). TESTAM is the first MoE-based spatio-temporal attention model for traffic forecasting with three heterogeneous experts using different spatial modeling methods (identity matrix / learnable static graph / spatial attention), adaptively routed via memory-augmented gating networks. Core innovations: (1) time-enhanced attention — transfers attention domain from historical to future time steps, eliminating autoregressive error propagation; (2) memory-augmented gating with dual classification losses (worst-route avoidance + best-route selection) that solve the MoE routing freeze problem in regression; (3) in-situ spatial modeling — routing to the most appropriate spatial expert per traffic condition. SOTA on all 3 benchmarks (METR-LA, PEMS-BAY, EXPY-TKY), particularly strong on large-scale graphs (1,843-node EXPY-TKY) and non-recurring conditions. Only 224K params — fewest among all compared models. ICLR 2024 accepted paper. Created source-summary (source-testam), entity (testam), technique (time-enhanced-attention, memory-augmented-gating) pages. Updated traffic-forecasting with TESTAM reference.

创建的页面：[[source-testam]], [[testam]], [[time-enhanced-attention]], [[memory-augmented-gating]]
更新的页面：[[traffic-forecasting]], [[index]], [[log]]

## [2026-06-08] ingest | DPGNet — Dynamic Graph Prediction Network (ICLR 2026, under review)

Ingest DPGNet paper (Anonymous authors, under review at ICLR 2026). DPGNet is a spatiotemporal forecasting model with two core components: (1) AGL (Adaptive Graph Learner) — a plug-and-play dynamic graph generator using L stacked G-RNN units that combine self-attention with gating (update gate + reset gate) to capture time-varying implicit node relationships while suppressing weak connections; (2) ASL (Adaptive Season Learner) — a multi-scale temporal decomposition module that separates trend and seasonal components, extracts features via TCN (trend) and FFT→Linear→iFFT (seasonal), and constructs pattern-specific graphs per scale with bottom-up (seasonal) and top-down (trend) fusion. SOTA on 5 datasets (METR-LA, PEMS-Bay, PEMS08, Electricity, Weather) vs 5 baselines. AGL replacement experiments show 85% improvement rate across GWNet, PMC-GCN, STIDGCN, STGCN, WAVGCRN. Only 184K params — lowest among compared models. ⚠️ Paper under review — not yet accepted; findings preliminary. Created source-summary, entity (dpgnet), and technique (adaptive-graph-learner, adaptive-season-learner) pages. Updated gwnet and dcrnn with DPGNet cross-references.

创建的页面：[[source-dpgnet]], [[dpgnet]], [[adaptive-graph-learner]], [[adaptive-season-learner]]
更新的页面：[[gwnet]], [[dcrnn]], [[index]], [[log]]

## [2026-06-08] ingest | HEPHAESTUS — Hierarchical Periodic Heterogeneous Adaptive Spatio-Temporal Unified System (ICLR 2026, under review)

Ingest HEPHAESTUS paper (Anonymous authors, under review at ICLR 2026). HEPHAESTUS is a unified spatio-temporal traffic forecasting framework with three core innovations: (1) AMS-MoE — Adaptive Multi-Scale Mixture of Experts with Moving-Patch and noise-injected Top-K sparse routing for input-adaptive temporal scale selection (M=4 experts, K=2 optimal); (2) PTA — Periodic Temporal Attention with learnable daily (288 intervals) and weekly (2016 intervals) embedding matrices as queries; (3) HSA — Heterogeneous Spatial Attention with low-rank pattern library (r=8 optimal) and gated fusion balancing shared global vs. node-specific spatial patterns. SOTA on 6 benchmarks (METR-LA, PEMS-BAY, PEMS03/04/07/08) vs 15 baselines. 716K params, 5475MB GPU memory. Key insight: input-adaptive scale routing outperforms fixed multi-scale decomposition (TimeMixer/PathFormer). Ablation confirms AMS-MoE > PTA > HSA in importance. ⚠️ Paper under review — not yet accepted; findings preliminary. Created source-summary (source-hephestus), entity (hephestus), and technique (ams-moe, periodic-temporal-attention, heterogeneous-spatial-attention) pages. Updated phat and timemixer with cross-references.

创建的页面：[[source-hephestus]], [[hephestus]], [[ams-moe]], [[periodic-temporal-attention]], [[heterogeneous-spatial-attention]]
更新的页面：[[phat]], [[timemixer]], [[index]], [[log]]

## [2026-06-08] ingest | DST-Mamba — Decomposed Spatio-Temporal Mamba for Long-Term Traffic Prediction (AAAI 2025)

Ingest DST-Mamba paper (Sicheng He, Junzhong Ji, Minglong Lei; Beijing University of Technology; AAAI 2025). DST-Mamba is a decomposed spatio-temporal Mamba framework for long-term traffic prediction that addresses the spatio-temporal entanglement problem overlooked by pure temporal decomposition methods (Autoformer, FEDformer). Core innovations: (1) Temporal decomposition — moving average separates traffic series into trend (X_TR) and seasonal (X_SE) parts; (2) Seasonal component → bidirectional Mamba encoder in spatial perspective — node tokens via graph aggregation (EI = S·X_SE) + learnable adaptive spatial embeddings → forward + backward Mamba pipelines → FFN; (3) Trend component → multi-scale linear prediction — down-sampling to m scales + top-down mixing + per-scale linear predictors; (4) Final prediction Ŷ = Ŷ_SE + λŶ_TR. SOTA on 5 datasets (Traffic, PEMS03/04/07/08) vs 8 baselines (iTransformer, Crossformer, PatchTST, FEDformer, Autoformer, DLinear, S-Mamba, SOR-Mamba). Key findings: decomposition is foundational (+16.8% MSE gain); seasonal component more critical than trend (w/o Sea. Avg MSE 0.513 vs 0.119); bidirectional Mamba >> unidirectional. Computational complexity ~O(N) via Mamba's near-linear SSM, favorable efficiency vs Transformer's O(N²). Created source-summary (source-dst-mamba), entity (dst-mamba), concept (spatio-temporal-decomposition), and technique (multi-scale-linear-prediction) pages. Updated s-mamba and mamba with cross-references.

创建的页面：[[source-dst-mamba]], [[dst-mamba]], [[spatio-temporal-decomposition]], [[multi-scale-linear-prediction]]
更新的页面：[[s-mamba]], [[mamba]], [[index]], [[log]]

## [2026-06-08] ingest | RAST — Retrieval-Augmented Spatio-Temporal Framework for Traffic Prediction (AAAI 2026)

Ingest RAST paper (Weilin Ruan, Xilin Dang, Ziyu Zhou, Sisuo Lyu, Yuxuan Liang; HKUST-GZ / CUHK; AAAI 2026 / arXiv:2508.16623). RAST is the first framework to integrate RAG-style retrieval-augmented mechanisms with spatio-temporal modeling for traffic prediction. Core innovations: (1) Dual-dimension feature disentanglement — separate temporal (1D Conv) and spatial (graph transform) encoding for dimension-specific retrieval; (2) Spatio-Temporal Retrieval Store — FAISS-indexed dual memory banks with momentum EMA updates and info-theoretic scoring; (3) Context-aware query generator — residual FFN fusing spatio-temporal embeddings with cross-attention fusion of retrieved patterns; (4) Universal backbone predictor — compatible with frozen pre-trained STGNNs or simple MLPs. SOTA on 6 datasets (PEMS03/04/07/08 + LargeST SD/GBA), fastest training/inference on large-scale networks (154s/epoch on GBA, 3.7GB memory vs D2STGNN 45.1GB). Ablation: query generator most critical (MAE ↓25.6%), dual encoders indispensable (spatial ↓17.2%, temporal ↓21.2%). Information-theoretic foundation: external memory extends parameter-bound mutual information. Created source-summary (source-rast), entity (rast), concept (retrieval-augmented-spatio-temporal-forecasting), and technique (spatio-temporal-retrieval-store, dual-dimension-feature-disentanglement) pages. Updated gtr and ragc with cross-references.

创建的页面：[[source-rast]], [[rast]], [[retrieval-augmented-spatio-temporal-forecasting]], [[spatio-temporal-retrieval-store]], [[dual-dimension-feature-disentanglement]]
更新的页面：[[gtr]], [[ragc]], [[index]], [[log]]

## [2026-06-08] ingest | MetaDG — Meta Dynamic Graph for Traffic Flow Prediction (AAAI 2026)

Ingest MetaDG paper (Yiqing Zou, Hanning Yuan, Qianyu Yang, Ziqiang Yuan, Shuliang Wang, Sijie Ruan; Beijing Institute of Technology; AAAI 2026 / arXiv:2601.10328). MetaDG is a GCRU-based spatio-temporal prediction framework that simultaneously models dynamics and heterogeneity by extending dynamic modeling beyond spatial topology to meta-parameters. Core innovations: (1) Dynamic Node Generation (DNG) — time-gated fusion of static node embeddings and hidden states to generate per-timestep dynamic embeddings; (2) Spatio-Temporal Correlation Enhancement (STCE) — spatial cross-attention (SCE) followed by temporal smoothing (TCE) in fusion-before-smoothing order; (3) Dynamic Graph Qualification (DGQ) — qualifies edge reliability via cross-time-step similarity, producing adaptive scaling coefficients for proportional edge strengthening/weakening; (4) Meta-DGCRU — generates meta-parameters, raw adjacency matrix, and edge-weight adjustment matrix at each time step, pushing ST-isolated base models toward ST-unification. SOTA on PEMS03/04/07/08 across all metrics. Key framing: ST-isolated → ST-unification spectrum, where dynamics bridges spatial and temporal dimensions. Created source-summary (source-metadg), entity (metadg), concepts (meta-dynamic-graph, st-unification), and technique (dynamic-graph-qualification) pages. Updated gwnet, stgcn, dcrnn, and traffic-forecasting with cross-references.

创建的页面：[[source-metadg]], [[metadg]], [[meta-dynamic-graph]], [[st-unification]], [[dynamic-graph-qualification]]
更新的页面：[[gwnet]], [[stgcn]], [[dcrnn]], [[traffic-forecasting]], [[index]], [[log]]

## [2026-06-08] ingest | LSCD — Lomb–Scargle Conditioned Diffusion for Time Series Imputation (ICML 2025)

Ingest LSCD paper (Fons, Sztrajman, El-Laham, Ferrer, Vyetrenko, Veloso; J.P. Morgan AI Research / Cambridge / UBA / CONICET; ICML 2025 / arXiv:2506.17039). LSCD is the first method to integrate a differentiable Lomb–Scargle periodogram layer into a conditional score-based diffusion model for time series imputation. Core innovations: (1) differentiable Lomb–Scargle layer computes power spectra directly from irregularly-sampled data, eliminating interpolation/zero-filling artifacts that plague FFT-based methods; (2) attention-based spectrum encoder Espec encodes inter-frequency and inter-feature dependencies, injecting spectral conditioning into every denoising step; (3) two-stage training: standard score matching followed by spectral consistency loss LSCons that enforces frequency-domain alignment between imputed and observed signals. SOTA on synthetic sine waves (S-MAE ↓62.5% vs CSDI at 10% missing), PhysioNet (MAE 0.211 vs CSDI 0.219 at 10%), and PM2.5 (MAE 9.069 vs CSDI 9.670). Ablation confirms LS conditioning > Espec > LSCons in importance. Theoretical foundation: conditional entropy of reverse process strictly decreases with spectral conditioning. Created source-summary (source-lscd), entity/technique (lscd), concept (lomb-scargle-periodogram), and technique (spectral-consistency-loss) pages. Updated CSDI with LSCD as follow-up work.

创建的页面：[[source-lscd]], [[lscd]], [[lomb-scargle-periodogram]], [[spectral-consistency-loss]]
更新的页面：[[csdi]], [[index]], [[log]]

## [2026-06-08] ingest | NsDiff — Non-stationary Diffusion For Probabilistic Time Series Forecasting (ICML 2025 Spotlight)

Ingest NsDiff paper (Yifan Li, Xiongxiao Xu, Weiye Wang, Huiyu Li, Cungen Cao, Kai Shu; IIT/Chinese Academy of Sciences; ICML 2025 Spotlight). NsDiff is the first method to integrate Location-Scale Noise Model (LSNM) into DDPM-based probabilistic time series forecasting. Core innovations: (1) LSNM replaces DDPM's fixed unit variance assumption with learnable endpoint distribution N(f_φ(X), g_ψ(X)) where both mean and variance are estimated; (2) Uncertainty-Aware Noise Schedule (UANS) injects data-dependent time-varying variance β_t²g_ψ(X) + β_tα_tσ_Y₀ directly into the forward diffusion process; (3) Inference-time σ_Y₀ estimation via Vieta quadratic solving from σ_θ. SOTA on 9 real-world datasets (CRPS + QICE), QICE reduced 47.9% on ETTh1, 53.6% on ETTh2, 66.3% on Traffic (uncertainty variation=181.83). Unifies TMDM (g_ψ(X)=I) and TimeGrad (f_φ(X)=0, g_ψ(X)=I) as special cases. Created source-summary (source-nsdiff), entity (nsdiff), concept (location-scale-noise-model), and technique (uncertainty-aware-noise-schedule) pages. Updated diffusion-models (added NsDiff reference) and timegrad (added cross-reference to NsDiff).

创建的页面：[[source-nsdiff]], [[nsdiff]], [[location-scale-noise-model]], [[uncertainty-aware-noise-schedule]]
更新的页面：[[diffusion-models]], [[timegrad]], [[index]], [[log]]

## [2026-06-08] ingest | HiFiNet: Hierarchical Frequency-Decomposition GNN for Road Network Representation Learning (Ma, Wang & U, AAAI 2026)

Ingest HiFiNet paper (Jingtian Ma, Jingyuan Wang et al., Beihang University/University of Macau, AAAI 2026 / arXiv:2511.12507). HiFiNet is the first unified spatial-spectral GNN framework for road network representation learning. Core innovations: (1) Three-level hierarchy (segment→locality→region) with learnable cross-attention assignment matrices enabling localized frequency analysis; (2) Decomposition–updating–reconstruction paradigm that explicitly separates low-frequency (smooth global) and high-frequency (local variation) graph signals; (3) Topology-Aware Graph Transformer (TGT) blending global self-attention with local adjacency via learnable parameter α; (4) Theoretical proof that hierarchical projection acts as spectral low-pass filter, naturally separating frequency components and mitigating over-smoothing. SOTA on Beijing/Chengdu/Xi'an datasets across 4 tasks (next location prediction, label classification, destination prediction, route planning). Created source-summary (source-hifinet), entity (hifinet), concepts (road-network-representation-learning, graph-frequency-decomposition, over-smoothing-in-gnns), and technique (topology-aware-graph-transformer) pages. Updated traffic-forecasting (added Road Network Representation Learning section) and stgcn (added HiFiNet cross-links).

创建的页面：[[source-hifinet]], [[hifinet]], [[road-network-representation-learning]], [[graph-frequency-decomposition]], [[over-smoothing-in-gnns]], [[topology-aware-graph-transformer]]
更新的页面：[[traffic-forecasting]], [[stgcn]], [[index]], [[log]]

## [2026-06-08] ingest | MTP: Multimodal Urban Traffic Profiling with Modality Augmentation and Spectrum Fusion (Xiang et al., AAAI 2026)

Ingest MTP paper (Haolong Xiang, Peisi Wang et al., NUIST/NJU/Macquarie/Auckland, arXiv:2511.10218, submitted to AAAI 2026). MTP is the first multimodal framework for urban traffic state profiling (classification rather than forecasting). Core innovations: (1) Modality augmentation — converts numerical time series into visual (FFT frequency images + periodicity images) and textual (LLM-generated descriptions) modalities; (2) Frequency-domain unified processing — all three modalities processed via FFT → complex-valued frequency MLPs (numerical) or FIR filter + Hamming window + average pooling (visual/textual) → IFFT; (3) Cross-modal spectrum enhancement — text spectrum ⊙ visual spectrum; (4) Hierarchical contrastive fusion — supervised contrastive + InfoNCE unsupervised + JS divergence distribution alignment. SOTA on 6 datasets (Chinatown, Melbourne, PEMS-BAY, METR-LA, DodgerLoop, PEMS-SF) vs 8 baselines. Key ablation: visual branch removal drops DodgerLoop F1 from 0.585 to 0.105. Created source-summary, entity (mtp), concept (multimodal-traffic-profiling), and technique (modality-augmentation, hierarchical-contrastive-fusion) pages. Updated multimodal-time-series-forecasting (added MTP classification section) and traffic-forecasting (added Related Tasks section).

创建的页面：[[source-mtp]], [[mtp]], [[multimodal-traffic-profiling]], [[modality-augmentation]], [[hierarchical-contrastive-fusion]]
更新的页面：[[multimodal-time-series-forecasting]], [[traffic-forecasting]], [[index]], [[log]]

## [2026-06-08] ingest | FENCE: Spatial-Temporal Feedback Diffusion Guidance for Controlled Traffic Imputation (Mao et al., AAAI 2026)

Ingest FENCE paper (Xiaowei Mao, Huihu Ding et al., Beijing Jiaotong University/Aalborg University/中国地质大学/华东师范大学; AAAI 2026 / arXiv:2601.04572). FENCE is a dynamic feedback diffusion guidance method for spatio-temporal traffic data imputation that solves the fixed CFG guidance scale problem. Core innovations: (1) Feedback guidance — guidance scale λ(x_k, k) is dynamically computed from posterior likelihood p(c|x_k), increasing when imputed values diverge from observations; (2) Cluster-aware guidance — nodes clustered by spatial attention scores at each denoising step with cluster-level posteriors aggregated for stable estimation; (3) Two-stage training — first unconditional model for prior, then fine-tuned for conditional imputation. SOTA on PEMS04/07/08 with 80% missing rate, MAPE avg 6.26% improvement over CSDI/PriSTI/ImputeFormer. Created source-summary, entity (fence), and technique (feedback-diffusion-guidance, cluster-aware-guidance) pages. Updated CSDI, PriSTI, classifier-free-guidance, diffusion-models with FENCE cross-references.

创建的页面：[[source-fence]], [[fence]], [[feedback-diffusion-guidance]], [[cluster-aware-guidance]]
更新的页面：[[csdi]], [[pristi]], [[classifier-free-guidance]], [[diffusion-models]], [[index]], [[log]]

## [2026-06-04] ingest | E²-CSTP: Causal Spatio-Temporal Prediction — An Effective and Efficient Multi-Modal Approach (Huang et al., NeurIPS 2025)

Ingest E²-CSTP paper (Yuting Huang, Ziquan Fang et al., Zhejiang University; NeurIPS 2025). E²-CSTP integrates cross-modal attention (BERT+CNN for text+image), dual-branch causal inference (backdoor adjustment blocking Xst←S→Yst confounders), and GCN+Mamba hybrid ST encoder. Core innovations: (1) DeepSHAP-based causal matrix construction blended with prior graph; (2) Dual-branch design — main branch pure ST, auxiliary multi-modal fused → MLP combined output; (3) Linear O(B·T·N²·d) complexity via GCN+Mamba vs Transformer's O(B·T²·N²·d). SOTA on 4 datasets (Terra, BjTT, GreenEarthNet, BikeNYC) with up to 9.66% MAE improvement and 17.37%-56.11% efficiency gains. Created source-summary and entity pages. Updated spatio-temporal-foundation-model (Multi-Modal entry) and multimodal-time-series-forecasting (cross-link).

创建的页面：[[source-e2-cstp]], [[e2-cstp]]
更新的页面：[[spatio-temporal-foundation-model]], [[multimodal-time-series-forecasting]], [[index]], [[log]]

## [2026-06-04] ingest | STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning (Ni et al., 2026)

Ingest STReasoner paper (Juntong Ni, Shiyu Wang et al., Emory/Microsoft/Griffith; arXiv 2026). STReasoner is the first TS-LM designed for explicit spatio-temporal reasoning — answering queries like "Which source node caused the congestion?" by tracing propagation paths through spatial dependencies and temporal dynamics. Core innovations: (1) ST-Bench — a 4-task benchmark (etiological/entity/correlation/in-context forecasting) built via network SDE-based multi-agent data synthesis; (2) STReasoner architecture — lightweight MLP TS encoder interleaved with LLM text tokens, trained in 3 stages (alignment + SFT-CoT via rejection sampling + spatial-aware RL); (3) S-GRPO — contrastive spatial reward that compares performance with/without spatial structure, explicitly incentivizing spatially grounded reasoning. Key results: 95.65% etiological ACC (GPT-5.2 text: 83.09%), 98.82% zero-shot on real-world CausalRivers, at 0.004× cost ($0.27 vs $22.48). Created source-summary, entity (streasoner), and concept (spatio-temporal-reasoning) pages. Updated time-llm (evolution link) and multimodal-time-series-forecasting (new STReasoner section).

创建的页面：[[source-streasoner]], [[streasoner]], [[spatio-temporal-reasoning]]
更新的页面：[[time-llm]], [[multimodal-time-series-forecasting]], [[index]], [[log]]

## [2026-06-04] ingest | AllSpark: A Multimodal Spatio-Temporal General Intelligence Model with Ten Modalities via Language as a Reference Framework (Shao et al., 2024)

Ingest AllSpark paper (Run Shao, Cheng Yang et al., Central South University; arXiv 2024, revised Jan 2025). AllSpark is a unified multimodal model integrating 10 spatio-temporal modalities (1D: language/code/table; 2D: RGB/SAR/MSI/HSI/graph/trajectory; 3D: point cloud) using Language as Reference Framework (LaRF) principle. Core innovations: (1) LaRF — language serves as universal alignment anchor balancing cohesion and autonomy across heterogeneous modalities; (2) Modal bridge (Perceiver-based) — learnable query vectors project diverse modality tokens into unified 4096-dim language space; (3) Training-free few-shot learning — 95.58% 5-way 1-shot on UC-Merced RGB without meta-learning, up to 41.82% improvement; (4) Cross-modality adaptability — competitive with SOTA across all 10 modalities despite no expert knowledge. Created source-summary, entity (allspark), and concept (language-as-reference-framework) pages. Updated spatio-temporal-foundation-model (added as multi-modal entry alongside MoST) and multimodal-time-series-forecasting (new AllSpark section). First ingestion covering remote sensing-specific modalities (SAR, hyperspectral, multispectral) in the wiki.

创建的页面：[[source-allspark]], [[allspark]], [[language-as-reference-framework]]
更新的页面：[[spatio-temporal-foundation-model]], [[multimodal-time-series-forecasting]], [[index]], [[log]]

## [2026-06-04] ingest | DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting (Cachay et al., NeurIPS 2023)

Ingest DYffusion paper (Salva Rühling Cachay, Bo Zhao, Hailey Joren, Rose Yu; UC San Diego, NeurIPS 2023). DYffusion proposes a novel dynamics-informed diffusion model for spatiotemporal forecasting that replaces the standard Gaussian noise-based diffusion process with temporal interpolation and forecasting. Core innovations: (1) Two-stage training — train a stochastic time-conditioned interpolator network to reconstruct intermediate snapshots, then freeze it and train a deterministic forecaster network (diffusion backbone) to predict the final snapshot from interpolated states; (2) Diffusion-dynamics coupling — diffusion steps map directly to physical time steps via a schedule, operating entirely in data space (never in noise space); (3) Cold Sampling inference — adapted from Cold Diffusion, alternating forecast+interpolate steps, theoretically proven equivalent to Euler's method for solving an implicit dynamical system ODE; (4) Constant training memory (only 3 snapshots: x_t, x_{t+i}, x_{t+h}) vs video diffusion models' O(h) memory; (5) Few diffusion steps (<50 vs 1000+ for MCVD). Evaluated on SST (sea surface temperatures), Navier-Stokes flows, and Spring Mesh systems — outperforms Dropout/DDPM/MCVD baselines in CRPS/MSE/SSR, with significant computational efficiency gains. Key ablation: Cold Sampling >> Naive Sampling (SST CRPS 0.181 vs 0.681), interpolator dropout indispensable (disabled: CRPS 0.181 → 0.320). Created source-summary, entity, and technique pages; added cross-links to diffusion-models, generative-time-series-forecasting.

创建的页面：[[source-dyffusion]], [[dyffusion]], [[cold-sampling]]
更新的页面：[[diffusion-models]], [[generative-time-series-forecasting]], [[index]], [[log]]

## [2026-06-04] ingest | Time-LLM: Time Series Forecasting by Reprogramming Large Language Models (Jin et al., ICLR 2024)

Ingest Time-LLM paper (Ming Jin, Shiyu Wang et al., Monash/Ant Group/IBM/Griffith/Alibaba/HKUST-GZ, ICLR 2024 / arXiv:2310.01728). Time-LLM is the first framework to repurpose frozen LLMs (Llama-7B/GPT-2) for general time series forecasting via model reprogramming — no backbone fine-tuning required. Core innovations: (1) Patch Reprogramming — multi-head cross-attention with learned text prototypes aligns time series patches to the LLM's pretrained word embedding space, each patch represented by a sparse combination of text prototypes like "periodic", "seasonal", "quantile"; (2) Prompt-as-Prefix (PaP) — natural language prompts with dataset context, task instructions, and input statistics guide the frozen LLM's transformation of reprogrammed patches; (3) output patches from LLM flattened + projected to forecasts. Only ~6.6M trainable parameters (0.2% of Llama-7B). SOTA on long-term (7/8 datasets MSE best), short-term (M4 benchmark), 10%/5% few-shot, and zero-shot cross-domain forecasting; scaling law preserved across backbones. PaP + Patch Reprogramming jointly essential: ablating either degrades >8% (full-shot) and >17% (few-shot). Key ablation: input statistics are the most critical PaP component (10.2% MSE impact). Created source-summary, entity, technique (patch-reprogramming, prompt-as-prefix), and concept (model-reprogramming) pages; added wikilinks from CVPE, source-cvpe-2025, multimodal-time-series-forecasting.

创建的页面：[[source-time-llm]], [[time-llm]], [[patch-reprogramming]], [[prompt-as-prefix]], [[model-reprogramming]]
更新的页面：[[cvpe]], [[source-cvpe-2025]], [[multimodal-time-series-forecasting]], [[index]], [[log]]

## [2026-06-01] query | 专题：扩散模型频域理论

创建分析页面 [[diffusion-frequency-domain-theory]]，以倒读法问题演化叙事综合 8 个源文件：谱偏置两种解释（经典 F-Principle 架构偏置 vs Wang & Pehlevan 数据驱动反比方差谱定律）、DDPM 前向加噪制造的频率层级（Falck EqualSNR）、频域噪声控制与 SAGD（归纳偏置塑造）、EqualSNR 的公平加噪与 FlippedSNR 失败、推理阶段 SNR-t Bias 与 DCW 小波域校正、范式转移（FreqFlow 双分支频率引导、SpecSTG 图谱域扩散、FEDformer 频域注意力、Crabbé 频域 SDE 时间序列谱系）。补充了图书管理员发现的 20+ 篇外部文献：Rissanen 逆热方程扩散、Dieleman "谱自回归"框架、Kadkhodaie 几何自适应调和偏置、Guth 小波 SGM、Spectrum Matching 感知价值论证、PaCoDi/StaTS/SpectFlow 时间序列频域扩散新进展。

创建的页面：[[diffusion-frequency-domain-theory]]
更新的页面：[[index]], [[log]]

## [2026-06-01] query | 专题：时空基础模型全景

创建分析页面 [[spatio-temporal-foundation-model-landscape]]，综合 12 个源文件从设计哲学（三重分裂：Scaling 派/先验注入派/解耦派）、架构范式（六条路线：纯 Transformer、Diffusion Transformer、Transformer+GNN、LLM-Based、因子化两阶段、VFM 重编程）、预训练策略（四象限：联合/解耦 × 重建/预测）、泛化机制（零样本/少样本/持续学习）四维深度解剖时空基础模型领域。揭示深层张力与收敛趋势：解耦派和 Scaling 派正收敛到同一洞察（时间和空间应分别处理）；Prompt → Zero-Shot 适配机制正在简化；单模态 → 多模态是下一代方向。补充了外部文献发现：UniSTD、ST-VFM（VFM 重编程路线）、天气/地球系统 FM 谱系、5 篇综述文献索引。

创建的页面：[[spatio-temporal-foundation-model-landscape]]
更新的页面：[[index]], [[log]]

## [2026-06-01] ingest | UrbanVerse: Learning Urban Region Representation Across Cities and Tasks (Sun et al., arXiv 2026)

Ingest UrbanVerse paper (Fengze Sun, Egemen Tanin, Shanika Karunasekera, Zuqing Li, Flora D. Salim, Jianzhong Qi; University of Melbourne / UNSW, arXiv:2602.15750, Feb 17 2026). UrbanVerse is a foundation-style model for cross-city urban region representation learning and cross-task urban analytics. Core paradigm shift from city-centric to region-centric: partitions cities into 150m hexagonal grid cells, uses Node2vec random walks + Transformer mask-reconstruct (CELearning) to learn transferable cell embeddings that aggregate into region embeddings via AdaRegionGen (inherited from FlexiReg). Cross-task learning via HCondDiffCT — a heterogeneous conditional diffusion regression module that models p(y|h,u) using region-conditioned prior retrieval (RegCondP, Top-5 cosine-similar neighbors) and task-conditioned denoising (TaskCondD, element-wise modulation of timestep+task embeddings). 3 cities (NYC/CHI/SF) × 6 tasks (Crime/Check-in/Service Call/Population/Carbon/Nightlight), 7 baselines. Cross-city: UrbanVerse consistently outperforms all baselines, R² gains up to +35.89% (SF crime) over FlexiReg. Suburban generalization: Staten Island population R²=0.781 vs FlexiReg 0.609 (+28.2%). HCondDiffCT as plug-and-play module: GURPP-DiffCT nightlight R² 0.035→0.171 (+388.6%), UrbanCLIP-DiffCT carbon R² 0.021→0.204 (+871.4%). Only uses POI+neighbor features (15-dim), minimal feature set maximizes cross-city transfer. Complementary to UrbanFM (traffic sequences vs. region attributes), both arXiv Feb 2026. Limitations: no temporal dimension, requires ≥2 training cities, only continuous regression tasks, US-city training bias. Created source-summary and technique pages; added reciprocal wikilinks to 4 existing pages (urbanfm, urbangpt, urbanpg, spatio-temporal-foundation-model).

创建的页面：[[source-urbanverse]], [[urbanverse]]
更新的页面：[[urbanfm]], [[urbangpt]], [[urbanpg]], [[spatio-temporal-foundation-model]], [[index]], [[log]]

## [2026-06-01] ingest | UrbanPG: An Efficient Framework with Personalized Context and General Backbone Interaction for Urban Spatio-Temporal Learning (Liu & Zhang, AAAI 2026)

Ingest UrbanPG paper (Aoyu Liu, Yaying Zhang; Tongji University, AAAI 2026). UrbanPG is an efficient and scalable urban spatio-temporal learning framework that decouples personalized context prompts (time/spatial embeddings with random perturbation regularization) from a general backbone (STCA linear spatio-temporal context attention via Performers' random feature mapping, O(N·d²) complexity). This decoupling allows UrbanPG to simultaneously address three challenges: large-scale forecasting (8600 nodes, SOTA with 48-72% efficiency gains over PatchSTG), few-shot generalization (10% training data, fine-tune only prompts), and continual learning (freeze backbone, expand spatial prompts for new nodes — zero forgetting). Three learning paradigms: standard, pre-training+fine-tuning, and continual prompt expansion. Evaluated on 8 datasets across 3 tasks (LargeST SD/GBA/GLA/CA, CA-D3/D5, PEMS-Stream, AIR-Stream). Ablation shows spatial context >> temporal context > random perturbation > STCA in importance. Only limitation: cannot support multi-task parallel training. Created source-summary and technique pages; added reciprocal wikilinks to 7 existing pages (spatio-temporal-foundation-model, traffic-forecasting, urbanfm, urbangpt, opencity, bigcity, linear-attention-unified-framework).

创建的页面：[[source-urbanpg]], [[urbanpg]]
更新的页面：[[spatio-temporal-foundation-model]], [[traffic-forecasting]], [[urbanfm]], [[urbangpt]], [[opencity]], [[bigcity]], [[linear-attention-unified-framework]], [[index]], [[log]]

## [2026-06-01] ingest | FactoST: Learning to Factorize and Adapt — A Versatile Approach Toward Universal Spatio-Temporal Foundation Models (Zhong et al., NeurIPS 2025 / arXiv 2026)

Ingest FactoST-v2 paper (Siru Zhong, Junjie Qiu, Yangyu Wu, Yiqiu Liu, Yuanpeng He, Zhongwen Rao, Bin Yang, Chenjuan Guo, Hao Xu, Yuxuan Liang; HKUST-GZ / Peking U / Huawei / ECNU, arXiv:2601.12083, Jan 2026 — journal extension of NeurIPS 2025 conference version). FactoST-v2 proposes the Pattern Factorization Hypothesis: effective ST generalization requires decoupling domain-invariant temporal dynamics from domain-specific spatial contexts. Two-stage framework: (1) UTP (Universal Temporal Pretraining) — minimalist encoder-only Transformer with random sequence masking, p-RoPE, gated attention, and quantile prediction, pretrained on 11B+ time points across 8 domains; (2) STA (Spatio-Temporal Adaptation) — lightweight adapter (STMF + STF + DSPA + CMR) injecting spatial awareness into frozen backbone. Achieves linear O(N) complexity vs. quadratic O(N²) for joint STFMs. SOTA on few-shot/full-shot/zero-shot across 9 benchmarks. Key findings: 10% labeled data nears full-shot performance; model scales (2.5M-30.4M params); architectural disentanglement trumps brute-force joint pretraining for zero-shot generalization. Same corresponding author (Yuxuan Liang) as UrbanFM. Created source-summary and technique pages; added reciprocal wikilinks to 6 existing pages.

创建的页面：[[source-factost]], [[factost]]
更新的页面：[[spatio-temporal-foundation-model]], [[traffic-forecasting]], [[urbanfm]], [[uniflow]], [[bigcity]], [[index]], [[log]]

## [2026-06-01] ingest | BIGCity: A Universal Spatiotemporal Model for Unified Trajectory and Traffic State Data Analysis (Xie Yu et al., arXiv 2024)

Ingest BIGCity paper (Xie Yu, Jingyuan Wang, Yifan Yang, Qian Huang, Ke Qu; Beihang University / Huawei, arXiv:2412.00953, Dec 2024). BIGCity is the first MTMD (Multi-Task, Multi-Data modality) spatio-temporal model — a single model simultaneously handling individual-level trajectory data and population-level traffic state data across 8 heterogeneous tasks. Core innovations: (1) ST-unit — a novel unified representation (static road features + dynamic traffic state + timestamp) that expresses both trajectories and traffic states as identical-format sequences, eliminating the modality gap; (2) ST Tokenizer — four-module pipeline (static GAT + dynamic GAT + fusion cross-attention + temporal MLP with δ_τ) converting ST-units to LLM-consumable tokens; (3) VMTP — GPT-2 backbone (1.5B) with LoRA (r=8) processing task-oriented prompts (text instruction + ST-tokens + [CLAS]/[REG] placeholders) to unify heterogeneous tasks without per-task fine-tuning; (4) two-stage training — masked ST-unit reconstruction pre-training → task-oriented prompt tuning. 3 cities (Beijing 101M trajectories/40K segments, Xi'an 385K/5K, Chengdu 560K/6K), 8 tasks, 18 baselines, SOTA across all. Cross-city generalization: BJ→XA/CD <7% avg performance loss. Ablation ranking: prompt removal (10.5% impact) >> dynamic encoder > static encoder. BIGCity is the first model to prove that trajectories and traffic states can share an atomic representation (ST-unit), and that cross-modal multi-task training yields stronger mutual benefit than same-modal. Created source-summary and technique pages; added reciprocal wikilinks to 8 existing pages (spatio-temporal-foundation-model, traffic-forecasting, urbangpt, unist, uniflow, opencity, urbandit, gpt-st).

创建的页面：[[source-bigcity]], [[bigcity]]
更新的页面：[[spatio-temporal-foundation-model]], [[traffic-forecasting]], [[urbangpt]], [[unist]], [[uniflow]], [[opencity]], [[urbandit]], [[gpt-st]], [[index]], [[log]]

## [2026-06-01] ingest | UrbanFM: Scaling Urban Spatio-Temporal Foundation Models (Chen et al., arXiv 2026)

Ingest UrbanFM paper (Wei Chen, Yuqian Wu, Junle Chen, Xiaofang Zhou, Yuxuan Liang; HKUST(GZ)/HKUST, arXiv:2602.20677, Feb 2026). UrbanFM is the first scaling-centric urban spatio-temporal foundation model that systematically addresses three dimensions grounded in first-principles: (1) WorldST — data scaling with 100+ cities, 8 domains, 1B+ data points (33-145× larger than UniST/OpenCity/BigCity); (2) MiniST — computation scaling via KD-Tree greedy capacity-constrained clustering unifying sensor and grid data into learnable tokens; (3) UrbanFM architecture — minimalist factorized spatio-temporal attention with ST-RoPE (T-RoPE + S-RoPE) and generative modeling objective unifying forecasting and imputation. Evaluated on EvalST (12 datasets, 22 baselines, 4 countries, 7 cities): zero-shot MAPE improvement 39.0-70.2% over existing foundation models, surpasses full-shot experts in long-term sensor forecasting; few-shot 28.2-65.2% further gains; cross-task imputation without imputation training; power-law scaling with data/model; 4× faster than Chronos, 10× faster than TimeMoE. Created source-summary and entity pages; added reciprocal wikilinks to 6 existing pages (spatio-temporal-foundation-model, opencity, unist, uniflow, urbangpt, traffic-forecasting).

创建的页面：[[source-urbanfm]], [[urbanfm]]
更新的页面：[[spatio-temporal-foundation-model]], [[opencity]], [[unist]], [[uniflow]], [[urbangpt]], [[traffic-forecasting]], [[index]], [[log]]

## [2026-06-01] ingest | USTD: Towards Unifying Diffusion Models for Probabilistic Spatio-Temporal Graph Learning (Hu et al., SIGSPATIAL 2024)

Ingest USTD paper (Junfeng Hu, Xu Liu, Zhencheng Fan, Yuxuan Liang, Roger Zimmermann; NUS/UTS/HKUST-GZ, SIGSPATIAL 2024, arXiv:2310.17360). USTD is the first framework to unify spatio-temporal forecasting and kriging into a single diffusion model. Core innovations: (1) Pre-trained GWNet-style ST encoder with graph sampling (80% nodes) and masking (75%, MAE-style) — decoupled from denoiser training, solving the "diffusion STG cannot beat deterministic baselines" problem that plagued CSDI/PriSTI/DiffSTG; (2) Task-specific denoising decoders — TGA (Temporal Gated Attention) for forecasting with cross-attention on temporal axis + self-attention across nodes + gated fusion; SGA (Spatial Gated Attention) for kriging with cross-attention on spatial axis; (3) TCN without zero-padding compresses conditions to low-dimensional latent space, enabling ~47% faster inference than CSDI. USTD surpasses all probabilistic baselines on 4 datasets (PEMS-03/BAY, AIR-BJ/GZ), and overturns the consensus by surpassing nearly all deterministic baselines on forecasting (CRPS ↓12% on PEMS-BAY). Kriging: USTD beats ALL baselines (MAE ↓10.5% on AIR-GZ). Created source-summary and entity pages; added reciprocal wikilinks to 6 existing pages (diffstg, specstg, traffic-forecasting, generative-time-series-forecasting, spatio-temporal-foundation-model, uniflow).

创建的页面：[[source-ustd]], [[ustd]]
更新的页面：[[diffstg]], [[specstg]], [[traffic-forecasting]], [[generative-time-series-forecasting]], [[spatio-temporal-foundation-model]], [[uniflow]], [[index]], [[log]]

## [2026-05-31] ingest | UniFlow: A Foundation Model for Unified Urban Spatio-Temporal Flow Prediction (Yuan et al., arXiv 2024)

Ingest UniFlow paper (Yuan Yuan, Jingtao Ding, Chonghua Han, Zhi Sheng, Depeng Jin, Yong Li, Tsinghua University, arXiv:2411.12972). UniFlow is the first foundation model to unify grid-based and graph-based urban spatio-temporal flow prediction into a single Transformer model. Core innovations: (1) multi-view spatio-temporal patching — 3D-CNN for grid data + METIS graph partitioning for graph data, converting heterogeneous formats into unified sequences; (2) Spatio-Temporal Memory Retrieval Augmentation (ST-MRA) — four structured learnable memory modules (time-domain, frequency-domain, time-spatial, frequency-spatial) that store shared spatio-temporal patterns and generate personalized prompts via cosine-similarity retrieval, enabling cross-learning across different data types; (3) encoder-decoder Transformer with MRA-augmented prompts. 9 datasets (6 grid + 3 graph, >10K nodes), 9.1% average RMSE improvement over best baselines in short-term prediction, 11.9% in long-term, superior few-shot/zero-shot capability, near-immune to noise perturbations (10% noise → only 10% RMSE increase vs baseline 148%). Same lab as UrbanDiT (NeurIPS 2025) and UniST (KDD 2024). Created source-summary and technique pages; added reciprocal wikilinks to 10 existing pages (spatio-temporal-foundation-model, traffic-forecasting, urbandit, urbangpt, opencity, gpt-st, stgcn, dcrnn, gwnet, channel-independence).

创建的页面：[[source-uniflow]], [[uniflow]]
更新的页面：[[spatio-temporal-foundation-model]], [[traffic-forecasting]], [[urbandit]], [[urbangpt]], [[opencity]], [[gpt-st]], [[stgcn]], [[dcrnn]], [[gwnet]], [[channel-independence]], [[index]], [[log]]

## [2026-05-31] ingest | UrbanGPT: Spatio-Temporal Large Language Models (Li et al., KDD 2024)

Ingest UrbanGPT paper (Zhonghang Li, Lianghao Xia, Jiabin Tang et al., HKU/SCUT/Baidu, KDD 2024 / arXiv:2403.00813). UrbanGPT is the first spatio-temporal large language model, integrating a multi-level gated dilated convolution encoder (graph-free) with Vicuna-7b instruction-tuning paradigm. Core innovations: (1) ST dependency encoder avoids graph structures entirely for zero-shot flexibility; (2) textual POI descriptions let LLM infer spatial semantics instead of using adjacency matrices; (3) regression layer resolves LLM's discrete-output vs continuous-regression mismatch by having LLM output rich hidden vectors Γ (not numbers), fused with raw ST features H to produce precise predictions. Four datasets (NYC-taxi/bike/crime, CHI-taxi), zero-shot superiority over 10 baselines (NYC-taxi inflow MAE=6.16 vs best baseline 9.75, ↓36.8%), crime prediction Recall=0.34 vs baselines≈0. Key limitations: 7B params, 174s inference per sensor, LLM backbone dependency, no graph exploitation when topology is available. Created source-summary and technique pages; added reciprocal wikilinks to ST foundation model, GPT-ST, OpenCity, traffic-forecasting, and related pages.

创建的页面：[[source-urbangpt]], [[urbangpt]]
更新的页面：[[spatio-temporal-foundation-model]], [[traffic-forecasting]], [[gpt-st]], [[opencity]], [[index]], [[log]]

## [2026-05-31] ingest | Rectified Flow: Flow Straight and Fast (Liu et al., arXiv 2022)

Ingest Rectified Flow paper (Xingchao Liu, Chengyue Gong & Qiang Liu, UT Austin, arXiv 2022 / arXiv:2209.14577). Rectified Flow is an ODE-based generative model that learns straight trajectories from noise to data through "rectification" — an iterative procedure that straightens probability flow ODE paths by learning deterministic mappings between source and target distributions sampled from the current flow. Core contributions: (1) rectification guarantees monotonic path length reduction and convergence to straight-line trajectories; (2) with independent coupling, rectified flow converges to the optimal transport (OT) map under convex cost; (3) Reflow — an unsupervised variant using only source samples — enables 1-2 step high-quality generation (CIFAR-10 FID≈4.85 in 2 steps, 4.23 in 4 steps). This work directly inspired InstaFlow (text-to-image, ICLR 2024), Shortcut Models, and the flow matching adoption in SD3/FLUX. Created source-summary and technique pages; added reciprocal wikilinks to [[flow-matching]], [[optimal-transport]], [[diffusion-model]], [[consistency-models]], [[shortcut-models]], [[urbandit]].

创建的页面：[[source-rectified-flow]], [[rectified-flow]]
更新的页面：[[flow-matching]], [[optimal-transport]], [[urbandit]], [[shortcut-models]], [[consistency-models]], [[diffusion-model]], [[index]], [[log]]

## [2026-05-31] ingest | InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation (Liu et al., ICLR 2024)

Ingest InstaFlow paper (Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, Qiang Liu, UT Austin/UIUC/Tsinghua, ICLR 2024 / arXiv:2309.06380). InstaFlow is the first work to successfully distill large-scale Stable Diffusion into a high-quality one-step text-to-image model. Core discovery: direct distillation of SD fails catastrophically (FID=40.9 vs SD 22.8), but Text-Conditioned Rectified Flow reflow before distillation reduces the teacher-student gap by half. InstaFlow-0.9B achieves FID-5k=23.4, FID-30k=13.1 at 0.09s (first one-step diffusion model to match GAN quality), InstaFlow-1.7B with Stacked U-Net achieves FID-5k=22.4 at 0.12s. Training cost ~199 A100 GPU days (vs SD from scratch ~6250). Key innovations: text-conditioned reflow training formula, two-stage L2+LPIPS distillation, CFG adaptation for rectified flow (optimal α≈1.5 vs SD's 5-7.5), and compatibility with pre-trained ControlNet. This work bridged academic Rectified Flow theory to industrial-scale generation, directly enabling SD3 and FLUX to adopt flow matching/rectified flow paradigms. Created source-summary and technique pages; added reciprocal wikilinks to existing pages.

创建的页面：[[source-instaflow]], [[instaflow]]
更新的页面：[[index]], [[log]], [[urbandit]], [[source-urbandit]], [[flow-matching]], [[diffusion-model]], [[ddpm]], [[probability-flow-ode]], [[dpm-solver]], [[consistency-models]], [[shortcut-models]], [[classifier-free-guidance]], [[optimal-transport]]

## [2026-05-31] ingest | DiT: Scalable Diffusion Models with Transformers (Peebles & Xie, ICCV 2023)

Ingest DiT paper (Peebles & Xie, UC Berkeley, ICCV 2023 / arXiv:2212.09748). DiT replaces the U-Net backbone in diffusion models with a Vision Transformer (ViT), operating in VAE latent space (reusing Stable Diffusion's pretrained VAE). Core innovations: (1) adaLN-Zero conditioning -- adaptive layer norm with zero-initialized residual scaling that makes each block an identity function at initialization, enabling extremely stable training across all model sizes without lr warmup/dropout/weight decay; (2) Gflops-based scaling analysis across 12 variants (S/B/L/XL x p=8/4/2) showing FID is strongly anti-correlated with Gflops (r=-0.93), not parameter count -- doubling Gflops reduces FID by ~0.3-0.35x; (3) SOTA ImageNet 256x256 FID=2.27 (cfg=1.50) and 512x512 FID=3.04. DiT uses roughly 1/10 the compute of pixel-space ADM to achieve better quality. Became the foundational architecture for Sora, SD3, Flux, PixArt-alpha, and UrbanDiT. Created source-summary and technique pages; added reciprocal wikilinks to 7 existing pages (diffusion-model, ddpm, latent-diffusion-models, classifier-free-guidance, urbandit, mae, source-urbandit).

创建的页面：[[source-dit]], [[dit]]
更新的页面：[[diffusion-model]], [[ddpm]], [[latent-diffusion-models]], [[classifier-free-guidance]], [[urbandit]], [[mae]], [[index]], [[log]]

## [2026-05-31] ingest | GWNet: Graph WaveNet for Deep Spatial-Temporal Graph Modeling (Wu et al., IJCAI 2019)

Ingest GWNet paper (Wu, Pan, Long, Jiang & Zhang, UTS/Monash, IJCAI 2019, ~1,600+ citations). GWNet is a spatial-temporal graph neural network whose defining contribution is the self-adaptive adjacency matrix — learning hidden spatial dependencies end-to-end via node embeddings E₁E₂ᵀ, without requiring a predefined graph. Combined with stacked dilated causal convolutions (from WaveNet) for exponentially growing temporal receptive fields, and non-autoregressive output (all 12 future steps in one forward pass). SOTA on METR-LA and PEMS-BAY traffic datasets, surpassing DCRNN, STGCN, and GGRU. Training 5× faster than DCRNN; inference 8× faster than STGCN (2.27s). Founded the adaptive graph learning paradigm, later continued by same team's MTGNN (KDD 2020). Created source-summary and technique pages; added wikilinks to 12 existing pages where GWNet appeared in plain text.

创建的页面：[[source-gwnet]], [[gwnet]]
更新的页面：[[stgcn]], [[hybrid-periodicity-decoupling]], [[diffstg]], [[ragc]], [[pristi]], [[guided-layer-normalization]], [[specstg]], [[dcrnn]], [[node-embedding-regularization]], [[spatiotemporal-mirage]], [[source-2401-08119-specstg]], [[source-2312-00516-std-mae]], [[source-astgcn]], [[index]], [[log]]

## [2026-05-31] ingest | MAE: Masked Autoencoders Are Scalable Vision Learners (He et al., CVPR 2022)

Ingest MAE paper (He, Chen, Xie, Li, Dollár & Girshick, FAIR, CVPR 2022 / arXiv Nov 2021). MAE is a foundational self-supervised vision pretraining method based on masked autoencoding with two core designs: (1) asymmetric encoder-decoder — encoder operates only on visible patches (no mask tokens), lightweight decoder processes full token set; (2) extremely high masking ratio (75%) — eliminates spatial redundancy in images, creating a challenging pretext task requiring holistic scene understanding. ViT-H achieves 87.8% on ImageNet-1K (pure IN1K SOTA at the time), transfer learning on COCO (+4.0 APbox) and ADE20K (+3.7 mIoU) outperforms supervised pretraining. MAE broke contrastive learning's monopoly in vision self-supervision, pivoting the field back toward autoencoding paradigms. Created source-summary and technique pages; cross-linked existing pages referencing MAE/masked autoencoder.

创建的页面：[[source-mae]], [[mae]]
更新的页面：[[std-mae]], [[patchtst]], [[source-patchtst]], [[inductive-bias-shaping]], [[index]], [[log]]

## [2026-05-31] ingest | DCRNN: Diffusion Convolutional Recurrent Neural Network (Li et al., ICLR 2018)

Ingest DCRNN paper (Li, Yu, Shahabi & Liu, USC/UCLA, ICLR 2018, ~3,000+ citations). DCRNN is the first end-to-end deep learning framework combining diffusion convolution (spatial on directed graphs) + DCGRU (temporal) + Seq2Seq + Scheduled Sampling (long-term prediction). Models traffic flow as a bidirectional K-step diffusion process on directed road networks — key innovation is replacing spectral GCN's undirected constraint with a diffusion operator based on random walks, handling directed graphs naturally. 12-15% improvement over FC-LSTM, advantage grows with horizon. Created source-summary and concept pages; added cross-links to traffic-forecasting, diffstg, specstg.

创建的页面：[[source-dcrnn]], [[dcrnn]]
更新的页面：[[traffic-forecasting]], [[diffstg]], [[specstg]], [[index]], [[log]]

## [2026-05-31] ingest | STGCN: Spatio-Temporal Graph Convolutional Networks (Yu et al., IJCAI 2018)

Full ingest of STGCN paper (Yu, Yin & Zhu, Peking University, IJCAI 2018, ~2,200 citations). STGCN is the first pure convolutional spatio-temporal graph network for traffic forecasting — replaces RNN with gated 1D causal convolution (GLU) for temporal modeling, uses spectral graph convolution (Chebyshev + 1st-order) for spatial modeling, organized as "sandwich" ST-Conv Blocks (time→space→time). Achieves 14× training speedup over GCGRU (272s vs 3825s on PeMSD7-M), SOTA on BJER4/PeMSD7 datasets across 15/30/45min horizons. Created source-summary and technique pages; added cross-links from 7 existing pages that already referenced STGCN in plain text. Founding paper of the STGNN lineage: STGCN → GWNet → ASTGCN → DiffSTG → SpecSTG → UrbanDiT.

创建的页面：[[source-stgcn]], [[stgcn]]
更新的页面：[[traffic-forecasting]], [[diffstg]], [[spatio-temporal-foundation-model]], [[most]], [[mtgnn]], [[source-mtgnn]], [[hybrid-periodicity-decoupling]], [[index]], [[log]]

## [2026-05-31] ingest | CSDI + PriSTI (Integration)

Integration ingest: CSDI (Tashiro, Song, Song & Ermon, NeurIPS 2021) and PriSTI (Liu, Huang et al., ICDE 2023). CSDI is the first conditional diffusion model for probabilistic multivariate time series imputation — core innovation is replacing post-hoc conditioning with explicit conditional training via self-supervised masking + dual-axis Transformer attention. PriSTI upgrades CSDI for spatiotemporal settings by separating conditional prior extraction from noise-guided denoising — linear interpolation enhanced conditions + prior-guided attention (Q,K from clean prior, V from noisy input) + virtual node spatial downsampling. Created source-summary and technique pages; updated shared files and cross-links in 7 existing related pages.

创建的页面：[[source-csdi]], [[csdi]], [[source-pristi]], [[pristi]]
更新的页面：[[index]], [[log]], [[cofill]], [[specstg]], [[traffic-forecasting]], [[imputeformer]], [[freqflow-ts]], [[gsli]], [[generative-time-series-forecasting]]

## [2026-05-31] lint | 全量 Wiki 检查

扫描 404 个 wiki 页面，发现 6 大类共 197 个问题。严重问题 22 个，警告 54 个，信息 121 个。

严重：2 个 YAML 解析错误 + 8 个断链（缺 source 页面）+ 9 个断链（缺 concept/technique 页面）+ 9 个 `\|` 格式错误 + 7 个脚注指向不存在页面
警告：17 个子目录页面孤立 + 23 个 confidence 过高 + 25 个 source_count 不一致 + 29 个孤立脚注定义
信息：137 个过期页面（>30 天未更新）+ 2 个概括性陈述缺支撑 + 1 个缺 required 字段

## [2026-05-31] ingest | S-Mamba: Is Mamba Effective for Time Series Forecasting? (Neurocomputing 2024, arXiv:2403.11144)

Ingest Zihan Wang et al. S-Mamba paper。首个将 Mamba 选择性 SSM 引入多变量时间序列预测（MTSF）的 baseline 框架。核心设计：双向 Mamba VC Encoding 层捕获全局跨变量相关性 + FFN TD Encoding 层提取时间依赖。13 数据集 × 9 SOTA 对比：Mamba 在 VC 编码上优于 Transformer，FFN 在 TD 编码上保持统治。关键发现：变量顺序不敏感、40%→100% 泛化、增窗性能持续提升、可提升现有 Transformer。GPU 内存和训练时间均低于 Transformer 基线。

创建的页面：[[source-s-mamba]], [[s-mamba]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | LSTNet: Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (SIGIR 2018)

Ingest LSTNet paper (Lai, Chang, Yang & Liu, CMU, SIGIR 2018, ~1,728 citations)。首个跨维度多变量时间序列深度学习框架：CNN 提取跨变量局部依赖 + GRU 建模长期趋势 + Skip-RNN（周期跳跃连接 p）捕获超长周期模式 + 并行 AR 线性模型解决神经网络尺度不敏感问题。4 数据集 × horizon 3/6/12/24 → 17 项最佳 (LSTNet-skip) + 7 项 (LSTNet-Attn)。MTS 深度学习路线的奠基之作。

创建的页面：[[source-lstnet]], [[lstnet]]
更新的页面：[[index]], [[log]], [[cross-dimension-dependency]]

## [2026-05-31] ingest | TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with LLM Agents (AAAI 2025 Oral)

Ingest TimeCAP paper (Lee, Yu, Shin, Cheng & Chen, KAIST / NEC Labs, AAAI 2025 Oral)。TimeCAP 首次将 LLM 用作时间序列的上下文理解器（而非仅预测器），通过双 LLM agent（AC contextualizer + AP predictor）+ 多模态编码器（BERT + Transformer）实现输入增强和提示增强。在 7 个真实数据集上平均 F1 提升 28.75%，仅 contextualization 即贡献 21.98%。LMaaS 兼容、数据稀缺友好、提供可解释性。

创建的页面：[[source-timecap]], [[timecap]]
更新的页面：[[index]], [[log]], [[timesfm]], [[chronos]]

## [2026-05-31] ingest | GTR: Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval (ICLR 2026)

Ingest GTR paper (Cao, Dai, Han & Xiong, HKUST-GZ/HKUST/SDU, ICLR 2026)。GTR 是一个轻量级即插即用模块，通过维护可学习的全局时间嵌入 Q ∈ R^(L×N)，根据绝对时间位置检索全周期信息，以 2D 卷积和残差连接融合局部与全局依赖。在 6 个数据集上全面超越 SOTA，与简单 MLP 主干组合仅 0.98M 参数。Pearson 相关性分析揭示全局周期相关性（0.96）强于局部邻近（0.94）——这是 GTR 设计的核心洞察。理论证明在贝叶斯框架下 GTR 能严格缩小相关性估计误差。

创建的页面：[[source-gtr]], [[gtr]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | TEDM: Time Series Forecasting with Elucidated Diffusion Models (ICLR 2026)

Ingest TEDM paper (Solano-Carrillo, Naveenachandran & Niebling, DLR, ICLR 2026)。TEDM 将 EDM 的完整设计空间迁移到时间序列预测，两大创新：(1) 扩散时间轴 = 物理时间轴，采样复杂度从 O(SH) 降至 O(H)；(2) 从数据中经验估计 noise/scale schedule，避免人工预设 schedule 的归纳偏置。8 个数据集上对比 5 个扩散 + 4 个非扩散 baseline，ETTh2/ETTm2/Exchange 上 SOTA，训练仅 0.004s/batch、21 MB 内存。

创建的页面：[[source-tedm]], [[tedm]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | CoRA: Covariate-Aware Adaptation of Time Series Foundation Models (ICLR 2026)

Ingest CoRA paper (Anonymous, ICLR 2026 double-blind)。CoRA 提出面向 TSFMs 的通用协变量适配框架，创新在于：冻结基础模型作为特征提取器 + 因果嵌入（Causality Embedding）实现可解释的 Granger 因果选择 + 零初始化 adaLN 条件注入避免灾难性遗忘。兼容 TimesFM/Chronos/Moirai/FlowState/Sundial，在单模态、多模态、多元预测任务上全面超越 UniCA 和其他适配方法。

创建的页面：[[source-cora]], [[cora-tsfm]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting (Ma et al., ICLR 2026)

Ingest Ma et al. (USTC, ICLR 2026) PHAT。首篇显式建模周期异质性的多变量时序预测工作——FFT 检测每变量周期→按周期分组形成 periodic bucket→折叠为 3D 张量（变量组 × phase-aligned × intra-period offsets）→PNA（正负分解注意力 + modulation 项注入周期先验 + cross-bucket masking 防干扰）。14 数据集 × 18 baselines → 73.95% metrics SOTA。数学支撑：stick-breaking 解释 + 方差削减证明。

创建的页面：[[source-phat]], [[phat]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | LogTrans: Enhancing the Locality and Breaking the Memory Bottleneck (Li et al., NeurIPS 2019)

Ingest Li et al. (UCSB, NeurIPS 2019) LogTrans。首个将 Transformer 成功应用于时间序列预测的工作——卷积自注意力用因果卷积替代线性投影使注意力匹配基于局部形状；LogSparse 注意力每层仅关注 O(log L) 个指数间隔历史位置，堆叠层保证全信息流通，总内存 O(L(log L)²)。开创 Transformer-for-time-series 研究范式。

创建的页面：[[source-logtrans]], [[logtrans]], [[logsparse-self-attention]]
更新的页面：[[logtrans]]（source_count 1→2）, [[index]], [[log]]

## [2026-05-31] ingest | Pyraformer: Low-Complexity Pyramidal Attention (Liu et al., ICLR 2022 Oral)

Ingest Liu et al. (Ant Group / SJTU / TU Wien, ICLR 2022 Oral) Pyraformer。首个同时达成 O(L) 复杂度 + O(1) 最大信号传播路径的 Transformer——PAM（金字塔注意力模块）C-叉树 inter-scale + intra-scale 连接；CSCM（粗尺度构建模块）带 Bottleneck 的卷积降尺度。Q-K pairs 较 LogTrans 减 65.4%，较全注意力减 96.6%；序列 20000 时仅 1.91GB（Informer OOM）。ETTh1 上 MSE 较 Informer 降 24.8%–28.9%。

创建的页面：[[pyraformer]], [[source-pyraformer]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | S-Mamba: Mamba for MTS Forecasting (Wang et al., Neurocomputing 2024)

Ingest Wang et al. (Neurocomputing 2024) S-Mamba。首个 Mamba-for-MTS 基线——双向 Mamba VC (变量相关性) + FFN TD (时序依赖)；CI backbone + selective SSM 替代 Transformer attention；13 datasets × 9 baselines lead with lower GPU/time；变量顺序不变、40%→100% 泛化。

创建的页面：[[source-s-mamba]], [[s-mamba]]
更新的页面：[[mamba]], [[channel-independence]], [[index]], [[log]]

## [2026-05-31] ingest | TimeCAP: LLM-Agent Contextualize + Predict (Lee et al., AAAI 2025 Oral)

Ingest Lee et al. (KAIST / NEC Labs, AAAI 2025 Oral) TimeCAP。双 LLM agent (contextualizer AC + predictor AP) + 多模态 encoder + mutual augmentation；7 真实数据集平均 28.75% F1 提升（仅 contextualization 贡献 21.98%），黑盒 LMaaS 兼容、零样本强、可解释。

创建的页面：[[source-timecap]], [[timecap]]
更新的页面：[[timesfm]], [[chronos]], [[index]], [[log]]

## [2026-05-31] ingest | LSTNet: Long- and Short-Term Temporal Patterns (Lai et al., SIGIR 2018)

Ingest Lai et al. (CMU, SIGIR 2018) LSTNet。首个跨维度 MTS 深度学习模型——CNN (短期局部) + RNN/GRU (长期趋势) + Skip-RNN skip=p (周期季节) + AR (尺度不变线性)。~1,728 citations，开创 CD 建模路线。

创建的页面：[[source-lstnet]], [[lstnet]]
更新的页面：[[cross-dimension-dependency]], [[index]], [[log]]

## [2026-05-31] ingest | GTR: Global Temporal Retrieval (Cao et al., ICLR 2026)

Ingest Cao et al. (HKUST-GZ/HKUST/SDU, ICLR 2026) GTR。轻量级可插拔全局时序检索模块——自适应全局时序嵌入 + 绝对位置检索 + 2D 卷积局部-全局融合。核心发现：全局周期相关性 (0.96) 强于局部邻近 (0.94)。MLP backbone 仅 0.98M 参数 (19% of iTransformer)。6 数据集 SOTA，Solar 降 8.2% MSE。

创建的页面：[[source-gtr]], [[gtr]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | TEDM: Elucidated Diffusion Models for TS (Carrillo et al., ICLR 2026)

Ingest Carrillo et al. (DLR, ICLR 2026) TEDM。首次将 EDM 框架适配时序预测——扩散时间 = 物理时间 (O(SH)→O(H))，经验估计噪声/尺度 schedule，structured noise + denoiser + ODE/SDE 数值积分。轻量架构 21MB / 0.004s-batch，ETTh2/ETTm2/Exchange SOTA。对角线近似下极简推理适配实时部署。

创建的页面：[[source-tedm]], [[tedm]]
更新的页面：[[edm]], [[simdiff]], [[index]], [[log]]

## [2026-05-31] ingest | PHAT: Modeling Period Heterogeneity (Ma et al., ICLR 2026)

Ingest Ma et al. (USTC, ICLR 2026) PHAT。周期异质性感知 Transformer——FFT 检测周期→按周期分桶→3D 张量展开 (variate groups × phase-aligned time × intra-period offsets)→PNA (Positive-Negative Attention) X 形注意 + modulation gating→cross-bucket masking 防干扰→Bucket B0 处理非周期变量。14 datasets × 18 baselines → 73.95% metrics SOTA, top-2 84.38%。

创建的页面：[[source-phat]], [[phat]]
更新的页面：[[periodicity-modeling-in-time-series]], [[index]], [[log]]

## [2026-05-31] ingest | CoRA: Covariate-Aware Adaptation of TSFMs (ICLR 2026)

Ingest Qin et al. (ICLR 2026, double-blind) CoRA。TSFM 协变量感知适配框架——冻结预训练 backbone 作 embedding extractor，Causality Embedding (Granger 因果评估) + zero-initialized condition-injection (防灾难遗忘)，支持多模态协变量 (时序+语言+图像)；31.1% MSE reduction on covariate-aware benchmarks，兼容 TimesFM/Chronos-Bolt/Moirai。

创建的页面：[[source-cora]], [[cora-tsfm]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | xCPD: Routing Channel-Patch Dependencies (Li et al., ICLR 2026)

Ingest Li et al. (UTokyo + MSRA, ICLR 2026) xCPD。图频谱分解 + 动态 MoE routing 实现 patch 级通道依赖自适应建模——频谱嵌入→低/中/高频分组→频谱路由。4 backbone × 9 数据集一致提升；CI backbone 零样本迁移最大获益 15.2%。

创建的页面：[[source-xcpd]], [[xcpd]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | CPiRi: Channel Permutation-Invariant Relational Interaction (Xu et al., ICLR 2026)

Ingest Xu et al. (SHUFE/ZUFE, ICLR 2026) CPiRi。CI+CD 解耦框架——frozen CI 时序编码器 + 可训练 CD 空间模块 + 通道 shuffle 训练策略实现排列不变跨通道交互；CD 模型通道重排后误差 +400%，CPiRi 保持稳定。

创建的页面：[[source-cpiri]], [[cpiri]]
更新的页面：[[channel-independence]], [[cross-dimension-dependency]], [[index]], [[log]]

## [2026-05-31] ingest | TimeMixer: Decomposable Multiscale Mixing (Wang et al., ICLR 2024)

Ingest Wang, Wu et al. (Ant Group / Tsinghua, ICLR 2024) TimeMixer。全 MLP 架构，PDM（Past-Decomposable-Mixing）在 fine-to-coarse 和 coarse-to-fine 两方向分别混合多尺度季节与趋势分量；FMM（Future-Multipredictor-Mixing）集成多尺度预测器。长短期预测均 SOTA——Weather MSE 0.240（vs PatchTST 0.265），Solar-Energy MSE 0.216（↓24.7%）。

创建的页面：[[source-timemixer]], [[timemixer]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | MTGNN: Connecting the Dots (Wu et al., KDD 2020)

Ingest Wu, Pan, Long, Jiang, Zhang (UTS/Monash, KDD 2020) MTGNN。首个通用 GNN-based MTS 预测框架，端到端学习图结构：graph learning layer（单向稀疏邻接矩阵）、mix-hop propagation（防过平滑）、dilated inception temporal convolution（多尺度周期）、curriculum learning。SOTA on Solar/Traffic/Electricity；无空间先验下与 METR-LA/PEMS-BAY 上预定义图 STGNN 持平。

创建的页面：[[source-mtgnn]], [[mtgnn]]
更新的页面：[[mtgnn]]（source_count 1→2）, [[index]], [[log]]

## [2026-05-30] ingest | PatchTST: A Time Series is Worth 64 Words (Nie et al., ICLR 2023)

Ingest Nie, Nguyen, Sinthong & Kalagnanam (Princeton/IBM, ICLR 2023) PatchTST 论文。PatchTST 是首个将 patch tokenization 和 channel independence 同时引入时序 Transformer 的模型，证明正确设计下 Transformer 可超越简单线性模型。Patching 将时间序列分段为子序列级 token（P=16, S=8），降低注意力复杂度并保留局部语义；Channel Independence 独立处理各通道，共享权重增加训练数据量。在大数据集上取得 21% MSE 降幅（vs 最佳 Transformer 基线），训练时间最高 22× 加速。自监督 masked patch autoencoder 支持表示学习和迁移学习。

创建的页面：[[source-patchtst]], [[patchtst]]
更新的页面：[[patch-based-tokenization]], [[channel-independence]], [[instance-normalization]], [[informer]], [[lstf]], [[tslib]], [[simdiff]], [[index]], [[log]]

## [2026-05-30] lint | 全量 Wiki 修复
修复 Lint 报告 + Oracle 两轮验证中发现的所有问题：
- 修复 17 个拼写错误/不一致的引用 slug
- 修复 source_count 不一致的页面
- 降级 confidence=high 但 source_count<2 的页面为 medium/low
- 为 12 个 source_count=0 的 source-summary 页面添加自引用脚注
- 将 10 个脚注定义从纯文本格式转为 `[[wikilink]]` 格式
- 为 6 个非 source 的 source_count=0 页面添加自引用并设 confidence=low
- 移除 kellerjordan-muon-blog 断裂引用
- 修复脚注格式问题（孤立冒号）
- index.md 全面修复：去重、去幽灵、补缺失
更新的页面：约 100 个文件

## [2026-05-30] ingest | Crossformer: Transformer Utilizing Cross-Dimension Dependency for MTS Forecasting (Zhang & Yan, ICLR 2023)

Ingest Zhang & Yan (ICLR 2023) Crossformer 论文。Crossformer 是首个显式利用跨维度依赖的 MTS Transformer，提出 DSW embedding（2D 向量阵列）、TSA layer（两阶段注意力 + Router 机制）和 HED（分层编码器-解码器）。在 6 个数据集 58 个设置中 36 个 top-1、51 个 top-2。Router 机制后被 CVPE 借鉴。

创建的页面：[[source-crossformer-2023]], [[crossformer]], [[cross-dimension-dependency]], [[dsw-embedding]], [[two-stage-attention]], [[router-mechanism-for-cross-dimension]], [[hierarchical-encoder-decoder-ts]]
更新的页面：[[channel-independence]], [[patch-based-tokenization]], [[lstf]], [[cvpe]], [[index]], [[log]]

## [2026-05-30] ingest | CVPE: Enhancing Channel-Independent Time Series Forecasting via Cross-Variate Patch Embedding (Shin & Zhang, arXiv 2025)

下载 arXiv 2505.12761v3 PDF 并 ingest。CVPE 由 Donghwa Shin (Humanity Unleashed / UVA) 和 Edwin Zhang (OpenAI) 提出，是一种轻量级模块，将跨变量上下文注入通道独立 (CI) 时间序列预测模型，仅需修改 patch embedding 步骤。核心创新：(1) 可学习位置编码 $W_P \in \mathbb{R}^{P \times d_m}$ 编码 patch 在时间和变量维度上的相对位置；(2) Router-Attention 机制（借鉴 Crossformer）通过两步 MHA（聚合-分发）高效注入跨变量信息，复杂度 $O(NP)$；(3) 仅修改 patch embedding 层，保留 CI backbone 鲁棒性。集成到 Time-LLM (GPT-2 backbone) 后，Weather ↓4.6% MSE, Traffic (Modified) ↓6.7% MSE；但 ETTh2/ETTm2 ↑5.2%（过拟合弱相关特征）。

创建的页面：[[source-cvpe-2025]], [[cvpe]], [[router-attention-for-cvpe]], [[learnable-patch-position-encoding]]
更新的页面：[[channel-independence]], [[patch-based-tokenization]], [[multimodal-time-series-forecasting]], [[index]], [[log]]

## [2026-05-28] ingest | UrbanDiT: Diffusion Transformers as Open-World Spatiotemporal Foundation Models (Yuan et al., NeurIPS 2025, 完整论文)

收到完整论文 PDF（arXiv:2411.12164v2），对之前基于摘要+README 创建的 Wiki 页面进行全面重写和大幅扩展。新增核心内容：
- 数据统一化机制（3D CNN / GCN / temporal patching）
- Rectified Flow 训练（InstaFlow / 25× 加速 vs DDPM）
- 掩码策略公式 $X_t = X_t \odot (1-M) + X_0 \odot M$
- 统一提示学习：三个 memory pool（时域/频域/空域）+ task-specific mask prompt 的完整实现细节
- 消融实验（频域 prompt 影响最大）
- 三种模型规模（S/M/L）+ 扩展性分析
- 6 grid + 3 graph 数据集的完整统计
- 20+ 基线的全面对比
- 关键性能数据（11.3% 提升 on forward prediction, 30.4% 提升 on backward prediction）

创建的页面：[[unified-prompt-learning]]
更新的页面：[[source-urbandit]], [[urbandit]], [[spatio-temporal-foundation-model]], [[index]], [[log]]
源文件：raw/urbandit.pdf

## [2026-05-13] ingest | Generative Modeling with Flux Matching (Pao-Huang et al., Stanford, arXiv 2026)

下载 arXiv 2605.07319 PDF 并 ingest。Flux Matching 由 Peter Pao-Huang、Xiaojie Qiu 和 Stefano Ermon（Stanford University）提出，是一种全新生成建模范式，将 score-based 模型推广到任意"生成向量场"（不限于保守的得分函数）。核心创新：(1) 基于 Fokker-Planck 平稳条件的 Flux Matching 损失，不要求逐点匹配得分，仅匹配概率通量的散度；(2) 投影 Fisher 散度 (Projected Fisher Divergence)，在保持 $L^2(p_{\text{data}})$ 几何的同时兼容非保守向量场；(3) 噪声退火扩展使其可接入现有扩散框架。应用：加速采样（优化混合速度）、可解释生成（RNA 速度的机制性 ODE）、因果掩码嵌入有向依赖、独立图像生成（CIFAR-10 / CelebA）。

创建的页面：[[source-2605-07319]], [[flux-matching]], [[projected-fisher-divergence]], [[generative-vector-field]]
更新的页面：[[index]], [[log]], [[score-matching]], [[fokker-planck-equation]], [[score-based-generative-modeling]]

## [2026-05-12] ingest | UrbanDiT: Diffusion Transformers as Open-World Spatiotemporal Foundation Models (NeurIPS 2025)

Ingested UrbanDiT paper from Zotero storage (Yuan et al., Tsinghua FIB Lab, NeurIPS 2025). UrbanDiT is a spatiotemporal foundation model based on Diffusion Transformer (DiT) architecture with prompt learning, unifying diverse urban spatiotemporal data types. Supports four tasks: bi-directional prediction, temporal interpolation, spatial extrapolation, and spatio-temporal imputation. Key advantage: zero-shot generalization surpassing many trained baselines. Applicable to traffic flow, crowd movement, taxi demand, bike-sharing, cellular network traffic, etc.

创建的页面：[[source-urbandit]], [[urbandit]]
更新的页面：[[index]], [[log]]

## [2026-05-12] ingest | Flow-GRPO: Training Flow Matching Models via Online RL (NeurIPS 2025)

下载 arXiv 2505.05470 PDF 并 ingest。Flow-GRPO 首次将在线策略梯度 RL（GRPO）引入流匹配模型。核心创新：(1) ODE-to-SDE 转换将确定性 ODE（dxt = vt dt）转换为等效 SDE，通过 Fokker-Planck 方程推导出漂移系数 fSDE = vt + (σt²/2)∇log pt，最终得到 Euler-Maruyama 离散化更新规则，在保持边缘分布的前提下引入随机性；(2) Denoising Reduction 训练时用 10 步去噪加速训练 4 倍以上，推理时保持 40 步保证质量。在 GenEval 上将 SD3.5-M 准确率从 63% 提升至 95%，超越 GPT-4o，且 reward hacking 极低。

创建的页面：[[source-flow-grpo]], [[ode-to-sde-conversion]], [[denoising-reduction]], [[flow-grpo]]
更新的页面：[[index]], [[log]]

## [2026-05-11] ingest | GSLI: Graph Structure Learning for Spatial-Temporal Imputation (AAAI 2025)

Ingested Yang et al. (Nankai University & HIT Shenzhen, AAAI 2025) paper proposing multi-scale graph structure learning for spatial-temporal imputation. GSLI addresses feature heterogeneity and cross-feature spatial dependencies through three innovations: (1) Node-scale graph structure learning — independent meta-graphs per feature with prominence modeling, solving the problem that standard graph convolution cannot handle different spatial correlations for different features (Propositions 1 & 2); (2) Feature-scale graph structure learning — meta-feature graph capturing common spatial correlations across features within all stations; (3) Cross-feature and cross-temporal representation learning via Transformer self-attention. Evaluated on 6 real-world incomplete datasets with consistent superiority across MCAR/MAR/MNAR mechanisms.

创建的页面：[[source-yang-gsli-2025]], [[gsli]], [[node-scale-graph-structure-learning]], [[feature-scale-graph-structure-learning]], [[prominence-modeling-gsl]]
更新的页面：[[imputeformer]], [[cofill]], [[traffic-forecasting]], [[index]], [[log]]

## [2026-05-11] maintenance | ImputeFormer 精读增强

基于微信公众号论文精读文章补充了 ImputeFormer 页面内容：
- 增加设计动机中的频谱可视化证据（图 1：低秩模型 vs 深度模型的奇异值分布差异）
- 增加相关工作的细化分类（低秩类：TRMF、TiDER；深度类：GRU-D、BRITS、GAIN、PriSTI、CSDI 等）
- 增加详细的实验结果数据（各数据集 MAE 对比、不同观测率鲁棒性、消融实验具体数值）
- 增加可解释性发现（频谱分析、t-SNE 空间嵌入、inflow/outflow 可视化、填补结果对比）
- 增加未来工作方向（多任务学习、大规模预训练、表示学习）
- 在 source-summary 中补充核心动机（频谱视角）和更完整的消融/鲁棒性数据

更新的页面：[[imputeformer]], [[source-2312-01728]]

## [2026-05-11] ingest | ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation (KDD 2024)

Downloaded arXiv 2312.01728 PDF and ingested ImputeFormer paper by Nie et al. (Tongji University & Hong Kong Polytechnic University, KDD 2024). ImputeFormer 是一种低秩性引导的 Transformer 时空填补模型，核心创新包括：(1) 时间投影注意力（Projected Attention）通过可学习投影器实现显式低秩分解，复杂度 O(TC)；(2) 空间嵌入注意力（Embedded Attention）利用节点嵌入作为低维代理计算空间相关性，复杂度 O(N·D_emb)；(3) 傅里叶填补损失（Fourier Imputation Loss）基于 DFT 核范数等价性，对填补频谱进行 ℓ1 稀疏正则化。在 10 个基准数据集的点缺失和块缺失场景下均取得 SOTA，训练速度比 SPIN 快 15×。

创建的页面：[[source-2312-01728]], [[imputeformer]], [[projected-attention]], [[embedded-attention]], [[fourier-imputation-loss]]
更新的页面：[[cofill]], [[index]], [[log]]

## [2026-05-11] ingest | CoFILL: Spatiotemporal Data Imputation by Conditional Diffusion (arXiv 2025)

Ingested CoFILL paper by He et al. (Hebei University of Technology, Tiangong University, University of Southern Queensland). CoFILL 是一种用于时空数据填补的新型条件扩散框架，核心创新包括：(1) 非递归扩散结构解决误差累积问题；(2) 双流架构同时处理时域（TCN+GCN）和频域（DCT）特征，通过 Cross-Attention 融合；(3) 双策略预处理（Forward Interpolation + Gaussian Noise）。在 AQI-36、METR-LA、PEMS-BAY 三个数据集上，在 MAE/MSE/CRPS 指标上 12/15 配置达到最优，相比 PriSTI 在 METR-LA Block 场景提升 10.22%。

创建的页面：[[source-cofill-spatiotemporal-imputation]], [[cofill]], [[dual-stream-temporal-frequency-processing]]
更新的页面：[[generative-time-series-forecasting]], [[spatio-temporal-foundation-model]], [[traffic-forecasting]], [[index]], [[log]]

## [2026-05-09] ingest | A Fourier Space Perspective on Diffusion Models / EqualSNR (Microsoft Research, 2025)

Ingested arXiv:2505.11278 by Falck et al. (Microsoft Research). 论文从傅里叶空间重写 DDPM 前向过程，给出每频率 SNR 公式 $s_t^{\mathrm{DDPM}}(i)=\bar\alpha_t C_i/(1-\bar\alpha_t)$，说明自然图像等数据的傅里叶功率律会让高频分量更早、更快降 SNR。核心贡献包括：(1) 理论与 KDE 可视化证明高频快速加噪会使反向后验 $q(y_{t-1}\mid y_t)$ 更易偏离单一高斯假设；(2) 提出 EqualSNR，令 $\Sigma_{ii}=cC_i$ 使所有频率等 SNR 加噪，并给出 $C^{-1/2}$ 加权傅里叶损失及其 ELBO 解释；(3) 在 CIFAR-10/CelebA/LSUN Church 上 FID 与 DDPM 大体持平，在高频谱统计与 Dots 高频任务上显著优于 DDPM；(4) FlippedSNR 多次训练失败，提示低频到高频层级可能具有优化价值但非绝对必要。

创建的页面：[[source-equal-snr]], [[equal-snr]], [[frequency-hierarchy-in-diffusion]]
更新的页面：[[diffusion-model]], [[frequency-based-noise-control]], [[ddpm]], [[inductive-bias-shaping]], [[index]], [[log]]

## [2026-05-09] ingest | An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models (NeurIPS 2025, Harvard)

Ingested Wang & Pehlevan (Kempner Institute, Harvard, NeurIPS 2025) 论文，首次对"扩散模型为什么先学低频"给出严格理论解答。核心贡献：利用高斯等价原理求解线性 denoiser 的梯度流闭式解，积分 PF-ODE 得到生成分布的解析表达式。发现反比方差谱定律 $\tau_k^* \propto \lambda_k^{-1}$——高方差模式（粗结构）比低方差模式（细纹理）收敛快一个数量级。扩展到深度线性网络和卷积网络，证明权重共享加速但不消除偏置，而局部卷积带来质的改变。MLP-UNet 实验确认谱定律存在；CNN-UNet 中谱偏置几乎消失，说明卷积架构重塑了学习动力学。

创建的页面：[[source-spectral-bias-learning-dynamics]], [[spectral-bias-training-dynamics]]
更新的页面：[[frequency-based-noise-control]], [[inductive-bias-shaping]], [[frequency-diffusion]], [[index]], [[log]]

## [2026-05-09] ingest | SAGD: Spectrally Anisotropic Gaussian Diffusion (arXiv 2510.09660)

Ingested SAGD 完整版论文（Scimeca, Jiralerspong, Earnshaw, Hartford, Bengio, 2025），将 workshop 版（2502.10236）的频域噪声控制形式化为各向异性高斯协方差 $\Sigma_w$ 框架。核心理论贡献：(1) 推导各向异性 score-$\epsilon$ 关系 $\nabla_{x_t} \log q_{w,t} = -\frac{1}{\sigma_t}\Sigma_w^{-1}\epsilon_\theta$；(2) 证明 $\Sigma_w \succ 0$ 时 $t \to 0$ score 收敛到真实数据 score；(3) rank-deficient $\Sigma_w$ 下 projected score 的选择性忽略理论。提出 plw-SAGD（幂律加权）和 bpm-SAGD（带通掩码）两种算子。ImageNet-1k 256×256 DiT 实验 FID 8.68→7.55（↓13%）。Workshop 版标记为 superseded。

创建的页面：[[source-sagd]]
更新的页面：[[source-2502-10236]]（superseded），[[frequency-diffusion]]，[[frequency-based-noise-control]]，[[inductive-bias-shaping]]，[[two-band-mixture-noise]]，[[index]]，[[log]]

## [2026-05-09] ingest | Elucidating the SNR-t Bias of Diffusion Probabilistic Models (CVPR 2026)

Ingested Yu et al. (AMAP Alibaba Group & Lanzhou University, CVPR 2026). 论文识别并理论证明了扩散模型中的 SNR-t Bias——推理阶段预测样本 SNR 与时间步之间的错配。核心贡献：(1) 通过滑动窗口实验发现 Key Finding 1（低 SNR → 高估噪声预测）和 Key Finding 2（逆过程 SNR 系统性低于前向过程）；(2) 提出更准确的重建样本假设 $x_\theta^0 = \gamma_t x_0 + \phi_t \epsilon_t$（修复了此前 $x_\theta^0 = x_0 + \phi_t \epsilon_t$ 与方差恒等式的矛盾），严格证明逆过程 SNR 始终低于前向过程 (Theorem 5.1)；(3) 提出 DCW 方法，在小波域对各频率子带做差分校正，用 $\sigma_t$ 动态调度低/高频校正系数。实验覆盖 9 种模型 (IDDPM/ADM/DDIM/A-DPM/EDM/PFGM++/FLUX/Qwen-Image/DiT)、4 个数据集、多种采样步数，FID 降幅最高 42.9%，计算开销仅 0.08%~0.47%。详细数学推导记录在 source-summary 页面 (>100 行)。

创建的页面：[[source-snr-t-bias]], [[snr-t-bias]], [[dcw]]
更新的页面：[[diffusion-model]], [[tweedies-formula]], [[index]], [[log]]

## [2026-05-09] ingest | FreqFlow: Frequency-Aware Flow Matching (arXiv 2026)

Ingested FreqFlow paper (arXiv:2604.15521) by Ren et al. (JHU & ByteDance, 2026). FreqFlow proposes a frequency-aware flow matching framework with a two-branch architecture: a frequency branch that separately processes low- and high-frequency components via DFT/Gaussian filtering, and a spatial branch (ConvNeXt) guided by frequency features. Key innovations: (1) adaptive time-dependent frequency integration $\omega_t = \sigma(\text{MLP}(h_t^L, h_t^H, t))$; (2) dual-domain supervision combining spatial L2 loss and frequency FFT loss; (3) unified frequency branch (ViT) outperforming separate networks. Achieves SOTA FID 1.38 on ImageNet-256, surpassing DiT (+0.79) and SiT (+0.58). Detailed mathematical derivations recorded in technique page (>100 lines).

创建的页面：[[source-freqflow]], [[freqflow]], [[frequency-aware-conditioning]]
更新的页面：[[flow-matching]], [[diffusion-model]], [[frequency-based-noise-control]], [[frequency-diffusion]], [[index]], [[log]]

## [2026-05-09] ingest | Shaping Inductive Bias in Diffusion Models through Frequency-Based Noise Control (ICLR 2025 Workshop)

Ingested Jiralerspong, Earnshaw, Hartford, Bengio & Scimeca (Mila/Valence Labs, 2025) 论文，提出频域扩散方法——通过在前向加噪过程中对噪声的频谱进行目的性操控来显式塑造扩散模型的归纳偏置。核心假设：前向加噪中被抹除的信息恰好是去噪模型有压力学习的信息。提出三种频域加权方式（幂律、指数衰减、带通混合），实验验证5个数据集中3个显著受益，并展示选择性忽略被噪声破坏频段的能力。

创建的页面：[[source-2502-10236]], [[frequency-diffusion]], [[frequency-based-noise-control]], [[inductive-bias-shaping]], [[two-band-mixture-noise]]
更新的页面：[[diffusion-model]], [[edm-design-space]], [[index]], [[log]]

## [2026-05-08] ingest | Demystify Mamba in Vision: A Linear Attention Perspective (NeurIPS 2024)

Ingested Han et al. (Tsinghua & Alibaba, NeurIPS 2024) paper that unifies Mamba and linear attention within a single framework, identifying 6 key differences. Created MILA architecture page and supporting technique/concept pages. Updated 4 existing pages with cross-references.

创建的页面：[[source-demystify-mamba-linear-attention-2024]], [[mamba]], [[mila]], [[linear-attention-unified-framework]], [[forget-gate-in-sequential-models]], [[mamba-block-design]]
更新的页面：[[linear-attention-bias]], [[generalized-positional-encoding-framework]], [[traffic-forecasting]], [[glu-gated-linear-unit]], [[index]], [[log]]

## [2026-05-08] ingest | SpecSTG: A Fast Spectral Diffusion Framework for Probabilistic Spatio-Temporal Traffic Forecasting

Ingest arXiv:2401.08119v3 (Lin, Shi, Han & Gao, 2024)。SpecSTG 是首个在图谱域执行扩散过程的概率时空图预测框架，通过生成图傅里叶表示而非原始序列来利用空间依赖关系，实现 8% RMSE 提升和 3.33× 训练加速。

创建的页面：[[source-2401-08119-specstg]], [[specstg]], [[fast-spectral-graph-convolution]], [[spectral-recurrent-encoder]]
更新的页面：[[traffic-forecasting]], [[generative-time-series-forecasting]], [[index]], [[log]]

## [2026-05-04] ingest | FEDformer: Frequency Enhanced Decomposed Transformer (ICML 2022) — 深度增强
详细 ingest FEDformer 论文完整 PDF，深度增强已有 source-summary 和 entity 页面。创建 3 个核心技术页面，更新 9 个交叉引用页面，并添加反向链接。

核心增强内容：
- **source-summary**：补充 Theorem 1（随机 Fourier 采样理论保证）、RIP 矩阵低秩近似理论、完整架构公式（Encoder/Decoder 的 Equations 1-7、MOEDecomp 公式）、消融研究详细结果（V1/V2/V3 的 10/12/16 改进数）、KS 分布检验完整分析、MOEDecomp vs 单一分解 (+2.96%)、复杂度表格对比、与 Autoformer 的 5 个关键差异
- **entity 页面**：重写为包含完整架构流程、复杂度对比表、6 个数据集的性能汇总、Connection 链完善（11 个关联页面）
- [[frequency-enhanced-block]] — FEB-f/FEB-w 的完整数学公式、递归分解流程、与标准 self-attention 的对比表
- [[frequency-enhanced-attention]] — FEA-f/FEA-w 的交叉注意力设计、消融证据（16/16 改进）、与标准 cross-attention 的对比
- [[moe-decomposition]] — MOEDecomp 的输入自适应加权机制、编码器/解码器中的三层部署、效果对比（+2.96%）、其他分解方法的全面对比

创建的页面：[[frequency-enhanced-block]], [[frequency-enhanced-attention]], [[moe-decomposition]]
更新的页面：[[source-fedformer]], [[fedformer]], [[autoformer]], [[dualsformer]], [[informer]], [[hyperd]], [[frequency-aware-residual-representation]], [[tslib]], [[traffic-forecasting]], [[periodicity-modeling-in-time-series]], [[index]], [[log]]

执行完整 lint 检查并修复以下问题：
- 修正 33 个页面 source_count 与页面内实际引用数量不一致
- 19 个页面 confidence: high → medium（source_count=0 或 1，不满足 high 标准）
- 修复 2 个 typo 断链：probparse → probsparse（informer.md, source-zhou-informer-2021.md）
- 修正 log.md 中 spurious-patterns-in-attention → spurious-patterns（页面不存在）
- 创建 7 个 stub 页面修复 broken wikilinks：energy-based-model, glu-gated-linear-unit, heterogeneous-moe-routing, staeformer, tweedies-formula, score-based-generative-models
- 更新 index.md 添加新 stub 页面条目

Pages created: [[energy-based-model]], [[glu-gated-linear-unit]], [[heterogeneous-moe-routing]], [[staeformer]], [[tweedies-formula]], [[score-based-generative-models]]
Pages updated: [[informer]], [[source-zhou-informer-2021]], [[log]], [[index]] + 33 source_count fixes + 19 confidence fixes

## [2026-05-04] ingest | TimesNet: Temporal 2D-Variation Modeling (ICLR 2023) — 补完

从 Zotero 存储解析 TimesNet 完整 PDF 并增强已有 source-summary。仅保留 arXiv 链接，不存储 PDF。
主要修改：
- **修复**：补充 `source-timesnet.md` 中缺失的脚注定义
- **扩充**：添加 FFT 周期发现公式、TimesBlock 六步架构流程、五个任务详细基准与定量结果、效率分析、加深的批判分析
- 更新 `timesnet.md` 实体页面：Connections 添加内联引用，移除未使用脚注

Pages updated: [[source-timesnet]], [[timesnet]]

## [2026-05-04] ingest | Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI 2021 Best Paper)

Ingested Informer paper from Zotero storage. Informer is the seminal work that pioneered efficient Transformer architectures for LSTF, addressing all three vanilla Transformer bottlenecks simultaneously: $O(L^2)$ computation → $O(L \log L)$ via ProbSparse attention, $O(J \cdot L^2)$ memory → $O((2-\epsilon) L \log L)$ via self-attention distilling, and slow autoregressive decoding → one-forward-pass generative decoder. AAAI 2021 Best Paper. Evaluated on ETT, ECL, and Weather datasets, significantly outperforming ARIMA, Prophet, LSTMa, LSTnet, DeepAR, LogTrans, and Reformer. Updated 10 existing pages (autoformer, fedformer, timesnet, source-autoformer, source-fedformer, source-timesnet, source-frets, source-deep-time-series-survey, source-language-in-the-flow-of-time, periodicity-modeling-in-time-series) with cross-references and Informer citations.

Pages created: [[source-zhou-informer-2021]], [[informer]], [[probsparse-self-attention]], [[generative-style-decoder]], [[lstf]]
Pages updated: [[autoformer]], [[fedformer]], [[timesnet]], [[source-autoformer]], [[source-fedformer]], [[source-timesnet]], [[source-frets]], [[source-deep-time-series-survey]], [[source-language-in-the-flow-of-time]], [[periodicity-modeling-in-time-series]], [[index]], [[log]]

## [2026-05-04] ingest | 数学直觉系列（二）：VAE与重参数化
Ingested bluuuuue 小红书技术教程文章（第二期），将重参数化技巧定位为让随机性与梯度共存的结构性方案。核心论点：采样不可导→REINFORCE高方差→两步分离重参数化→双重功效（打通反传+降方差）→适用前提（连续+位置-尺度族）→三大应用（扩散/VLA/SAC）。
创建的页面：[[source-bluuuuue-reparameterization-trick]], [[reparameterization-trick]]
更新的页面：[[variational-autoencoder]], [[elbo]], [[diffusion-model]], [[ddpm-simplified-training-objective]], [[index]], [[log]]

## [2026-05-04] ingest | 数学直觉系列（一）：缩放因子 1/√dₖ —— 注意力机制的数值稳定性条件
Ingested bluuuuue 小红书技术教程文章，将 Scaled Dot-Product Attention 的缩放因子 $1/\sqrt{d_k}$ 重新定位为数值稳定性条件。核心论点：点积方差膨胀（$Var(Z)=d_k$）导致 Softmax 饱和与梯度消失；$1/\sqrt{d_k}$ 将方差归一化至 1；选择 $\sqrt{d_k}$ 而非 $d_k$ 避免过缩放；缩放保持 argmax 不变。更新了 4 个现有注意力稳定性相关页面添加交叉引用。
创建的页面：[[source-bluuuuue-scaling-factor-intuition]], [[scaling-factor-sqrt-dk]]
更新的页面：[[attention-entropy-collapse]], [[attention-logit-explosion]], [[attention-temperature-scaling]], [[key-normalization]], [[index]], [[log]]

## [2026-05-03] query | 多模态数据的语义理解
基于 7 个源文件（MindTS, VoT, TaTS/CTR, Aurora, MoST, UniCA, SimDiff）综合分析多模态语义理解的对齐范式（对比对齐/频域融合/自然共振）、融合策略（注意力引导/交叉视图/同质化/SNR门控/频域加权）和冗余过滤（信息瓶颈压缩/SNR模态选择）。提炼三层统一框架（理论基础→对齐机制→模型实例）和四个开放问题。归档为 analysis 页面。

Pages created: [[multimodal-semantic-understanding]]
Pages updated: [[index]], [[log]]

## [2026-05-03] lint | 近 6 篇论文（42 页面）lint 检查与修复
执行 lint 检查，修复以下问题：
- 修正 aurora.md source_count=5→4（实际仅引用 4 个源文件）
- 重命名 MindTS 系列页面 4 个文件中的脚注 slug：`src-multimodal-ts-ad` → `src-multimodal-ts-anomaly-detection`（不符合源文件命名规则）
- 创建 6 个 stub 页面修复 broken wikilinks：[[signal-to-noise-ratio-modality-selection]], [[opencity]], [[mutual-information]], [[cross-view-text-fusion]], [[contrastive-learning]], [[information-bottleneck-principle]]
- 更新 [[index]] 添加新 stub 页面条目

Pages created: [[signal-to-noise-ratio-modality-selection]], [[opencity]], [[mutual-information]], [[cross-view-text-fusion]], [[contrastive-learning]], [[information-bottleneck-principle]]
Pages updated: [[aurora]], [[mindts]], [[multimodal-time-series-anomaly-detection]], [[fine-grained-time-text-semantic-alignment]], [[content-condenser-reconstruction]], [[index]], [[log]]

## [2026-05-03] ingest | UniExtreme: A Universal Foundation Model for Extreme Weather Forecasting (arXiv 2025)
Ingested UniExtreme paper (arXiv:2508.01426v2) by Ni, Zhang & Liu (HKUST Guangzhou). UniExtreme is the first extreme weather foundation model trained on both labeled data from 18 types of real-world extreme weather events and general meteorological data. Key innovations: (1) Adaptive Frequency Modulation (AFM) — learnable Beta-distribution spectral filters + multi-granularity spatiotemporal band aggregation that captures the "right-shift" spectral disparity between normal and extreme weather regions; (2) Event Prior Augmentation (EPA) — categorized extreme event memory pool + dual-level (intra-type + inter-type) attention fusion that resolves hierarchical extreme diversity and compound event schema. Empirical analysis on ~36.4M normal and ~882K extreme US weather regions confirms spectral right-shift (Wasserstein distance 3.1e-3 vs 2.4e-4) and 86% composite extreme co-occurrence rate. UniExtreme achieves ~11% MAE and ~10% RMSE improvement over best baseline in extreme weather forecasting, and reduces normal-extreme gap by ~37% for MSL. Built HR-Extreme-V2 (26TB, 2019-2024, 18 event types).
Pages created: [[source-uniextreme]], [[uniextreme]], [[extreme-weather-forecasting]], [[adaptive-frequency-modulation]], [[event-prior-augmentation]]
Pages updated: [[aurora]], [[simdiff]], [[timesfm]], [[index]], [[log]]

## [2026-05-03] ingest | Language in the Flow of Time: TaTS (ICLR 2026)
Ingested TaTS paper (arXiv:2502.08942) by Li et al. (UIUC/Meta/IBM Research). TaTS is a plug-and-play multimodal time series framework that treats time-series-paired texts as auxiliary variables, enabling any existing TS model to handle multimodal data without architecture modification. Key innovations: (1) Chronological Textual Resonance (CTR) — the discovery that time-series-paired texts exhibit periodic properties mirroring the original time series, motivated by the Platonic Representation Hypothesis; (2) TT-Wasserstein — a metric to quantify CTR level and alignment quality; (3) Texts as Time Series (TaTS) — a simple framework that encodes texts via LLM, reduces dimensionality via MLP, and concatenates as auxiliary variables. Evaluated on 18 datasets across 9 TS models, achieving >5% average improvement on 6/9 datasets and >30% on the largest dataset. Compared with VoT (LLM reasoning), MindTS (anomaly detection), UniCA (covariate adaptation), Aurora (generative foundation model), and Chronos (tokenization).
Pages created: [[source-language-in-the-flow-of-time]], [[tats]], [[chronological-textual-resonance]], [[tt-wasserstein]], [[texts-as-auxiliary-variables]]
Pages updated: [[multimodal-time-series-forecasting]], [[vot]], [[chronos]], [[endogenous-text-alignment]], [[fine-grained-time-text-semantic-alignment]], [[aurora]], [[index]], [[log]]

## [2026-05-03] ingest | Aurora: Towards Universal Generative Multimodal Time Series Forecasting (arXiv 2026)
Ingested Aurora paper (arXiv:2509.22295) by Wu, Jin, Qiu, Chen, Shu, Yang & Guo. Aurora is the first Multimodal Time Series Foundation Model supporting multimodal inputs (text, image, numerical) and zero-shot inference. Key innovations: (1) Modality-Guided Multi-head Self-Attention — extracts domain knowledge from text/image modalities via tokenization-encoding-distillation and injects it into temporal representation modeling; (2) Prototype-Guided Flow Matching — uses multimodal representations to generate conditions and prototypes for future tokens, enabling generative probabilistic forecasting. Evaluated on 5 benchmarks (TimeMMD, TSFM-Bench, ProbTS, TFB, EPF), achieving SOTA on both unimodal and multimodal scenarios. Aurora fills the gap between single-modal TSFMs (TimesFM, Chronos) and end-to-end multimodal supervised models by supporting both multimodal inputs and zero-shot inference.
Pages created: [[source-aurora]], [[aurora]], [[modality-guided-self-attention]], [[prototype-guided-flow-matching]], [[generative-time-series-forecasting]]
Pages updated: [[multimodal-time-series-forecasting]], [[simdiff]], [[most]], [[timesfm]], [[chronos]], [[flow-matching]], [[vot]], [[mindts]], [[index]], [[log]]

## [2026-05-03] ingest | VoT: Event-Driven Reasoning and Multi-Level Alignment for Time Series Forecasting (ICLR 2026)
Ingested VoT paper (arXiv:2603.15452) from East China Normal University. VoT is a multimodal time series forecasting method that unlocks the value of text through two complementary mechanisms: (1) Event-driven Reasoning with Historical In-Context Learning (HIC) — a three-step generative pipeline that uses LLMs to reason over exogenous text (news, policy documents) and retrieves corrected historical examples as error-informed guidance; (2) Multi-level Alignment — Endogenous Text Alignment (ETA) at the representation level (decomposed trend/seasonal contrastive learning) and Adaptive Frequency Fusion (AFF) at the prediction level (learnable per-band frequency fusion). Evaluated on 10 real-world datasets, achieving 20/20 first-place counts against baselines. Same lab (ECNU) as MindTS.
Pages created: [[source-event-driven-ts-forecasting]], [[vot]], [[event-driven-reasoning]], [[historical-in-context-learning]], [[multi-level-alignment]], [[endogenous-text-alignment]], [[adaptive-frequency-fusion]]
Pages updated: [[multimodal-time-series-forecasting]], [[fine-grained-time-text-semantic-alignment]], [[mindts]], [[content-condenser-reconstruction]], [[index]], [[log]]

## [2026-05-03] ingest | Multimodal Time Series Anomaly Detection with Semantic Alignment and Condensed Interaction (ICLR 2026)
Ingested MindTS paper (arXiv:2603.21612) from East China Normal University. MindTS is the first dedicated multimodal anomaly detection model that jointly leverages time series and text modalities. Key innovations: (1) fine-grained time-text semantic alignment via cross-view fusion of endogenous and exogenous text, and (2) content condenser reconstruction using Information Bottleneck principle to filter redundant text and enhance cross-modal interaction. Evaluated on 6 real-world multimodal datasets, outperforming 17 baselines.
Pages created: [[source-multimodal-ts-anomaly-detection]], [[mindts]], [[multimodal-time-series-anomaly-detection]], [[fine-grained-time-text-semantic-alignment]], [[content-condenser-reconstruction]]
Pages updated: [[multimodal-time-series-forecasting]], [[channelmts]], [[most]], [[multi-modality-refinement]], [[index]], [[log]]

## [2026-04-26] maintenance | Wiki reset
Cleared sample/demo content. Vault is empty and ready for first ingest.

## [2026-04-27] ingest | Mathematical Foundations of Reinforcement Learning (Readme + Grid World Code)
Ingested two source files from `raw/math-foundation-rl/`. Created full wiki scaffolding for the RL textbook and its grid-world environment code.
Pages created: [[source-math-foundation-rl-readme]], [[source-grid-world-code-readme]], [[math-foundation-of-reinforcement-learning]], [[shiyu-zhao]], [[grid-world-environment]], [[bellman-equation]], [[temporal-difference-learning]], [[rl-learning-path-mfrl]]
Pages updated: [[index]], [[log]]

## [2026-04-27] ingest | HyperD: Hybrid Periodicity Decoupling Framework for Traffic Forecasting
First ingest. Downloaded arXiv 2511.09275 PDF and created full wiki scaffolding.
Pages created: [[source-hyperd-hybrid-periodicity-decoupling]], [[hyperd]], [[hybrid-periodicity-decoupling]], [[traffic-forecasting]], [[frequency-aware-residual-representation]], [[spatial-temporal-attentive-encoder]], [[dual-view-alignment-loss]], [[demlp-decoder]]
Pages updated: [[index]], [[log]]

## [2026-04-30] ingest | ChannelMTS: Multi-modal Time-Series Framework for High-Speed Railway Channel Prediction
Ingested KDD 2026 paper from Zotero storage. ChannelMTS solves HSR channel prediction by integrating environmental information (position, K-factor, RMS delay) with channel states through retrieval-augmented statistical channel (RAGC), modality alignment (median+IQR normalization), and adaptive fusion.
Pages created: [[source-channelmts]], [[channelmts]], [[retrieval-augmented-statistical-channel]]
Pages updated: [[multimodal-time-series-forecasting]], [[index]], [[log]]

## [2026-04-29] ingest | UniCA: Unified Covariate Adaptation for Time Series Foundation Model
Ingested ICLR 2026 paper from Zotero storage. UniCA solves the problem of adapting Time Series Foundation Models (TSFMs) to handle heterogeneous covariates (categorical, image, text).
Pages created: [[source-unca]], [[unified-covariate-adaptation]], [[covariate-homogenization]], [[heterogeneous-covariates]], [[conditional-attention-pooling]], [[multimodal-time-series-forecasting]], [[timesfm]], [[source-timesfm]], [[chronos]], [[source-chronos]]
Pages updated: [[index]], [[log]]
为便于理解，扩充了 11 个极短的强化学习概念/算法页面。
扩充的页面：[[mdp-formal-definition]], [[exploration-vs-exploitation]], [[value-iteration]], [[policy-iteration]], [[policy-evaluation]], [[truncated-policy-iteration]], [[q-learning-algorithm]], [[sarsa-algorithm]], [[expected-sarsa]], [[epsilon-greedy]], [[contraction-mapping-theorem]]
主要增加内容：算法细节、收敛性分析、对比表格、变体扩展等。

## [2026-04-27] ingest | Mathematical Foundations of RL Chapters 1/2/7 (Deep Ingest)
第二轮深度 ingest：围绕第 1/2/7 章补充 MDP、贝尔曼方程与 TD 算法主线，新增章节级 source-summary 与算法专题页。
Pages created: [[source-chapter-1-basic-concepts]], [[source-chapter-2-state-values-and-bellman-equation]], [[source-chapter-7-temporal-difference-methods]], [[mdp-formal-definition]], [[policy-evaluation]], [[action-value-function]], [[sarsa-algorithm]], [[expected-sarsa]], [[n-step-sarsa]], [[q-learning-algorithm]], [[on-policy-vs-off-policy]]
Pages updated: [[bellman-equation]], [[temporal-difference-learning]], [[grid-world-environment]], [[rl-learning-path-mfrl]], [[math-foundation-of-reinforcement-learning]], [[index]], [[log]]

## [2026-04-28] ingest | Understanding Diffusion Models: A Unified Perspective (Luo, 2022)
核心论文 ingest：扩散模型统一视角教程论文，提供 VAE → HVAE → VDM 的完整数学推导。
Pages created: [[source-understanding-diffusion-models]], [[diffusion-model]], [[elbo]], [[variational-autoencoder]], [[score-function]], [[classifier-guidance]], [[classifier-free-guidance]]
Pages updated: [[index]], [[log]]

## [2026-04-27] ingest | Mathematical Foundations of RL Chapters 3/4/5 (Deep Ingest)
第三轮深度 ingest：补齐最优性方程、动态规划算法与蒙特卡洛主线。
Pages created: [[source-chapter-3-optimal-state-values-and-bellman-optimality-equation]], [[source-chapter-4-value-iteration-and-policy-iteration]], [[source-chapter-5-monte-carlo-methods]], [[bellman-optimality-equation]], [[value-iteration]], [[policy-iteration]], [[truncated-policy-iteration]], [[monte-carlo-methods-rl]], [[epsilon-greedy]], [[exploration-vs-exploitation]], [[contraction-mapping-theorem]]
Pages updated: [[bellman-equation]], [[policy-evaluation]], [[temporal-difference-learning]], [[rl-learning-path-mfrl]], [[math-foundation-of-reinforcement-learning]], [[index]], [[log]]

## [2026-04-29] lint | 全量 lint 检查与修复
执行完整 lint 检查，修复以下问题：
- 修复 9 个 wikilinks 指向不存在页面
- 修正 39 个页面 source_count 不一致
- 将 142 个页面的 confidence 从 high 改为 medium（source_count=1 不满足 high 标准）
- 修复 3 个 source-summary 引用 raw/ 目录的错误
- 为 2 个孤立页面添加入口链接
- 修正 9 个页面中对不存在源文件的引用

## [2026-04-28] ingest | Deep Time Series Forecasting — 12 papers
Bulk ingest of 12 papers covering frequency-domain models, periodicity-based approaches, spatio-temporal forecasting, and a comprehensive survey.
Pages created: [[source-dualformer]], [[source-fedformer]], [[source-frets]], [[source-afe-tfnet]], [[source-timesnet]], [[source-penguin]], [[source-prnet]], [[source-autoformer]], [[source-tips]], [[source-st-resnet]], [[source-astgcn]], [[source-deep-time-series-survey]], [[dualsformer]], [[fedformer]], [[timesnet]], [[autoformer]], [[tslib]]
Pages updated: [[traffic-forecasting]], [[hyperd]], [[hybrid-periodicity-decoupling]], [[frequency-aware-residual-representation]], [[index]], [[log]]

## [2026-04-28] analysis | 新增"时序周期性建模文献梳理"专题研究
Added a comprehensive literature review page that synthesizes periodicity modeling approaches (frequency-domain, decomposition-based, adaptive period extraction) across the ingested time series forecasting papers.
Pages created: [[periodicity-modeling-in-time-series]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | DDPM: Denoising Diffusion Probabilistic Models (NeurIPS 2020)
里程碑式论文，首次证明扩散模型可生成高质量图像。CIFAR-10 达到 IS 9.46, FID 3.17。建立扩散模型与去噪得分匹配的数学等价性，提出简化训练目标 L_simple。
Pages created: [[source-ddpm]], [[ddpm]], [[ddpm-simplified-training-objective]]
Pages updated: [[diffusion-model]], [[ncsn]], [[index]], [[log]]

## [2026-04-28] ingest | Score-Based SDE: SMLD and DDPM unified (ICLR 2021)
里程碑论文，用 SDE 统一 NCSN (SMLD) 和 DDPM。引入 VE SDE、VP SDE、Sub-VP SDE，PC 采样器，概率流 ODE。CIFAR-10 取得 IS 9.89, FID 2.20，NLL 2.99 bits/dim。首次实现 1024×1024 生成。
Pages created: [[source-sde]], [[score-based-sde]], [[predictor-corrector-sampling]], [[probability-flow-ode]]
Pages updated: [[diffusion-model]], [[ncsn]], [[ddpm]], [[index]], [[log]]

## [2026-04-28] ingest | DPM-Solver: fast ODE solver for diffusion models (NeurIPS 2022)
快速扩散模型采样，利用半线性 ODE 结构在约 10 步内生成高质量样本。揭示 DDIM 等价于 DPM-Solver-1。提出一/二/三阶求解器，训练免费、即插即用。CIFAR-10: 4.70 FID@10 NFE, 2.87@20。
Pages created: [[source-dpm-solver]], [[dpm-solver]]
Pages updated: [[diffusion-model]], [[index]], [[log]]

## [2026-04-28] ingest | Consistency Models (ICML 2023)
单步生成扩散模型，通过学习 PF ODE 轨迹上任意点到起点的映射。支持蒸馏训练 (CD) 和独立训练 (CT) 两种模式。保留多步采样和零样本编辑能力。CIFAR-10: 1步 FID 3.55, 2步 2.93。
Pages created: [[source-consistency-models]], [[consistency-models]]
Pages updated: [[diffusion-model]], [[index]], [[log]]

## [2026-04-28] correction | 补充 HyperD 到周期性建模专题
HyperD (2025) 是短/长周期解耦的代表性工作，原专题遗漏。补充 HyperD 章节、频率分离策略表格、时间线标记。
Pages updated: [[periodicity-modeling-in-time-series]]

## [2026-04-28] ingest | Tutorial on Diffusion Models for Imaging and Vision
Ingested Stanley Chan's diffusion model tutorial (arXiv:2403.18103v3). This comprehensive tutorial covers VAE, DDPM, SMLD, SDE, and Fokker-Planck equations with rigorous mathematical foundations.
Pages created: [[source-chan-diffusion-tutorial]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | ConFormer: Accident-Informed Traffic Forecasting (KDD 2026)
Ingested KDD 2026 paper on accident-aware traffic forecasting. ConFormer addresses the critical gap where existing models fail during accidents which create non-stationary perturbations with directional shockwaves.
Pages created: [[source-conformer]], [[conformer]], [[guided-layer-normalization]], [[accident-aware-traffic-forecasting]]
Pages updated: [[traffic-forecasting]], [[index]], [[log]]

## [2026-04-28] enhancement | 添加论文发表 venue 信息
通过 web search 补充各模型的中稿会议/期刊信息，添加到时间线表格。确认 HyperD 中稿 AAAI 2026。
Pages updated: [[periodicity-modeling-in-time-series]]

## [2026-04-28] ingest | TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting
Ingested ICML 2025 论文，提出 Temporal Query (TQ) 技术——用周期性偏移的可学习向量作为注意力 Query，融合全局和局部变量相关性。极简架构（单层注意力 + 浅层 MLP）在 12 个数据集上取得 SOTA。
Pages created: [[source-tqn]], [[tqn]], [[temporal-query-technique]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | SparseTSF: Lightweight and Robust Time Series Forecasting via Sparse Modeling
Ingested TPAMI 2026 论文（ICML 2024 Oral），提出 Cross-Period Sparse Forecasting 技术——通过跨周期下采样将模型参数量降至 1k 以下。首次从理论上证明稀疏技术等价于隐式 L1 正则化。
Pages created: [[source-sparsetsf]], [[sparsetsf]], [[cross-period-sparse-forecasting]]
Pages updated: [[index]], [[log]], [[periodicity-modeling-in-time-series]]

## [2026-04-28] ingest | CycleNet: Modeling Periodic Patterns for Time Series Forecasting (NeurIPS 2024)
Ingested NeurIPS 2024 论文，提出 Residual Cycle Forecasting (RCF) 技术——使用可学习的循环周期 Q ∈ ℝ^(W×D) 显式建模时序数据的周期性模式，然后对残差分量进行预测。CycleNet 在电力、天气、能源等多个数据集上取得 SOTA，参数减少 90%+。RCF 可作为即插即用模块显著提升 PatchTST 和 iTransformer 的性能。
Pages created: [[source-cyclenet]], [[cyclenet]], [[residual-cycle-forecasting]], [[learnable-recurrent-cycles]], [[instance-normalization]]
Pages updated: [[index]], [[log]], [[periodicity-modeling-in-time-series]]

## [2026-04-28] ingest | ALiBi: Attention with Linear Biases Enables Input Length Extrapolation (ICLR 2022)
Ingested ICLR 2022 论文提出 Attention with Linear Biases (ALiBi) 方法——通过在注意力分数上添加与距离成线性关系的偏置来实现位置编码首次实现 Transformer 在训练短序列后能高效外推到更长序列进行推理。1.3B 参数模型在 L=1024 训练可外推到 L=2048 性能与 sinusoidal L=2048 相当训练速度快 11%内存节省 11%。
Pages created: [[source-alibi]], [[alibi]], [[linear-attention-bias]], [[position-extrapolation]], [[geometric-slope-schedule]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | YaRN: Efficient Context Window Extension of Large Language Models (2023)
Ingested 2023 论文提出 YaRN (Yet another RoPE extensioN method) 方法——整合 NTK-aware 插值、NTK-by-parts 插值和注意力温度缩放三项技术在仅使用 <0.1% 原始预训练数据微调后即可达到 SOTA 上下文扩展性能。将 Llama 2 7B/13B 从 4k 扩展到 128k。Dynamic-YaRN 在零微调情况下可扩展 2x 以上上下文。
Pages created: [[source-yarn]], [[yarn]], [[ntk-aware-interpolation]], [[ntk-by-parts-interpolation]], [[attention-temperature-scaling]], [[dynamic-scaling]], [[context-window-extension]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | Long Context, Less Focus: A Scaling Gap in LLMs (Gu et al., 2026)
Ingested 2026 论文提出 PAPerBench 基准评估长上下文下个性化生成和隐私推理能力揭示了统一的长上下文缩放 gap——随着上下文长度增加，所有模型的个性化与隐私性能均一致下降。理论分析表明 Attention Dilution 机制（注意力按 O(1/n) 衰减）是根本原因。大型模型渐进式下降，小模型提前崩溃。
Pages created: [[source-paperbench]], [[paperbench]], [[long-context-scaling-gap]], [[attention-dilution]], [[decoy-injection]], [[long-context-personalization]], [[privacy-reasoning]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | Vetcha 2026: Towards Infinite Length Extrapolation - A Unified Approach
Ingested 2026 论文提出统一位置编码框架 (GPE)，将注意力分数分解为乘法变换和加性偏置。基于此提出 Adaptive Positional Encoding (APE)，结合自适应频率调制和线性+对数+平方根衰减偏置。理论证明无限上下文外推的四个关键条件：收敛归一化、熵有界性、远距离相关性保持 (LDCP)、梯度位置敏感性 (GPS)。同时发布 LongTinyStories 数据集用于长上下文评估。
Pages created: [[source-vetcha-2026-towards-infinite-length-extrapolation]], [[adaptive-positional-encoding]], [[generalized-positional-encoding-framework]], [[convergent-normalization]], [[entropy-boundedness]], [[long-distance-correlation-preservation]], [[gradient-positional-sensitivity]], [[long-tiny-stories-dataset]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting (AAAI 2026)
Ingested AAAI-26 论文，提出首个纯端到端扩散模型 SimDiff，在时间序列点预测任务上取得 SOTA 结果，无需依赖任何外部预训练或联合训练的回归器。核心创新包括：Normalization Independence (N.I.) 技术缓解分布漂移、Median-of-Means (MoM) 集成将概率样本聚合为精确点估计、统一 Transformer 同时作为去噪器和预测器、无跳跃连接设计避免噪声放大。9 个数据集平均 rank 1.33，推理速度比现有扩散方法提升超 90%。
Pages created: [[source-simdiff]], [[simdiff]], [[normalization-independence]], [[median-of-means-ensemble]], [[patch-based-tokenization]], [[channel-independence]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | QUEST: A Robust Attention Formulation Using Query-Modulated Spherical Attention (ICLR 2026)
Ingested ICLR 2026 论文提出 QUEST (Query-modulated Spherical Attention) 方法——通过仅对键进行 ℓ2 归一化来消除键范数对注意力的"窃取"效应，同时保持每个查询独立控制其注意力锐度。核心洞见：查询范数控制锐度、键范数导致"全局注意力窃取"、Q-K 交叉依赖导致训练不稳定。实验验证：标准注意力在 ViT-Base/Large 上训练崩溃，QUEST 可稳定训练所有规模；ImageNet Top-1 提升 0.5-6.5%；对抗攻击和数据损坏下更鲁棒。
Pages created: [[source-quest]], [[quest-attention]], [[key-normalization]], [[attention-logit-explosion]], [[attention-entropy-collapse]], [[spurious-patterns]]
Pages updated: [[index]], [[log]]

## [2026-04-28] maintenance | ELBO concept page
Created concept page for Evidence Lower Bound (ELBO) covering its definition, derivation via Jensen's inequality and KL divergence, forms in VAE and diffusion models (VDM), and importance in variational inference. Page includes required frontmatter and inline citation placeholder.
Pages created: [[elbo]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | NCSN: Generative Modeling by Estimating Gradients of the Data Distribution (Song & Ermon, 2020)
Ingested NeurIPS 2019/2020 paper proposing Noise Conditional Score Networks (NCSN). Core innovations: score matching for score estimation, multi-noise-level perturbation to handle manifold hypothesis and low-density regions, annealed Langevin dynamics for sampling. Achieved SOTA Inception Score 8.87 on CIFAR-10 (unconditional), FID 25.32.
Pages created: [[source-ncsn]], [[ncsn]], [[score-based-generative-modeling]], [[annealed-langevin-dynamics]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | EDM: Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., NeurIPS 2022)
Ingested Karras et al. NeurIPS 2022 paper presenting unified design space for diffusion models. Core contributions: Heun 2nd-order ODE solver, EDM preconditioning (cskip/cout/cin), log-normal noise distribution, non-leaking augmentation. Achieved CIFAR-10 FID 1.79 (conditional), 1.97 (unconditional), ImageNet-64 FID 1.36.
Pages created: [[source-edm]], [[edm]], [[edm-design-space]], [[heun-sampler]], [[edm-preconditioning]], [[edm-stochastic-sampler]], [[edm-noise-distribution]], [[non-leaking-augmentation]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | LDM: High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., CVPR 2022)
Ingested Rombach et al. CVPR 2022 paper presenting latent diffusion model. Core contribution: perceptual compression via pretrained autoencoders (f=4-16), latent space diffusion training, cross-attention conditioning for flexible multimodal conditioning. Achieved CelebA-HQ FID 5.11, text-to-image FID 12.63 on MS-COCO, class-conditional ImageNet FID 3.60.
Pages created: [[source-rombach-ldm-2022]], [[latent-diffusion-models]], [[perceptual-compression]], [[cross-attention-conditioning]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | Neural ODE: Neural Ordinary Differential Equations (Chen et al., NeurIPS 2018)
Ingested Chen et al. NeurIPS 2018 paper presenting Neural ODE. Core contribution: continuous-depth networks via ODE solver, adjoint sensitivity method for memory-efficient backprop, instantaneous change of variables formula for continuous normalizing flows (CNF). Achieved MNIST 0.42% error, density estimation SOTA on Two Circle/Two Moons.
Pages created: [[source-neural-ode]], [[neural-ordinary-differential-equation]], [[adjoint-sensitivity-method]], [[continuous-normalizing-flow]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | Glow (NeurIPS 2018)
归一化流生成模型，引入可逆 1×1 卷积层替代 RealNVP 的固定通道置换。ActNorm 层解决小批量训练问题。首个高效生成 256×256 高分辨率图像的似然模型。CIFAR-10 bits/dim 3.35, ImageNet 64×64 3.81。
Pages created: [[source-glow]], [[glow]], [[normalizing-flow]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | Flow Matching (NeurIPS 2023)
Flow Matching 提出无需模拟的训练 CNF 框架，通过条件概率路径构造和条件流匹配 (CFM) 目标实现。核心贡献：1) FM 目标直接回归向量场；2) CFM 目标与 FM 梯度等价；3) 高斯条件路径的解析向量场公式；4) OT 路径比扩散路径更简单高效。OT 路径：直线轨迹、恒定方向。CIFAR-10 FID 6.35 (OT) vs 7.48 (DDPM)，采样 NFE 142 vs 274。
Pages created: [[source-flow-matching]], [[flow-matching]], [[optimal-transport]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | Shortcut Models (arXiv 2025)
Shortcut Models 提出单网络、单训练阶段的少步/单步生成模型。核心思想：不仅根据噪声水平 t，还根据期望步长 d 调节网络。训练目标：Flow Matching 目标 (d=0) + 自一致性目标 (d>0)。自一致性约束：s(t,2d) = 0.5*s(t,d) + 0.5*s(t+d,d)。优势：无需两阶段训练、灵活推理预算、仅比基础扩散模型多 16% 计算量。CelebA-HQ-256: 1步 FID 20.5 vs Flow Matching 280.5。ImageNet-256: 1步 FID 40.3 vs 324.8。
Pages created: [[source-shortcut-models]], [[shortcut-models]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | TIPS: Integrating Inductive Biases in Transformers via Distillation for Financial Time Series Forecasting (AAAI 2026)
TIPS 提出金融时序预测需要"状态依赖的归纳偏置适应"——不同市场环境下需要不同的归纳偏置（因果性、局部性、周期性）。核心创新：1) 通过注意力掩码训练 7 个偏置专业化教师；2) 正则化知识蒸馏将偏置融合到单一学生模型；3) 发现"合并惩罚"现象——直接训练多偏置模型反而性能下降。TIPS 将 ALiBi 的距离衰减作为"局部性"归纳偏置之一应用于金融时序。在四个股票市场取得 SOTA，年化收益 +55%，Sharpe +9%，Calmar +16%。
Pages created: [[source-tips]], [[tips]]
Pages updated: [[index]], [[log]], [[alibi]]

## [2026-04-29] ingest | SIREN-RoPE: Temporal and Semantic Rotary Encoding (arXiv 2026)
首篇 ingest 2026-04-27 新发布的 arXiv 论文，提出将 RoPE 旋转流形从固定序数索引扩展为可学习的时间条件化空间。核心贡献：1) 双分支 SIREN-DNN 将时间戳映射为旋转角，捕获日/周周期和近因衰减；2) 可学习频率缩放替代固定逆频率；3) 可学习门控 λ 融合时间与序数信号。在 LinkedIn 生产社交信息流数据集上，三个参与度任务的校准和排序指标均取得一致提升，额外参数量仅 0.2%。
Pages created: [[source-siren-rope]], [[siren-rope]], [[dual-branch-siren]], [[temporal-rotation]], [[ordinal-temporal-fusion]], [[learnable-frequency-scaling]]
Pages updated: [[index]], [[log]]

## [2026-04-29] ingest | CBSA: Towards Interpretable and Efficient Attention (NeurIPS 2025)
Wen, Huang & Li (BUPT) 提出 CBSA (Contract-and-Broadcast Self-Attention)，一种通过算法展开从 MCR² 优化目标推导出的可解释且高效的注意力机制。核心贡献：1) 引入代表性 token 概念，将"压缩所有 token"转化为"收缩少数代表"，实现线性复杂度；2) CBSA 可统一 softmax/linear/channel/agent attention 作为不同代表结构下的实例；3) CBT 在 ImageNet-1K 以 ViT-S 30% 参数达到 71.4% (vs 72.4%)，语义分割 ADE20K mIoU 超越 Segmenter 1.5%。
Pages created: [[source-cbsa]], [[cbsa]], [[cbt]], [[crate-white-box-transformer]], [[algorithm-unrolling]], [[mcr2]], [[coding-rate]], [[union-of-subspaces-model]], [[contract-and-broadcast-mechanism]], [[representative-token-extraction]]
Pages updated: [[index]], [[log]]

## [2026-04-29] ingest | FaST: Long-Horizon Forecasting for Large-Scale Spatial-Temporal Graph via MoE (KDD 2026)
Zhao, Zhong, Wang, Wen, Jin, Liang, Wan, Wu 提出 FaST 框架，解决大规模时空图长视野预测的计算瓶颈。核心创新：1) 异质性感知 MoE (HA-MoE) 使用 GLU experts 和动态路由解决 expert 极化；2) 自适应图代理注意力 (AGA-Att) 用 a ≪ N 个代理 tokens 将空间复杂度从 O(N²) 降至 O(Na)。首次实现 672 步（1 周）预测在 8600 节点上可训练，MAE 提升 4.4%-18.4%，推理速度 1.3x-2.2x SOTA。Dense MoE 设计配合 GLU 并行化实现高效计算。
Pages created: [[source-fast-long-horizon-forecasting]], [[mixture-of-experts]], [[adaptive-graph-agent-attention]], [[gated-linear-units]], [[large-scale-spatial-temporal-graph]]
Pages updated: [[traffic-forecasting]], [[index]], [[log]]

## [2026-04-29] ingest | UniCA: Unified Covariate Adaptation for Time Series Foundation Model (ICLR 2026)
Han, Liu, Li, Deng, Jiang, Sun, Yu, Wang, Lu, Ma, Ye, Zhan (Nanjing University & Ant Group) 提出 UniCA 框架，解决时间序列基础模型（TSFMs）无法处理异构协变量（分类/图像/文本）的问题。核心创新：1) 协变量同质化（Covariate Homogenization）通过预训练编码器（CLIP/BERT）+ 线性投影将异构协变量转换为统一表示；2) 注意力双融合模块（Pre-fusion + Post-fusion）在 TSFM 编码前后双阶段注入协变量信息；3) 保持 TSFM 主干冻结，仅训练轻量级融合模块。UniCA 是首个系统化处理 TSFMs 异构协变量适应问题的通用框架，在 12 个单模态数据集和 2 个多模态基准（MMSP、Time-MMD）上超越 ChronosX、TTM-R2 等基线方法。
Pages created: [[source-unica]], [[unica]], [[covariate-homogenization]], [[covariate-fusion-module]], [[unified-covariate-adaptation]]
Pages updated: [[instance-normalization]], [[normalization-independence]], [[timesnet]], [[tqn]], [[source-deep-time-series-survey]], [[index]], [[log]]

## [2026-04-30] ingest | Muon: An optimizer for hidden layers in neural networks (Jordan, 2024)
Keller Jordan 提出 Muon 优化器，针对神经网络隐藏层的 2D 参数。核心创新：在 SGD-动量更新后应用 Newton-Schulz 迭代进行正交化。实验结果：CIFAR-10 速度纪录 3.3→2.6 A100-秒，NanoGPT 速度纪录提升 1.35x，1.5B 模型训练 10h vs 13.3h AdamW。提出竞争性任务框架来解决优化器研究中的基线调优不足问题。
Pages created: [[source-muon-optimizer]], [[source-kellerjordan-muon-blog]], [[muon-optimizer]], [[newton-schulz-iteration]], [[gradient-orthogonalization]]
Pages updated: [[index]], [[log]]

## [2026-04-30] ingest | Muon优化器赏析：理论补充 (苏剑林, 2024)
补充科学空间的深度理论分析，从谱范数视角解释 Muon 的有效性。核心洞见：1) msign 是 sign 函数的矩阵推广；2) Muon 等价于谱范数约束下的最速下降；3) 当 Shampoo 的 β=0 时与 Muon 等价；4) 2015 年论文已提出类似算法 (Stochastic Spectral Descent)。详细推导了 Newton-Schulz 迭代的系数优化过程。
Pages created: [[source-kexue-muon-analysis]]
Pages updated: [[muon-optimizer]], [[newton-schulz-iteration]], [[index]], [[log]]

## [2026-05-03] ingest | IGSTGNN: Incident-Guided Spatiotemporal Traffic Forecasting (KDD 2026)
Ingested KDD 2026 论文，提出 IGSTGNN 框架通过 ICSF + TIID 两个即插即用模块显式建模非重复性事件对交通预测的时空影响。基于 XTraffic 基准构建三个事件对齐数据集，SOTA 全面超越。ICSF/TIID 可集成到 AGCRN、GWNET、STTN、D2STGNN 等骨干网络。
Pages created: [[source-incident-guided-st-forecasting]], [[igstgnn]], [[incident-context-spatial-fusion]], [[temporal-incident-impact-decay]]
Pages updated: [[accident-aware-traffic-forecasting]], [[traffic-forecasting]], [[large-scale-spatial-temporal-graph]], [[conformer]], [[index]], [[log]]

## [2026-05-03] ingest | MoST: A Foundation Model for Multi-modality Spatio-temporal Traffic Prediction (KDD 2026)
MoST 是首个多模态时空交通预测基础模型，通过 SNR 自适应模态选择和多模态引导空间专家实现零样本跨城市泛化。在五个大规模数据集上零样本超越所有基线（包括 OpenCity 基础模型和多数全量训练模型）。
Pages created: [[source-most]], [[most]], [[multi-modality-refinement]], [[multi-modality-guided-spatial-expert]], [[spatio-temporal-foundation-model]]
Pages updated: [[traffic-forecasting]], [[multimodal-time-series-forecasting]], [[large-scale-spatial-temporal-graph]], [[mixture-of-experts]], [[timesfm]], [[chronos]], [[unified-covariate-adaptation]], [[index]], [[log]]

## [2026-05-03] ingest | Aurora: Towards Universal Generative Multimodal Time Series Forecasting (arXiv 2026)
Aurora 是首个多模态时间序列基础模型，支持文本/图像/数值多模态输入和零样本推理。通过 Modality-Guided Self-Attention 注入领域知识，Prototype-Guided Flow Matching 实现生成式概率预测。在 5 个基准上单模态和多模态场景均 SOTA。
Pages created: [[source-aurora]], [[aurora]], [[modality-guided-self-attention]], [[prototype-guided-flow-matching]], [[generative-time-series-forecasting]]
Pages updated: [[simdiff]], [[chronos]], [[timesfm]], [[mindts]], [[vot]], [[most]], [[flow-matching]], [[multimodal-time-series-forecasting]], [[index]], [[log]]

## [2026-05-12] ingest | STD-MAE: Spatial-Temporal-Decoupled Masked Pre-training for Spatiotemporal Forecasting (IJCAI-2024)

Downloaded arXiv 2312.00516 PDF and ingested STD-MAE paper by Gao et al. (The University of Tokyo, SUSTech, UTS, IJCAI-2024). STD-MAE proposes a spatial-temporal-decoupled masked pre-training framework using two decoupled masked autoencoders for spatiotemporal forecasting. Key innovations: (1) Spatial-Temporal-Decoupled Masking — separately masking along spatial (S-Mask) and temporal (T-Mask) dimensions to learn clear heterogeneity; (2) non-architecture-modifying integration — pre-trained representations seamlessly added to any downstream predictor's hidden states; (3) identification of "spatiotemporal mirage" — a fundamental limitation of end-to-end models under short input windows. Achieves SOTA across PEMS03/04/07/08, METR-LA, and PEMS-BAY with 22.6%-72.5% pre-training speedup compared to STEP.

创建的页面：[[source-2312-00516-std-mae]], [[std-mae]], [[spatiotemporal-mirage]]
更新的页面：[[traffic-forecasting]], [[index]], [[log]]

## [2026-05-13] ingest | ELF: Embedded Language Flows (arXiv 2605.10938, MIT)

Downloaded arXiv PDF (2605.10938) and ingested ELF paper by Hu, Qiu, Li, Kim, Lu, Zhao, Andreas, He (MIT, equal contribution). ELF is a continuous diffusion language model based on Flow Matching that operates entirely in continuous embedding space, discretizing only at the final time step. Key contributions: (1) showing continuous DLMs can be highly competitive with proper design — the performance gap vs discrete DLMs is due to algorithmic design, not inherent discreteness of language; (2) shared-weight denoiser-decoder via x-prediction parameterization; (3) native CFG support for language diffusion, enabling direct inheritance of image-domain diffusion advances; (4) 10x fewer training tokens than leading discrete DLMs, with better quality at fewer sampling steps. ELF achieves Gen. PPL 24 @ 32 steps (OWT), BLEU 26.4 (WMT14 De-En), and ROUGE-1 36.0 (XSum), outperforming both discrete (MDLM, Duo) and continuous (FLM, LangFlow) baselines.

Pages created: [[source-elf-embedded-language-flows]], [[elf]], [[continuous-diffusion-language-model]]
Pages updated: [[flow-matching]], [[index]], [[log]]

## [2026-05-13] ingest | Back to Basics: Let Denoising Generative Models Denoise (arXiv 2511.13720, MIT)

Downloaded arXiv PDF (2511.13720) and ingested paper by Tianhong Li and Kaiming He (MIT). The paper argues that under the manifold assumption, predicting clean data (x-prediction) is fundamentally different from predicting noise (ε-prediction) or velocity (v-prediction), because clean data lies on a low-dimensional manifold while noised quantities fill the entire high-dimensional space. Proposes JiT (Just image Transformers) — a plain ViT on pixel patches with x-prediction, achieving competitive generation quality without tokenizers, pre-training, or extra losses. Key results: JiT-G/16 achieves 1.82 FID on ImageNet 256×256 (383 Gflops); JiT-G/32 achieves 1.78 FID on ImageNet 512×256 (384 Gflops); ε-/v-prediction fail catastrophically at high patch dimensions. The paper also reveals that EDM's pre-conditioner deviates from x-prediction and similarly fails in high-dimensional settings.

Pages created: [[source-back-to-basics-let-denoising-generative-models-denoise]], [[jit]], [[x-prediction]]
Pages updated: [[diffusion-model]], [[edm-design-space]], [[index]], [[log]]

## [2026-05-13] ingest | Flow-OPD: On-Policy Distillation for Flow Matching Models (arXiv 2605.08063)

Downloaded arXiv PDF (2605.08063) and ingested Flow-OPD paper by Fang, Huang et al. (USTC, UCLA, CUHK, Xiaohongshu). Flow-OPD is the first framework to integrate On-Policy Distillation (OPD) into Flow Matching post-training, addressing two multi-task alignment bottlenecks: reward sparsity and gradient interference. Proposes two-stage alignment: (1) single-reward GRPO fine-tuning yields domain-specialized teachers; (2) multi-teacher OPD with Cold-Start, task-routing labeling, and dense trajectory-level supervision consolidates expertise into a single student. Introduces Manifold Anchor Regularization (MAR) with a task-agnostic aesthetic teacher to prevent RL-induced aesthetic degradation. Built on SD3.5-Medium, raises GenEval from 63→92 and OCR from 59→94, achieving ~10-point improvement over vanilla GRPO with emergent teacher-surpassing effect.

创建的页面：[[source-flow-opd]], [[flow-opd]], [[manifold-anchor-regularization]]
更新的页面：[[flow-grpo]], [[index]], [[log]]

## [2026-05-13] ingest | FrèqFlow/SpectFlow: Long-term forecasting using lightweight flow matching (NeurIPS 2025)

Downloaded arXiv 2511.16426 PDF 并 ingest。FrèqFlow（别名 SpectFlow）由 Moghadas 等人（Vrije Universiteit Brussel & imec, NeurIPS 2025）提出，首次将条件流匹配（Conditional Flow Matching）引入频域进行多元时间序列确定性预测。核心创新：(1) 复值线性层在频域中插值频谱，通过复乘法建模幅度缩放和相位平移；(2) 低通滤波保留 6 次谐波内低频结构；(3) 流匹配头专用于残差学习，频率插值头提供趋势和季节性。仅 89k 参数（比扩散模型小一个数量级），在 Brussels/PEMS08/PEMS04 三个交通数据集上平均 RMSE 提升 7%，超越 [[d3vae|GCRDD]]/DiffSTG/PriSTI/SpecSTG 和 Moirai-MoE 基础模型。

创建的页面：[[source-2511-16426]], [[freqflow-ts]]
更新的页面：[[traffic-forecasting]], [[flow-matching]], [[generative-time-series-forecasting]], [[index]], [[log]]

## [2026-05-31] analysis | UrbanDiT Paper River
4 层向后引文搜索，追溯 UrbanDiT (NeurIPS 2025) 从基础模型到最终产品的完整演化链。
创建的页面：[[urbandit-paper-river]]
涉及的源文件：STGCN (IJCAI 2018), DDPM (NeurIPS 2020), MAE (CVPR 2022), PatchTST (ICLR 2023), DiT (ICCV 2023), CSDI (NeurIPS 2021), STD-MAE (IJCAI 2024), GPT-ST (NeurIPS 2023), ST-SSL (AAAI 2023), GPD (ICLR 2024), UniST (KDD 2024), UrbanGPT (KDD 2024), OpenCity (2024)

## [2026-05-31] ingest | TimeGrad: Autoregressive Denoising Diffusion for Multivariate Probabilistic Time Series Forecasting (ICML 2021)

Ingest Rasul, Seward, Schuster & Vollgraf (Zalando Research, ICML 2021) TimeGrad。首个将 DDPM 扩散模型引入多变量时间序列概率预测的方法。核心设计：RNN 自回归时间编码（2 层 LSTM, h=40）+ DDPM 条件扩散去噪（8 块膨胀卷积残差网络, GAU 门控）取代预设参数化输出分布。6 数据集上 5 个 CRPS_sum 第一；消融揭示 N≈10 即接近最优（vs 图像扩散 1000 步）——RNN 隐状态已提供强引导信号。为 CSDI/DiffSTG/SpecSTG 等后续扩散+时序工作奠定范式基础。主要局限：推理需 2400 次前向（24 步×100），自回归串行无并行。

创建的页面：[[source-timegrad]], [[timegrad]]
更新的页面：[[index]], [[log]]

## [2026-05-31] ingest | DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with Denoising Diffusion Models (AAAI 2023)

Ingest Wen et al. (北京交通大学/NUS/DAMO Academy/HKUST(GZ), AAAI 2023) DiffSTG。首个将 DDPM 扩散推广到时空图预测的非自回归框架。核心创新：(1) 广义条件扩散——将历史和未来统一为 X^all，通过 mask 使去噪网络同时学习历史重建和未来预测；(2) UGnet——TCN 门控因果卷积 + Unet 多尺度 + vanilla GCN 的三维异构去噪架构；(3) 非自回归一次全窗口生成 + DDIM 子集采样 + 尾步复用，推理加速约 3200× vs TimeGrad。PEMS08/AIR-BJ/AIR-GZ 三数据集 CRPS 降 4.3–14.3%。确定性精度落后最佳 STGNN 约 5–10%（ELBO 系统性局限）。开创扩散+STG 范式，后续催生 SpecSTG/D3/DiffLoad/UrbanDiT。

创建的页面：[[source-diffstg]], [[diffstg]]
更新的页面：[[index]], [[log]]

## [2026-05-31] integration | TimeGrad + DiffSTG cross-linking and index update

将 TimeGrad 和 DiffSTG wiki 页面接入共享文件，修复交叉链接。更新 [[index]] 添加四个新页面条目，在 [[source-diffstg]] 和 [[diffstg]] 中将 TimeGrad 纯文本提及替换为 [[timegrad|TimeGrad]] wikilink，在 [[specstg]] 和 [[traffic-forecasting]] 中将已有 TimeGrad/DiffSTG 纯文本提及转为 wikilink，在 [[generative-time-series-forecasting]] 中添加 TimeGrad 作为扩散预测路线奠基工作的条目。创建 WHY 报告 `ingest-reports/timegrad-diffstg-2026-05-31.md`。

更新的页面：[[index]], [[source-diffstg]], [[diffstg]], [[specstg]], [[traffic-forecasting]], [[generative-time-series-forecasting]]

## [2026-05-31] ingest | VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training (Tong et al., NeurIPS 2022)

Ingest VideoMAE paper (Tong, Song, Wang & Wang, Nanjing University / Tencent AI Lab / Shanghai AI Lab, NeurIPS 2022, arXiv:2203.12602). VideoMAE extends MAE's masked autoencoding paradigm to video, introducing two key designs: (1) tube masking — same spatial position masked/kept across all frames to prevent temporal information leakage; (2) extremely high mask ratio (90-95%) exploiting video's spatial + temporal redundancy. Key contributions: vanilla ViT trained from scratch on video data alone achieves 87.4% on Kinetics-400 and 75.4% on SSV2 (no extra data), data-efficient learner works with only 3.5k videos (HMDB51 62.6% vs MoCo v3 39.2%), and data quality > data quantity proven (42k in-domain > 240k cross-domain). Information density ladder: language 15% → image 75% → video 90%+ mask ratio. Created source-summary and technique pages; added wikilinks to 4 existing pages where VideoMAE or masked autoencoding was mentioned.

创建的页面：[[source-videomae]], [[videomae]]
更新的页面：[[mae]], [[source-mae]], [[std-mae]], [[patchtst]], [[index]], [[log]]

## [2026-05-31] ingest | D³VAE (GCRDD): Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement (NeurIPS 2022)

Ingest Li, Lu, Wang & Dou (Baidu Research / Zhejiang University, NeurIPS 2022, arXiv:2301.03028) D³VAE paper。首个将耦合扩散（Coupled Diffusion）+ BVAE 逆向过程 + 多尺度降噪得分匹配（DSM）+ 潜变量 Total Correlation 解耦四者统一为端到端框架的生成式时间序列预测模型。核心设计：(1) 耦合扩散——对输入序列 X 和目标序列 Y 以不同方差调度同步扩散，在不增加随机不确定性的前提下扩充数据分布空间，理论保证 Lemma 1-2；(2) BVAE 替换传统扩散的逆向过程，一次前向生成而无需 T 步迭代采样；(3) 多尺度 DSM 通过能量函数梯度跳跃将生成目标矫正回真实方向，同时提供不确定性显式估计；(4) TC 最小化解耦潜变量，不同维度对应不同时序模式（趋势/季节等）。6 真实数据集 + 2 合成数据集上平均 43% MSE 降低、23% CRPS 降低；Traffic 数据集 input-8-predict-8 设定下 MSE 降低 90%、CRPS 降低 73%。在后续 STG 文献（SpecSTG, FrèqFlow 等）中，该模型的图卷积变体被广泛引用为 GCRDD 基线——"当前最高效的 STG 扩散方法"（训练速度 1.0×, SpecSTG 以其为速度基准 3.33×）。D³VAE 区别于 [[timegrad|TimeGrad]]/[[diffstg|DiffSTG]]/[[csdi|CSDI]] 的核心特征在于"扩散作为数据扩充, VAE 作为生成引擎"的非自回归范式。主要局限：扩散偏差控制依赖精细调参（β, ω）、无监督解耦缺乏真实因子标签、GCRDD 形态的空间信息仅编码阶段利用。

创建的页面：[[source-gcrdd]], [[d3vae]]
更新的页面：[[index]], [[log]], [[traffic-forecasting]], [[generative-time-series-forecasting]], [[specstg]], [[freqflow-ts]], [[source-dcrnn]], [[source-2511-16426]], [[source-2401-08119-specstg]]

## [2026-05-31] ingest | GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks (Li et al., NeurIPS 2023)

Ingest GPT-ST paper (Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang, SCUT / HKU / PAZHOU LAB, NeurIPS 2023, arXiv:2311.04245). GPT-ST is the first general-purpose spatio-temporal pre-training framework that seamlessly integrates with 13 diverse downstream STGNN baselines without architecture modification. Core contributions: (1) customized temporal pattern encoding — time-specific and region-specific parameters generated via a parameter learner, using temporal hypergraph propagation (H_T=8 hyperedges); (2) hierarchical spatial pattern encoding — hypergraph capsule clustering network with dynamic routing soft-assigns regions to H_S=10 cluster centroids, plus a high-level cross-cluster hypergraph (H_M=16 hyperedges) modeling inter-cluster migration patterns; (3) cluster-aware adaptive mask strategy — progressively increases mask difficulty from intra-cluster (easy) to whole-cluster (hard) with rt=0.25 total mask ratio. All 13 baselines improved on all 4 datasets (PEMS08, METR-LA, NYC Taxi, NYC Citi Bike) across all 3 metrics. Classic baselines (STGCN, TGCN) benefit much more than advanced ones (MSDR, STWA). Pre-trains 26× faster (12.5s vs 327.8s/epoch) and outperforms STEP while using only 12-step input. Replaced stub [[source-gpt-st]] with formal source-summary; created [[gpt-st]] technique page; cross-linked [[traffic-forecasting]], [[spatio-temporal-foundation-model]], [[mae]], [[std-mae]], [[urbandit-paper-river]].

创建的页面：[[gpt-st]]
更新的页面：[[source-gpt-st]] (stub→formal), [[traffic-forecasting]], [[spatio-temporal-foundation-model]], [[mae]], [[std-mae]], [[urbandit-paper-river]], [[index]], [[log]]

## [2026-05-31] ingest | OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction (Li et al., arXiv 2024)

Ingest OpenCity paper (Zhonghang Li, Long Xia, Lei Shi, Yong Xu, Dawei Yin, Chao Huang, HKU / SCUT / Baidu, arXiv:2408.10269, August 2024). OpenCity is a spatio-temporal foundation model enabling zero-shot traffic prediction across unseen cities without any fine-tuning. Core innovations: (1) instance normalization replaces Z-score, eliminating dependence on training statistics; (2) patch embedding (P=12, hourly) compresses sequences and improves robustness; (3) TimeShift Transformer (PTTM + DTP dual attention) decouples periodic and dynamic patterns; (4) Laplacian eigenvector spatial encoding provides zero-external-data spatial context; (5) GCN mixed aggregation with α=0.05 for robust cross-city transfer. Pre-trained on 21 datasets (151M observations), achieves zero-shot performance surpassing full-shot baselines on 4/6 test datasets. vs UniST/UrbanGPT on CHI-TAXI: MAE 1.74 (OpenCity_mini 2M) vs 2.94 (UniST) vs 3.26 (UrbanGPT); inference 1.5s vs 45,000s (~30,000×). Fast adaptation with 3-epoch fine-tuning: SZ-DIDI MAE 2.42 vs best baseline 2.87, training 2.8s vs 46.8s. Three scales: mini 2M, base 5M, plus 26M. Replaced stub [[opencity]] (entity→technique) with full page; created formal source-summary; cross-linked [[spatio-temporal-foundation-model]], [[traffic-forecasting]], [[most]], [[gpt-st]].

创建的页面：[[source-opencity]]
更新的页面：[[opencity]] (stub→full technique page), [[spatio-temporal-foundation-model]], [[traffic-forecasting]], [[most]], [[index]], [[log]]

## [2026-06-01] ingest | GPD: Spatio-Temporal Few-Shot Learning via Diffusive Neural Network Generation (Yuan et al., ICLR 2024)

Ingest GPD paper (Yuan, Shao, Ding, Jin & Li, Tsinghua FIB Lab, ICLR 2024, arXiv:2402.11922)。GPD 是时空少样本学习的生成式预训练框架——核心创新在于参数空间预训练：使用 Transformer-based 扩散模型作为 hypernetwork，在多个 source cities 上学习从 prompt（spatial UKG + temporal MAE-style）条件生成时空预测模型参数的能力。与 DiffSTG/SpecSTG 在数据空间做扩散预测不同，GPD 的扩散过程发生在参数空间。Model-agnostic，兼容 STGCN/GWN/STID 三种 base model。4 数据集平均较最优 baseline 降低 7.87% 误差，长期预测优势显著（Baltimore Step 6 MAE -22.1% vs STGFSL）。位于 Tsinghua FIB Lab 研究脉络中从传统 transfer learning 向参数化少样本生成的过渡节点。

创建的页面：[[source-gpd]], [[gpd]]
更新的页面：[[source-urbandit]], [[urbandit-paper-river]], [[index]], [[log]]

## [2026-06-08] query | 多模态外生信息引导的长期时空预测：研究路线分析

结合用户提供的《2026 AI/ML 顶会论文汇编》（2048 篇 ICLR/ICML/CVPR/AAAI Oral/Highlight/最佳候选，2026-06-05 快照）与本知识库，对研究方向"多模态外生信息引导的长期时空预测"做系统调研与方向规划。方法：17 路并行挖掘（11 个 PDF 分区 + 6 个 wiki 集群，~1.8M tokens）综合出现状地图与缺口结构，再综合为分层研究议程。核心结论：(1) 中心缺口——无任何工作同时具备〔多模态外生融合〕＋〔显式长期影响机制〕＋〔因果去混杂〕＋〔可解释归因〕且在长 horizon 验证（E²-CSTP/IGSTGNN/VoT/FactoST/VisiFold/Swift 各缺一角）；(2) 重构——时空幻象本质是"消歧"问题，外生信息是未被点名的消歧器，可把 STD-MAE 掩码预训练扩出跨模态第三轴；(3) 统一架构论点——分解骨干 + 晚注入外生适配器 + 分模态分频滞后核 + 因果双分支 + 可靠性闸门；(4) 分层方向（快赢/核心赌注/登月/横切基础设施）与优先级（先做基准 EN1 + 滞后核 QW1，主线 CB2→CB1）。归档为 analysis 页，source_count=25，confidence=medium（描述性高、处方性为提案）。

创建的页面：[[multimodal-exogenous-guided-long-term-st-forecasting]]
更新的页面：[[index]], [[log]], [[spatiotemporal-mirage]], [[spatio-temporal-foundation-model-landscape]]

## [2026-06-09] ingest | 批量摄入时空/时序预测论文（13 成功 / 2 跳过）
通过 workflow 一次只开 2 个并发子代理，下载并摄入用户提供的 15 篇候选论文。
跳过 (2)：**pi-stgnn**（ICML 2025 *workshop*，仅 OpenReview 且 /pdf 返回 403，无 arXiv/PMLR 镜像）；**bigst**（VLDB host 从沙箱不可达 / TLS error 35，非 arXiv，ACM/ResearchGate 受阻）。
创建源页面 (13)：[[source-stop]]、[[source-fstllm]]、[[source-st-ttc]]、[[source-s2dbm]]、[[source-ratd]]、[[source-armd]]、[[source-doflow]]、[[source-k2vae]]、[[source-weathergfm]]、[[source-maginet]]、[[source-stamimputer]]、[[source-st-vision-llm]]、[[source-motm]]
创建实体页 (12)：[[stop]]、[[fstllm]]、[[st-ttc]]、[[s2dbm]]、[[ratd]]、[[armd]]、[[doflow]]、[[k2vae]]、[[weathergfm]]、[[maginet]]、[[stamimputer]]、[[st-vision-llm]]
创建概念/技术页 (28)：含 [[ood-generalization]]、[[distributionally-robust-optimization]]、[[kalman-filter]]、[[koopman-linearization-for-forecasting]]、[[causal-time-series-forecasting]]、[[weather-foundation-model]]、[[mask-aware-imputation-no-prefilling]]、[[few-shot-traffic-forecasting]]、[[test-time-computing-st]]、[[vision-language-traffic-forecasting]]、[[brownian-bridge-diffusion]]、[[sliding-window-diffusion]]、[[spectral-domain-calibration]]、[[centralized-message-passing]] 等。
更新页面 (~44)：为 43 个既有页面添加引用反向链接（[[traffic-forecasting]]、[[diffusion-models]]、[[timegrad]]、[[csdi]]、[[time-llm]]、[[imputeformer]]、[[urbangpt]]、[[continuous-normalizing-flow]] 等）；[[motm]] 富化为主要来源。

## [2026-06-09] contradiction | [[motm]] | 解决方案：corroboration（增强，非矛盾）
既有 [[motm]] 由评测论文（TMLR 2026）建立；新增原始论文 [[source-motm]]（AALTD 2025）与之一致 → 采用情况 1 式增强（非历史论断）；source_count 1→2，confidence medium→high。

## [2026-06-09] ingest | BigST（补摄入，用户提供 PDF）
此前因 VLDB host 从沙箱不可达而跳过的 BigST，用户提供 PDF（raw/bigst-pvldb2024.pdf，ACM DOI 10.14778/3641204.3641217）后补摄入。
创建：[[source-bigst]]、[[bigst]]、[[linearized-spatial-convolution]]、[[long-sequence-feature-extractor]]
更新（反向链接）：[[gwnet]]（直接前身，O(N²) 自适应邻接→O(N)）、[[large-scale-spatial-temporal-graph]]、[[ragc]]、[[traffic-forecasting]]、[[centralized-message-passing]]（恢复 BigST 链接）
仍跳过：PI-STGNN（用户无访问权限）。

## [2026-07-04] ingest (cross-linking) | CoRA 全方位整合：从孤立论文到方法全景

原始 CoRA ingest（2026-05-31）仅创建了 source-summary 和 entity 页面，缺乏与 wiki 中相关工作的交叉链接。本次执行"全方位 ingest"——将 CoRA 置于 TSFM 协变量适配方法的全景图中：
- 创建跨方法分析页面 [[tsfm-covariate-adaptation-comparison]]（CoRA/UniCA/DiTS/ChronosX/AdaPTS/Gen-P-Tuning 六路线系统对比）
- 创建通用概念页 [[zero-initialized-adaptation]]（LoRA→DiT→CoRA 的零初始化适配谱系）
- 为 CoRA/UniCA/Sundial/DiTS 等 7 个实体页面及 5 个技术页面添加双向交叉链接（共 30+ 新链接）
- CoRA 页面 source_count 从 1→4（新增 source-unica/source-dit/source-dits/source-sundial 引证），confidence 从 medium→high

创建的页面：[[tsfm-covariate-adaptation-comparison]], [[zero-initialized-adaptation]]
更新的页面：[[cora-tsfm]], [[source-cora]], [[unica]], [[dits]], [[mm-dit-for-time-series]], [[sundial]], [[heterogeneous-covariates]], [[covariate-homogenization]], [[covariate-fusion-module]], [[conditional-attention-pooling]]

## [2026-07-07] ingest | ExoST (arXiv:2509.05779)
ExoST 框架：首次系统研究时空预测外生变量建模的两大挑战（不一致变量效应 + 不平衡类型效应），提出 select-then-balance 即插即用范式。旧 slug 页面 [[source-select-then-balance]] 已被本页面取代。
创建的页面：[[source-exost]]
更新的页面：[[source-select-then-balance]]（标注 superseded → [[source-exost]]），[[index]]
## [2026-07-07] ingest | ExoLLM (WWW 2025)
首个 LLM-driven 外生变量预测方法。Meta-task Instruction 激活 LLM 从 NLP 到 FEV 的跨任务迁移，Multi-grained Prompts 捕获外生变量的多粒度影响，Dual TS-Text Attention 对齐文本-数值特征空间。
创建的页面：[[source-exollm]]
更新的页面：[[index]]
## [2026-07-07] ingest | Solar-VLM (arXiv:2604.04145)
Solar-VLM: unified multimodal VLM framework fusing satellite imagery, text weather reports, and time series with GAT + cross-site attention for multi-site solar power spatiotemporal prediction. Evaluated on 8 PV stations in Hebei, China — outperforms 7 baselines across all horizons (T=3 to 96).

创建的页面：[[source-solar-vlm]]
更新的页面：[[index]]

## [2026-07-07] ingest | From News to Forecast (NeurIPS 2024)
From News to Forecast: LLM generative agent iteratively filters news/event text, uses reflection + CoT-style reasoning to assess impact on time series, fuses with numerical data for prediction. Cross-linked with [[source-exollm]].

创建的页面：[[source-from-news-to-forecast]]
更新的页面：[[index]]

## [2026-07-07] ingest | STLLM (arXiv:2507.05258)
Spatio-Temporal LLM: multimodal STLLM fusing point cloud, video, and text for reasoning about environments and actions. REA dataset (5 tasks, 24K+ samples) + two STLLM baselines (STLLM-3D and STLLM-Aligner). Cross-linked with [[source-st-vision-llm]].

创建的页面：[[source-stllm]]
更新的页面：[[index]]

## [2026-07-07] ingest | Terra (NeurIPS 2024)
Terra: large-scale multimodal Earth spatiotemporal dataset — 6.48M global grids × 45 years hourly meteorological time series, geo-images, and LLM-generated explanatory text. Critical benchmark enabling multimodal exogenous ST research. Cross-linked with [[source-exost]], [[source-aurora]].

创建的页面：[[source-terra]]
更新的页面：[[index]]

## [2026-07-07] ingest | PI-MFM (2512.23056)
PI-MFM: Physics-informed multimodal foundation model for solving PDEs by M. Zhu, J. Sun, Z. Zhang, H. Schaeffer, L. Lu (Yale, JHU, Notre Dame, UCLA, 2025). First PDE-encoding MOL framework that directly enforces governing equations as physics losses during pretraining and adaptation, with automatic vectorized PDE residual computation from symbolic expressions. On 13 parametric 1D time-dependent PDE families, consistently outperforms purely data-driven counterparts especially under sparse data and enables zero-shot physics-informed fine-tuning to unseen PDE families (~1% error). PROSE backbone.

创建的页面：[[source-pi-mfm]]
更新的页面：[[index]]

## [2026-07-07] ingest | ExoTST (2410.12184)
ExoTST: Exogenous-Aware Temporal Sequence Transformer for Time Series Prediction by K. Tayal, A. Renganathan, X. Jia, V. Kumar, D. Lu (ORNL, UMN, Pitt, 2024). Treats past and future exogenous variables as distinct modalities with a cross-temporal fusion module (aggregation token + cross-attention), enabling autoregressive Transformer to incorporate current/projected exogenous drivers. Outperforms TiDE, PatchTST, iTransformer by 8-12% on carbon flux datasets. Robust to missing/noisy exogenous drivers.

创建的页面：[[source-exotst]]
更新的页面：[[index]]

## [2026-07-07] ingest | TimeXer — Empowering Transformers for Time Series Forecasting with Exogenous Variables

Ingested TimeXer paper (Wang et al., Tsinghua; NeurIPS 2024; arXiv:2402.19072). TimeXer proposes a dual-granularity representation: patch-wise self-attention for endogenous variables and variate-wise cross-attention for exogenous variables, with a learnable global token bridging the two. Achieves consistent SOTA on 12 benchmarks for forecasting with exogenous variables.

Created: [[source-timexer]]
Updated: [[index]]
Cross-linked: [[source-patchtst]], [[source-exost]]

## [2026-07-07] ingest | STG-Mamba — Spatial-Temporal Graph Learning via Selective State Space Model

Ingested STG-Mamba paper (Li et al., UNSW; arXiv:2403.12418). First exploration of selective SSM (Mamba) for STG prediction. Proposes GS3B (Graph Selective State Space Block) + KFGN (Kalman Filtering GNN) for adaptive graph structure upgrading with linear complexity. Surpasses Transformer-based SOTA on PeMS04, HZMetro, and KnowAir.

Created: [[source-stg-mamba]]
Updated: [[index]]
Cross-linked: [[source-diffstg]], [[source-s-mamba]], [[source-dst-mamba]]

## [2026-07-07] ingest | RAF — Retrieval Augmented Time Series Forecasting

Ingested RAF paper (Tire, Taga, Ildiz, Oymak; UT Austin / U. Michigan; arXiv:2411.08249). First principled RAG framework for time series foundation models. Formulates TS-R (Time-Series Retrieval) problem, proves two-layer Transformer can solve it. Shows RAF improves zero-shot forecasting across Chronos, Moirai, TimesFM, Lag-Llama with gains scaling with model size.

Created: [[source-raf]]
Updated: [[index]]
Cross-linked: [[source-time-llm]], [[chronos]], [[retrieval-augmented-spatio-temporal-forecasting]]
## [2026-07-08] 勘误 | ExoST
确认 ExoST (arXiv:2509.05779) 截至 2026-07 仅发布于 arXiv，未经同行评审。已更新 source-exost、source-select-then-balance、research-gaps-analysis、research-multimodal-exogenous-spatiotemporal、paper-river 笔记中的相关标注。

## [2026-07-12] ingest | KITE (ICML 2026)
KITE: Knowledge-Guided Probabilistic Modeling for Time Series Forecasting with Exogenous Variables (Cheng, Zhou, Shu, Guo; ECNU). 全维度 ingest，并显式串联 History-Conditional Manifold、Knowledge-Guided Conditioning、Classifier-Free Guidance。

创建的页面：[[source-kite]], [[kite]], [[history-conditional-manifold]], [[knowledge-guided-conditioning]], [[kite-manifold-guidance-chain]]
更新的页面：[[classifier-free-guidance]], [[flow-matching-forecasting]], [[tsflow]], [[gaussian-process-prior-flow-matching]], [[source-timexer]], [[index]]
源文件：raw/kite-cheng-2026.pdf（不可变拷贝自 Zotero BM79KEZT）

## [2026-07-12] ingest | DAG (IJCAI 2026)
DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables (Qiu, Zhu, Li, Wu, Yang, Hu; ECNU). 沿时间+通道双维发现并注入外生-内生相关性。

创建的页面：[[source-dag]], [[dag]], [[dual-correlation-injection]]
更新的页面：[[kite]], [[source-kite]], [[index]]
源文件：raw/dag-qiu-2026.pdf（不可变拷贝自 Zotero 3MBN63QI）

## [2026-07-13] ingest | DistDF (ICLR 2026)
DistDF: Distribution-aware Direct Forecast for time-series forecasting via joint-distribution Wasserstein alignment. 提出 MSE 的自相关偏差理论（Theorem 3.1），用联合分布 Wasserstein discrepancy 替代似然估计，Bures-Wasserstein 实现均值+协方差对齐，模型无关即插即用。

创建的页面：[[source-distdf]], [[autocorrelation-bias]], [[joint-distribution-wasserstein-alignment]]
更新的页面：[[index]]
源文件：raw/DistDF_Wang_2026_ICLR.pdf
