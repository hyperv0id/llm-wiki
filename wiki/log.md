---
title: Log
type: concept
created: 2026-04-26
last_updated: 2026-04-28
tags:
  - meta
---

# Wiki Log

Chronological record of all wiki activity.

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
Ingested 2026 论文提出 PAPerBench 基准评估长上下文下个性化生成和隐私推理能力揭示了统一的长上下文缩放 gap——随着上下文长度增加，所有模型的个性化与隐私性能均一致下降。理论分析表明 Attention Dilution 机制（注意力按 O(1/n) 衰减）是根本原因。大��型渐进式下降，小模型提前崩溃。
Pages created: [[source-paperbench]], [[paperbench]], [[long-context-scaling-gap]], [[attention-dilution]], [[decoy-injection]], [[long-context-personalization]], [[privacy-reasoning]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | Vetcha 2026: Towards Infinite Length Extrapolation - A Unified Approach
Ingested 2026 论文提出统一位置编码框架 (GPE)，将注意力分数分解为乘法变换和加性偏置。基于此提出 Adaptive Positional Encoding (APE)，结合自适应频率调制和线性+对数+平方根衰减偏置。理论证明无限上下文外推的四个关键条件：收敛归一化、熵有界性、远距��相关性保持 (LDCP)、梯度位置敏感性 (GPS)。同时发布 LongTinyStories 数据集用于长上下文评估。
Pages created: [[source-vetcha-2026-towards-infinite-length-extrapolation]], [[adaptive-positional-encoding]], [[generalized-positional-encoding-framework]], [[convergent-normalization]], [[entropy-boundedness]], [[long-distance-correlation-preservation]], [[gradient-positional-sensitivity]], [[long-tiny-stories-dataset]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting (AAAI 2026)
Ingested AAAI-26 论文，提出首个纯端到端扩散模型 SimDiff，在时间序列点预测任务上取得 SOTA 结果，无需依赖任何外部预训练或联合训练的回归器。核心创新包括：Normalization Independence (N.I.) 技术缓解分布漂移、Median-of-Means (MoM) 集成将概率样本聚合为精确点估计、统一 Transformer 同时作为去噪器和预测器、无跳跃连接设计避免噪声放大。9 个数据集平均 rank 1.33，推理速度比现有扩散方法提升超 90%。
Pages created: [[source-simdiff]], [[simdiff]], [[normalization-independence]], [[median-of-means-ensemble]], [[patch-based-tokenization]], [[channel-independence]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | QUEST: A Robust Attention Formulation Using Query-Modulated Spherical Attention (ICLR 2026)
Ingested ICLR 2026 论文提出 QUEST (Query-modulated Spherical Attention) 方法——通过仅对键进行 ℓ2 归一化来消除键范数对注意力的"窃取"效应，同时保持每个查询独立控制其注意力锐度。核心洞见：查询范数控制锐度、键范数导致"全局注意力窃取"、Q-K 交叉依赖导致训练不稳定。实验验证：标准注意力在 ViT-Base/Large 上训练崩溃，QUEST 可稳定训练所有规模；ImageNet Top-1 提升 0.5-6.5%；对抗攻击和数据损坏下更鲁棒。
Pages created: [[source-quest]], [[quest-attention]], [[key-normalization]], [[attention-logit-explosion]], [[attention-entropy-collapse]], [[spurious-patterns-in-attention]]
Pages updated: [[index]], [[log]]

## [2026-04-28] maintenance | ELBO concept page
Created concept page for Evidence Lower Bound (ELBO) covering its definition, derivation via Jensen's inequality and KL divergence, forms in VAE and diffusion models (VDM), and importance in variational inference. Page includes required frontmatter and inline citation placeholder.
Pages created: [[elbo]]
Pages updated: [[index]], [[log]]

## [2026-04-28] ingest | NCSN: Generative Modeling by Estimating Gradients of the Data Distribution (Song & Ermon, 2020)
Ingested NeurIPS 2019/2020 paper proposing Noise Conditional Score Networks (NCSN). Core innovations: score matching for score estimation, multi-noise-level perturbation to handle manifold hypothesis and low-density regions, annealed Langevin dynamics for sampling. Achieved SOTA Inception Score 8.87 on CIFAR-10 (unconditional), FID 25.32.
Pages created: [[source-ncsn]], [[ncsn]], [[score-based-generative-modeling]], [[annealed-langevin-dynamics]]
Pages updated: [[index]], [[log]]
