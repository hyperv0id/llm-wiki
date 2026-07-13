---
title: "ICML 2026 Spotlight 与多模态外生信息引导长期时空预测的相关分析"
type: analysis
tags:
  - icml-2026
  - spatiotemporal
  - time-series
  - multimodal
  - exogenous-guidance
  - long-term-forecasting
created: 2026-07-12
last_updated: 2026-07-12
source_count: 1
confidence: medium
status: active
---

# ICML 2026 Spotlight 与"多模态外生信息引导的长期时空预测"的相关分析

> 本页基于 ICML 2026 的 538 篇 spotlight 论文（截至 2026-07-12 自 OpenReview 抓取）[^src-icml-2026-spotlight-papers]，从研究方向"多模态外生信息引导的长期时空预测"出发，挑出**直接相关**与**方法可迁移**的论文，并说明可用之处。
>
> 方向拆解：**时空预测**（spatiotemporal forecasting，交通/天气/传感场）× **多模态**（文本/图像/图等异质信息）× **外生信息引导**（exogenous guidance，用目标系统之外的变量/事件/语义引导预测）× **长期**（long-horizon，远距离 rollout 稳定性）。

## 一、直接相关：面向时空/时序预测且引入外部信息或多模态

这些论文本身就落在"时空或时序预测 + 外部信息引导"的交集上，是该方向最近的邻居，方法、实验设置、benchmark 可直接对照或继承。

### 1. From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space（TESS）

> 把**文本信息**（新闻、事件描述等外生信号）引入**时序预测**，专门解决事件驱动的非平稳性。核心洞察：文本表达时间影响是"隐式、定性"的，而预测模型需要"显式、定量"信号，两者之间存在根本 modality gap。TESS 构造一个 **Temporal Evolution Semantic Space** 作为中间瓶颈，由 LLM 通过结构化 prompting 抽出四类可解释的数值化时间原语——分布漂移、波动率、形状、滞后——再用 confidence-aware gating 过滤噪声 token。在四个真实数据集上把预测误差最多降低 29%。

- **为什么直接相关**：这是"外生信息（文本）引导时序预测"的范式样本，是本方向"多模态外生引导"的最直接同形问题。它的"中间语义空间 + 数值化原语 + 置信门控"三段式，几乎可以原样迁移到时空预测——把文本事件映射成"空间分布漂移 / 区域波动率 / 时空形状模板 / 空间滞后"等时空原语。
- **可借鉴**：modality gap 的形式化诊断方法（半合成对照实验）、"over-attend 到冗余 token"的失效分析、confidence-aware gating 设计。
- 代码：https://github.com/olivia3395/TESS

### 2. Transforming Weather Data from Pixel to Latent Space（WLA）

> 天气/气候是典型的大尺度时空预测场景。本文把天气数据从**像素空间**压到**潜空间**，用 Weather Latent Autoencoder 解耦重建与下游任务，并配 Pressure-Variable Unified Module 统一多个气压-变量子集。把 ERA5 的 244.34 TB 压到 0.43 TB，下游任务在潜空间反而比像素空间更准更锐利。

- **为什么直接相关**：天气预测本身就是长期时空预测的核心场景之一；在**潜空间**做预测是与"多模态外生引导"天然契合的接口——外生信息可以作用在紧凑潜表示上而非原始网格。
- **可借鉴**：潜空间解耦 + 多变量统一表示、ERA5-Latent 数据集可作为长期时空预测的标准化 benchmark 底座。

### 3. ConFlux: Multivariate Time Series in Flux, One Unified Forecast in Confluence

> 通用多变量时序预测基础模型。针对"变量异步演化、时变交互"的结构性矛盾，先用变量重排降低跨变量纠缠，再把相邻变量聚合成 patch 喂入 Vision Transformer 式架构。在 25 个公开数据集的 zero-shot / fine-tune / from-scratch 三种设置下均 SOTA，推理更快、内存更低。

- **为什么直接相关**：多变量时序的"异步 + 异质 + 时变交互"正是时空预测里跨传感器/跨区域的结构。它的"重排降纠缠 + patch 聚合 + 统一 token 表示"是把异质多源信号收敛到统一预测的模板，与"多模态外生信息如何汇入统一时空预测"同构。
- **可借鉴**：变量重排去纠缠、patch 化统一 token 表示、作为预训练基础模型的训练范式。

### 4. Latent Laplace Diffusion for Irregular Multivariate Time Series（LLapDiff）

> 面向**不规则多变量时序长期预测**的生成式框架。把目标建模为低维潜轨迹，实现**全 horizon 一次性生成**而无需逐步在物理时间上积分。用 stochastic port-Hamiltonian 动力学启发的稳定模态参数化，均值演化在**拉普拉斯域**用可学习的共轭复极点表示，可直接在不规则时间戳上求值。long-horizon 预测优于基线，且同模型可在历史时间戳查询做缺失值插补。

- **为什么直接相关**：长期 horizon + 时空（多变量可视为空间维度）+ 不规则观测三者齐备，是"长期时空预测"的生成式骨干候选。拉普拉斯域的共轭极点参数化提供了远距离 rollout 不漂移的稳定机制——这是长期预测的关键瓶颈。
- **可借鉴**：潜轨迹 + 全 horizon 生成避免逐步漂移；port-Hamiltonian 稳定化；gap-aware history summarizer 处理观测稀疏。
- 代码：https://github.com/pixelhero98/LLapDiffusion

### 5. Interpretable Functional Koopman Learning with Non-Markovian Closure for Spatiotemporal Systems（MERLIN）

> Koopman 框架，把时空动力学提升为**学习到的观测泛函**的近线性演化，支持任意分辨率全场重建。理论上发展 PDE 的泛函 Koopman 理论，并用 **Mori–Zwanzig 形式化**补偿有限维线性不变性的损失，加入**非马尔可夫记忆项**提升预测精度。用与离散化无关的函数编码器/解码器处理不规则观测，在超低维潜空间也能稳定长 horizon rollout。

- **为什么直接相关**：明确以"spatiotemporal systems 的长期预测"为目标，且用非马尔可夫记忆处理长程依赖——这与"长期"维度直接命中。非马尔可夫记忆天然是外生/历史信息引导的一种形式。
- **可借鉴**：泛函 Koopman + Mori–Zwanzig 非马尔可夫闭合作为长期时空预测的可解释骨干；任意分辨率编解码处理不规则/稀疏传感。
- 代码：https://github.com/RobinLufdu/MERLIN

## 二、方法可迁移：不直接做时空预测，但关键机制可搬到本方向

这些论文的研究对象不是时空预测，但其中的**引导机制、动力学建模、多模态融合、长期稳定性**等关键拼图，正是本方向缺失或可强化的环节。

### 6. Learning Coupled Continuous-Time Latent Dynamics from Irregular Events（CoCLD）

> 从不规则事件序列学习**耦合的连续时间潜动力学**：个体级状态连续演化，同时受群体级动力学影响。用 Diffusion-based Latent Interpolator + neural ODE 在连续时间潜空间对齐个体与群体动力学。覆盖 next-event 预测、移动轨迹生成、序列行为建模。

- **可迁移点**：个体-群体耦合对应时空预测中"单传感器 vs 区域/全网"的层次结构；连续时间潜空间对齐可直接服务于不规则时空观测。"移动轨迹生成"本身就是时空预测任务。

### 7. Training-Free Bayesian Filtering with Generative Emulators

> 用**扩散模型作为动力学系统的模拟器**实现无训练的贝叶斯滤波，把粒子滤波扩展到高维（含大气动力学）。理论上对非线性动力学/观测精确，解决了经典粒子滤波高维不可扩展的痛点。

- **可迁移点**：时空预测本质是"从含噪观测估计动态场状态"的状态估计问题，贝叶斯滤波是其概率上正确的基础框架。用生成式 emulator 做高维状态估计，可直接用作长期时空预测的**不确定性量化**与**数据同化**主干，外生信息则作为滤波器的观测/先验注入。

### 8. Generative Modeling of Irregular Time Series via SDE-Induced Continuous-Discrete Variational Inference（SDEVI）

> 不规则时序的生成建模，变分推断直接作用在离散观测的联合分布上，同时保证与底层 SDE 连续过程一致。非线性 SDE 诱导的变分后验作为可扩展推断骨干。在医疗、物理、气候、IoT 上做插值/外推/回归/分类。

- **可迁移点**：连续-离散一致性的变分推断，是把"不规则稀疏时空观测"与"连续时空场动力学"统一建模的概率框架；气候/IoT benchmark 与时空预测高度重叠。

### 9. VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs

> 自动驾驶的流式世界模型，在**矢量图**上增量生成 ego-centric lane-agent tiles。组合 motion-aware gated VAE（历史兼容初始化）、edge-gated relational DiT + interval-conditioned MeanFlow（无求解器外推）、ΔSim（物理对齐 NPC 策略），实现实时 1km+ 闭环 rollout。

- **可迁移点**：矢量图上的时空生成是交通时空预测的天然形态；interval-conditioned 条件生成 + 物理对齐约束可作为"外生信息（路网/事件）引导时空生成"的工程模板。

### 10. LASER: Learning Active Sensing for Continuum Field Reconstruction

> 连续物理场的高保真重建，建模为 POMDP。核心是**连续场潜世界模型**捕捉底层物理动力学并提供内在奖励，RL 策略在潜想象空间里做 what-if 传感，按预测潜态条件化地移动传感器到高信息区域。

- **可迁移点**：连续场潜世界模型直接就是时空场的生成式动力学模型；"条件化于预测潜态引导传感"与"外生信息引导预测"在机制上同构——可把"外生信息注入"替换"主动传感动作"。

### 11. Geometric Flow Grounding（GFG）

> 统一框架，强制动态演化严格沿学习到的数据流形切丛进行，用可微 Neural Tangent Projection Layer 把状态表示与切向动力学几何解耦。在稀疏动力系统里减少数值混叠、提升长 horizon 稳定性；投影残差还能零样本检测 deepfake 视频（揭示与预训练世界模型隐流的矛盾）。

- **可迁移点**：流形约束投影作为"长期 rollout 不漂移"的通用算子，可直接嫁到任何时空预测骨干上抑制 off-manifold hallucination；投影残差作为预测可信度的零样本指标。

### 12. NeuronCtrl: Geometry-Aware Safe Closed-Loop Generative Control for Neuronal Microenvironment Dynamics

> 高维**时空场**（不规则 3D 形态上的电生理 + 离子反应扩散）闭环生成式控制。history-conditioned observer 推断潜场，morphology-aware neural operator 预测一步动力学，flow-matching conditional flow 提出受用户偏好条件化的动作，并用 barrier 机制在动作层和场层双重保安全。

- **可迁移点**：虽领域是神经调控，但"不规则几何上的时空场 + 潜态观测器 + 神经算子预测 + 条件流生成 + 安全约束"这套模块化操作级框架，几乎可整套平移到"外生信息引导的时空场预测/控制"。

### 13. Solving Time-Dependent Differential Equations with Physical Dynamical Systems（DS-TS）

> 用物理动力系统机器（DSM）解时间依赖微分方程，兼顾精度与延迟。三创新：兴奋-抑制耦合建模空间交互、状态感知动态非线性、分层时间积分捕获高阶时间依赖。比基线快 ~10³×、能效 ~10⁵×。

- **可迁移点**：若时空预测有 PDE 先验，这种"连续物理演化 + 状态感知非线性 + 分层时间积分"可作为高效长 horizon rollout 的物理先验骨干。

### 14. Adaptive Memory Retention in Dynamic Graphs（LAMP）

> 动态图的快照模型，在动力系统框架内引入自适应、可学习的耗散。用脉冲神经 ODE + 反对称参数化建模保守信息流，加上数据驱动的耗散调控时空信息留存。理论给出稳定性保证。在需要长程依赖的任务上 SOTA。

- **可迁移点**：动态图 = 时空图；自适应耗散解决"长程传播 vs 噪声抑制"的权衡，正是长期时空预测里信息累积失控的痛点。

### 15. HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation

> 时序插补，给每个特征分配**可学习的特征身份**（持久嵌入），从时间共变中端到端学习任意特征依赖，不依赖预定义图拓扑。集成混合时间-特征注意力，5 个数据集 21 项设置全 SOTA。

- **可迁移点**：时空预测常面临缺失值/稀疏观测；"可学习特征身份 + 无拓扑依赖的跨维依赖学习"是把异质多模态特征统一编码并交叉利用的实用方案。

## 三、基础设施：数据、骨干、算子支撑

不直接贡献预测方法，但提供本方向所需的数据集、高效算子或序列骨干。

### 16. OSM+: Billion-Level Open Street Map Dataset for City-wide Experiments

> 十亿级全球路网图数据集，提供空间查询接口。给出 31 城交通预测 benchmark 和 6 城交通政策控制数据，并提供**多模态时空数据与 OSM+ 整合的工具**，用于地理空间基础模型训练。

- **怎么用**：城市交通长期时空预测的标准化大规模数据底座；其"多模态时空数据整合工具"直接服务"多模态外生信息"接入。

### 17. Robust Causal Discovery in Real-World Time Series with Power-Laws

> 利用真实时序的功率谱幂律特性做鲁棒因果发现，在合成与真实有已知因果结构的数据上超越 SOTA。应用覆盖金融、经济、神经科学、气候。

- **怎么用**：长期时空预测里"哪些外生变量真正驱动目标"是核心问题；鲁棒因果发现可做外生信息的选择/加权/先验结构。

### 18. MuonSSM / S³GNN / Efficient Scaling of GNNs via IO-Aware Layers

> MuonSSM：正交化状态空间模型，用动量路径 + Newton-Schulz 变换稳定长 horizon 的 SSM 训练，抑制谱放大、丰富长程记忆。S³GNN：兼顾全局谱混合与局部消息传递的长程图学习，缓解 oversquashing。IO-Aware GNN：给 SpMM / 归约 / 注意力三类图算子写高效 GPU kernel，最高 8.5× 加速、76× 降内存。

- **怎么用**：MuonSSM 作为长期时空序列的稳定序列骨干；S³GNN 作为长程时空图骨干；IO-Aware 算子支撑大规模时空图的高效训练/推理。

### 19. Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation

> 学任意分布到分布的映射，把先验 GMM 经自注意力 + 对数据点的交叉注意力变成后验 GMM。可在先验变化时无需重训，把推断从分钟降到毫秒。覆盖序列推断、量子参数推断、GP 预测后验。

- **怎么用**：外生信息变化时"在线适配先验"的快速近似推断——适合把动态外生信息作为时变先验注入时空预测的贝叶斯框架。

## 四、用得上的"远亲"：思路启发但不构成直接方法

- **HDFlow（分层扩散-流规划做长 horizon 任务）**：分层分解 + 扩散-流混合生成长 horizon 轨迹的思路，启发"长期时空预测可分层 coarse-to-fine"。
- **Robust Causal Discovery**（见上）：也可归为远亲——因果结构作为外生选择的先验。
- **Conditional Diffusion 的引导机制群**（Manifold-Optimal Guidance、Concept Removal Guidance、Control Consistency Losses for Diffusion Bridges）：这些是扩散引导的纯方法论，启发"如何在时空扩散生成里几何正确地注入外生引导方向"。

## 五、缺口与判断

ICML 2026 spotlight 里**没有一篇**把"多模态 + 外生信息引导 + 长期 + 时空预测"四者同时完整命中：最接近的是 TESS（外生文本→时序，但非空间）、MERLIN（长期时空，但无多模态外生）、WLA（天气时空潜空间，但无外生引导）。这恰好说明该方向是一个**有空间、有缺口**的研究点——把 TESS 的外生语义空间机制嫁到 MERLIN/WLA 的长期时空骨架上，用 CoCLD/LLapDiff 的连续时间潜动力学做 rollout，用 LAMP/GFG 抑制长程漂移，用 OSM+ 做数据底座，是一个可行的组合创新路径。

[^src-icml-2026-spotlight-papers]: [[source-icml-2026-spotlight-papers]]
