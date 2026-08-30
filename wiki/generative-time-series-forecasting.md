---
title: "Generative Time Series Forecasting"
type: concept
tags:
  - time-series
  - generative-model
  - probabilistic-forecasting
  - flow-matching
  - diffusion-models
created: 2026-05-03
last_updated: 2026-08-30
source_count: 19
confidence: high
status: active
---

# Generative Time Series Forecasting (生成式时间序列预测)

## 定义

**生成式时间序列预测**是指使用生成模型（如扩散模型、流匹配、归一化流）直接建模未来时间序列的条件概率分布 $p(\mathbf{y} \mid \mathbf{x})$，而非仅输出点估计的预测范式[^src-aurora][^src-simdiff]。

## 与判别式预测的对比

| 维度 | 判别式预测 | 生成式预测 |
|------|-----------|-----------|
| 输出 | 点估计 $\hat{y}$ | 概率分布 $p(y \mid x)$ |
| 不确定性 | 隐式（需额外建模） | 显式（分布自然包含） |
| 典型方法 | MSE/MAE 回归 | Diffusion / Flow Matching |
| 代表模型 | PatchTST, iTransformer | SimDiff, Aurora |

## 现有方法

### 扩散模型方法

**[[timegrad|TimeGrad]]** (ICML 2021) 是首个将扩散模型引入时间序列概率预测的开创性工作，将 RNN 自回归编码与 DDPM 条件扩散结合，在 6 个数据集的 CRPS 评估中全面领先 14 种基线[^src-timegrad]。其"RNN 负责时间记忆 + 扩散负责分布建模"的二段式架构为后续 CSDI、DiffSTG、SpecSTG 等扩散+时序工作奠定了范式基础。

在插补方向，**[[csdi|CSDI]]** (NeurIPS 2021) 首次将条件扩散模型显式用于时间序列缺失值插补，其自监督训练策略和双轴 Transformer 注意力设计成为后续扩散插补工作的标准范式，在预测任务上也展现了与 TimeGrad 相当的竞争力。

**[[tsdiff|TSDiff]]** (NeurIPS 2023, *Predict, Refine, Synthesize*) 将路线从任务专用条件训练转为**无条件** DiffWave+S4 扩散，并用 [[observation-self-guidance|observation self-guidance]]（MSE / 分位数）在推理期条件化到任意观测子集；同一模型还支持基预测精炼与合成数据生成，并定义 [[linear-predictive-score|LPS]] 评测合成样本的下游预测质量[^src-prs]。在 8 个 GluonTS 单变量基准上，TSDiff-Q 与 CSDI/条件扩散及 [[deepar|DeepAR]] 等竞争，且可在未知缺失模式下复用同一检查点[^src-prs]。后续 [[tsflow|TSFlow]] 将“无条件训练 → 推理条件化”推进到 Flow Matching + GP 先验。

**[[simdiff|SimDiff]]** (AAAI 2026) 是首个纯端到端扩散模型用于时间序列点预测，使用 DDPM 框架并通过 Median-of-Means 将概率样本聚合为点估计[^src-simdiff]。SimDiff 仅支持单模态数值输入。

**[[specstg|SpecSTG]]** (arXiv 2024) 是首个在图谱域执行扩散过程的概率时空图预测框架。核心创新是将扩散过程转移到图傅里叶域——生成未来时间序列的傅里叶表示而非原始序列，使得空间依赖关系自然融入扩散基中。通过 [[fast-spectral-graph-convolution|Fast Spectral GC]] 将图卷积复杂度从 $O(N^2)$ 降至 $O(N)$，训练速度达 [[d3vae|GCRDD]] 的 3.33 倍，点估计最高提升 8%[^src-2401-08119-specstg]。

**[[ustd|USTD]]** (SIGSPATIAL 2024) 首次统一时空预测和插值到扩散框架。核心贡献：预训练编码器（GWNet backbone + 75% masking + 80% graph sampling）与任务特定 gated attention denoisers（TGA/SGA）的解耦两阶段训练，打破了此前"diffusion STG 打不过 deterministic baseline"的共识。4 数据集 × 16 baselines，CRPS 最高降低 12%，推理比 CSDI 快 ~47%。

**[[dyffusion|DYffusion]]** (NeurIPS 2023) 是首个将扩散模型的退化和重建过程完全替换为时序插值和预测的框架，不使用高斯噪声[^src-dyffusion]。通过两阶段训练（插值器 $\mathcal{I}_\phi$ → 冻结 → 预测器 $F_\theta$）和 Cold Sampling 推理，实现了常数级训练内存和 $<50$ 步扩散推理。在 SST、Navier-Stokes 和 Spring Mesh 物理系统上全面超越 Dropout/DDPM/MCVD 基线[^src-dyffusion]。

**[[middir|MiDDiR]]** (ICLR 2026 under review) 提出混合通道依赖扩散模型，核心创新包括 CD 编码 + CI 去噪的混合策略和推理时检索引导[^src-middir]。CD 编码器捕获跨通道信息后，CI 去噪降低联合分布的建模复杂度；检索引导通过训练集相似样本分析性偏置得分估计，改善低密度区域采样。7 数据集 SOTA，CRPS 超越 NsDiff 约 21.9%，参数效率对通道数不敏感。[^src-middir]

### 流匹配方法

**[[sundial|Sundial]]** (ICML 2025) 是首个将 Flow Matching 应用于时间序列基础模型的工作，提出了 TimeFlow Loss — 在连续值域中学习的生成式训练目标[^src-sundial]。Sundial 系列（32M/128M/444M）在 1 万亿时间点 (TimeBench) 上预训练，在 TSLib、GIFT-Eval (MASE #1) 和 FEV Leaderboard 上取得 SOTA 零样本性能。使用连续 patch tokenization 避免 Chronos 的离散 tokenization 问题，生成式建模对抗 mode collapse，支持测试时校准。CPU 推理 ~1s。详见 [[sundial]]、[[timeflow-loss]]、[[timebench]]。

**[[flowts|FlowTS]]** (arXiv 2025) 是首个将 rectified flow 用于时间序列生成的 ODE 模型，通过直线概率路径替代迭代式扩散，30 步采样即 SOTA[^src-flowts]。无条件模型可无缝适应条件预测，Context-FID Stocks 0.019 (vs 此前最优 0.067)，Solar 预测 MSE 213 降低 43.2%[^src-flowts]。详见 [[rectified-flow-for-time-series|Rectified Flow for TS]]、[[adaptive-sampling-flow-matching|Adaptive Sampling]]。

**[[tsflow|TSFlow]]** (ICLR 2025) 是首个将条件流匹配 (CFM) 应用于时间序列预测的模型，由 TU Munich 提出[^src-tsflow]。核心创新包括：使用高斯过程先验 (SE/OU/PE 三种核函数) 替代各向同性高斯先验以对齐时序结构、通过 mini-batch 最优传输耦合拉直概率路径、以及提出条件先验采样 (CPS) + 引导实现无条件模型的条件化预测。在 8 个真实数据集上 SOTA（6/8 CRPS 最优），以更少 NFE 全面超越扩散基线 [[csdi|CSDI]]、SSSD、[[tsdiff|TSDiff]] 和 Biloš et al. (2023)[^src-tsflow]。架构使用 DiffWave+S4（3 个残差块，~176k 参数），Euler ODE 32 步采样。

**[[freqflow-ts|FrèqFlow/SpectFlow]]** (NeurIPS 2025) 首次将条件流匹配引入频域进行确定性 MTS 预测。通过复值线性层在频域中插值频谱，配合流匹配头进行残差学习，仅 89k 参数即达到 SOTA。采用 ODE 单次确定性采样，推理速度远超扩散方法[^src-2511-16426]。

**[[aurora|Aurora]]** (ICLR 2026) 提出 Prototype-Guided Flow Matching，使用多模态领域知识生成条件和原型来引导流匹配过程，实现生成式概率预测[^src-aurora]。与 Sundial/TSFlow 从高斯噪声出发不同，Aurora 的生成起点是"未来原型 + 噪声"（$y^{(0)}=\tilde{P}_i+\epsilon_i$）——原型由 1000 个周期/趋势基向量按文本/图像知识检索加权合成，论文称该起点已包含未来周期与趋势的雏形（rudiments of periodicity and trend），流匹配在此基础上生成具体数值。Aurora 支持多模态输入（文本、图像、数值）和零样本推理。

### 归一化流方法

**[[manf|MANF]]**（arXiv:2205.07493）将 [[multi-scale-attention|多尺度注意力]] 编码器与条件 [[normalizing-flow|RealNVP]] 结合，以**非自回归**方式生成未来窗口联合分布：预测窗观测不回馈模型，解码器各层条件驱动堆叠 affine coupling，从而避免 AR 流的误差累积并保持时间维并行[^src-maf]。在 Exchange/Solar/Electricity/Traffic/Taxi/Wikipedia 上相对 LSTM-MAF、Transformer-MAF、NKF 等取得文中报告的 CRPS-sum/MSE SOTA，并在加倍预测长度与缺失噪声压力下更稳[^src-maf]。它代表离散归一化流在多变量概率预测中的早期 NAR 路线，与后续 Flow Matching（TSFlow/Sundial）及并行的 AR 扩散路线（[[timegrad|TimeGrad]] 等）形成对照。

### 自回归参数化似然（AR-RNN）

**[[deepar|DeepAR]]** (arXiv:1704.04110) 用全局共享 LSTM 自回归建模大量相关序列：每步输出高斯或负二项等似然参数，祖先采样 Monte Carlo 轨迹以得到任意跨度分位数；并用序列尺度因子缩放与按销量速度加权采样应对幂律量级差异。它是工业概率预测中长期对照的 AR + 参数化输出基线，也为后续 [[timegrad|TimeGrad]]（扩散头）与 [[deepstate|DeepState]]（非目标输入 SSM）提供了对照起点。[^src-deepar]

### 状态空间 / 深度 SSM 方法

**[[deepstate|DeepState]]** (NeurIPS 2018) 用全局共享 LSTM 从协变量映射出每条序列的线性高斯 [[kalman-filter|SSM]] 参数，以 Kalman 滤波计算边际似然与多步预测后验；目标值不直接作网络输入，从而在可解释季节/趋势结构、小样本效率与跨序列联合学习之间折中，并对照同期 [[deepar|DeepAR]] 与经典 ETS/ARIMA[^src-deepstate]。它是 [[deep-state-space-model|深度状态空间模型]] 在概率预测中的早期工业代表，也是后续 [[k2vae|K²VAE]]（Koopman 线性化 + 神经 Kalman）路线的前驱对照。

### VAE 方法

**[[k2vae|K²VAE]]** (ICML 2025 Spotlight) 是 VAE 路线 + 一步生成的代表，把概率预测重构为在 Koopman 测量空间中对线性动力系统的过程不确定性建模：KoopmanNet 线性化 + KalmanNet 精炼并输出协方差作为变分后验。短期 CRPS 较 CSDI 降低 7.3%，长期 CRPS 较 PatchTST 提升 20.9%，且因 VAE 一步生成而显存最低、推理最快——克服了扩散/流模型在长期预测下崩溃且低效的问题[^src-k2vae]。

### 一致性模型方法

**[[swift|Swift]]** (arXiv 2025) 首次将 [[autoregressive-consistency-models|自回归一致性模型]] 应用于天气时间序列预测，单步 NFE=1 取代扩散模型的 20–40 NFE，实现 39× 推理加速[^src-swift]。通过 [[crps-autoregressive-finetuning|CRPS 自回归微调]] 在多步 rollout 上直接优化集合校准度，实现 75 天稳定预报，集合技能与 IFS ENS 竞争[^src-swift]。

### 自回归生成式方法（VAR）

**[[climatear|ClimateAR]]** (ICML 2026) 是首个将视觉自回归（VAR）引入概率气候预测的模型。通过 VQ + 分段码本对齐 CMIP6-ERA5 跨域异构数据，以混合尺度条件控制（intra-scale mix token + hybrid-scale prompt）在自回归 token 预测中同时捕获尺度内与跨尺度气候交互。在 1–10 月 lead time 上平均 ACC 提升 37.56% vs Pangu/[[graphcast|GraphCast]]/ClimaX，ENSO 概率预测校准优良[^src-climatear]。

### 方法对比

| **ClimateAR** | **VAR (离散 Token 预测)** | **多变量网格数据** | **✓ (零样本 CMIP6→ERA5)** | **概率分布 (集合)** | **多尺度 VQ Token + 混合条件** |
| SpecSTG | Diffusion (谱域) | 仅数值 | ✗ | 概率分布 + 点估计 | **谱域** |
| **FrèqFlow** | **Flow Matching (频域)** | **仅数值** | **✗** | **点估计（确定性）** | **频域** |
| Aurora | Flow Matching (OT) | 文本 + 图像 + 数值 | ✓ | 概率分布 | 原始域 |
| DYffusion | Diffusion (非高斯) | 仅数值 | ✗ | 概率分布 | 原始域 |
| **MiDDiR** | **Diffusion (DDPM)** | **仅数值** | **✗** | **概率分布** | **CI 去噪 + CD 编码** |
| **Sundial** | **Flow Matching (OT)** | **仅数值** | **✓** | **概率分布** | **原始域 + Patch Token + TimeFlow** |
| **FlowTS** | **Rectified Flow (ODE)** | **仅数值** | **✗** | **概率分布** | **原始域 + Trend-Season + RoPE** |
| **TSDiff** | **Diffusion (DDPM, 无条件)** | **仅数值** | **✗** | **概率分布** | **原始域 + observation self-guidance** |
| **TSFlow** | **Flow Matching (OT)** | **仅数值** | **✗** | **概率分布** | **原始域 + GP 先验** |
| **CoGenCast** | **Flow Matching (平均速度)** | **文本 + 数值** | **✓** | **概率分布 (一步)** | **LLM Encoder-Decoder + 平均速度 JVP** |
| **Swift** | **Consistency Model (TrigFlow)** | **仅数值 + 静态强迫** | **✗** | **概率分布 (NFE=1)** | **原始域 + CRPS 微调** |
| **MANF** | **Normalizing Flow (RealNVP)** | **仅数值** | **✗** | **概率分布** | **原始域 + 多尺度注意力 NAR** |
| **DeepAR** | **AR-RNN + 参数化似然** | **数值协变量** | **✗** | **概率分布 (MC 轨迹)** | **全局共享 LSTM + 高斯/负二项 + 尺度缩放** |
| **DeepState** | **Linear SSM + RNN params** | **数值协变量** | **✗** | **概率分布** | **Kalman 解析 + 季节潜状态** |

## 优势

1. **不确定性量化**：生成式方法自然输出预测分布，无需额外的不确定性建模
2. **多模态条件化**：Flow Matching 和扩散模型天然支持条件生成，便于融入多模态信息[^src-aurora]
3. **灵活采样**：可从预测分布中采样多个实现，支持风险分析和决策

## 挑战

1. **计算成本**：生成式方法通常需要多步采样（扩散模型）或 ODE 求解（流匹配）
2. **训练稳定性**：扩散/流匹配训练比判别式回归更复杂
3. **评估指标**：概率预测需要 CRPS、NLL 等分布级指标，而非简单的 MSE/MAE
4. **长程分布预测**：[[probts|ProbTS]] 表明既有概率方法（AR 扩散/流与 NAR 扩散）在长 horizon 上 CRPS 显著恶化——AR 误差累积，NAR 显存与学习效率受限；同时高 [[non-gaussianity|非高斯性]] 场景仍需要比预定义分布头更强的表达力[^src-probts]

## 相关工作

- [[s2dbm]] — S²DBM，布朗桥扩散桥模型，s=0 时退化为无噪声确定性生成器以做点对点预测、s=1 时做概率预测（arXiv 2024）[^src-s2dbm]
- [[deepstate]] — DeepState，RNN 参数化线性 SSM + Kalman（NeurIPS 2018）[^src-deepstate]
- [[deep-state-space-model]] — 深度状态空间模型概念

## 相关页面

- [[aurora]] — 流匹配生成式预测模型
- [[specstg]] — 谱域扩散时空图预测模型
- [[simdiff]] — 扩散式生成预测模型
- [[freqflow-ts|FrèqFlow/SpectFlow]] — 频域流匹配确定性预测（NeurIPS 2025）
- [[ustd]] — USTD，解耦预训练的统一时空扩散预测与插值框架（SIGSPATIAL 2024）
- [[climatear]] — ClimateAR，VAR 自回归概率气候预测（ICML 2026）
- [[mixed-scale-conditioning]] — 混合尺度条件控制
- [[dyffusion]] — DYffusion，动力学信息扩散模型（NeurIPS 2023）
- [[dits|DiTS]] — DiTS，MM-DiT 双流架构 + Rectified Flow 用于协变量感知概率预测 (arXiv 2026)
- [[flow-matching-forecasting]] — Flow Matching 在时间序列预测中的应用范式
- [[middir]] — MiDDiR，混合通道依赖扩散 + 检索引导（ICLR 2026 under review）
- [[mixed-channel-dependency]] — 混合通道依赖策略
- [[retrieval-guidance]] — 扩散采样的检索引导技术
- [[tsflow]] — TSFlow，首个 CFM 时间序列模型，GP 先验 + OT 路径 (ICLR 2025)
- [[tsdiff]] — TSDiff，无条件时序扩散 + observation self-guidance (NeurIPS 2023)
- [[observation-self-guidance]] — 观测自引导
- [[linear-predictive-score]] — LPS 合成样本指标
- [[flowts]] — FlowTS，首个 rectified flow TS 生成模型，30 步 SOTA (arXiv 2025)
- [[rectified-flow-for-time-series]] — Rectified Flow 在 TS 生成中的应用范式
- [[flow-matching]] — Flow Matching 理论基础
- [[diffusion-model]] — 扩散模型理论基础
- [[multimodal-time-series-forecasting]] — 多模态时间序列预测
- [[gaussian-process-prior-flow-matching]] — GP 先验在流匹配中的应用
- [[sundial]] — Sundial，首个 FM TS 基础模型系列，TimeFlow Loss + TimeBench (ICML 2025)
- [[timeflow-loss]] — TimeFlow Loss，原生生成式训练目标
- [[timebench]] — TimeBench，万亿级时序预训练数据集
- [[cogencast]] — CoGenCast，首个混合 LLM + FM 编码器-解码器预测模型，一步生成 (ICML 2026)
- [[hybrid-llm-flow-matching-forecasting]] — 混合 LLM-流匹配预测范式
- [[one-step-flow-generation]] — 一步流生成，NFE=1 的高效推理技术
- [[average-velocity-modeling]] — 平均速度建模，JVP 修正的流匹配训练技术
- [[swift]] — Swift，首个自回归一致性模型用于天气预测，NFE=1 (arXiv 2025)
- [[autoregressive-consistency-models]] — 自回归一致性模型概念
- [[crps-autoregressive-finetuning]] — CRPS 自回归微调技术
- [[probts|ProbTS]] — 点+分布、短+长 horizon 统一基准与 AR/NAR 分析
- [[ar-vs-nar-decoding]] — 多步解码方案权衡
- [[non-gaussianity]] — 窗口分布复杂度指标
- [[manf]] — MANF，多尺度注意力 + 条件 RealNVP（NAR）
- [[multi-scale-attention]] — 多尺度窗口注意力
- [[normalizing-flow]] — 离散归一化流
- [[deepar]] — DeepAR，全局共享 AR-RNN 概率预测基线


[^src-aurora]: [[source-aurora]]
[^src-simdiff]: [[source-simdiff]]
[^src-timegrad]: [[source-timegrad]]
[^src-2401-08119-specstg]: [[source-2401-08119-specstg]]
[^src-2511-16426]: [[source-2511-16426]]
[^src-dyffusion]: [[source-dyffusion]]
[^src-middir]: [[source-middir]]
[^src-flowts]: [[source-flowts]]
[^src-tsflow]: [[source-tsflow]]
[^src-sundial]: [[source-sundial]]
[^src-swift]: [[source-swift]]
[^src-k2vae]: [[source-k2vae]]
[^src-s2dbm]: [[source-s2dbm]]
[^src-probts]: [[source-probts]]
[^src-prs]: [[source-prs]]
[^src-maf]: [[source-maf]]
[^src-deepstate]: [[source-deepstate]]
[^src-deepar]: [[source-deepar]]
[^src-climatear]: [[source-climatear]]
