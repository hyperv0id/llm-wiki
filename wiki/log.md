---
title: Log
type: concept
created: 2026-04-26
last_updated: 2026-05-11
tags:
  - meta
---

# Wiki Log

Chronological record of all wiki activity.

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
- **修复**：补充 `source-timesnet.md` 中缺失的 `[^src-timesnet]` 脚注定义
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
