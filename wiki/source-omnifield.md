---
title: "OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning"
type: source-summary
tags:
  - multimodal-spatiotemporal
  - neural-field
  - cross-modal-fusion
  - scientific-ml
created: 2026-07-21
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# OmniField: Conditioned Neural Fields for Robust Multimodal Spatiotemporal Learning

**Venue**: ICLR 2026 | **arXiv**: 2511.02205 | **Authors**: Kevin Valencia (UCLA), Thilina Balasooriya (Columbia), Xihaier Luo, Shinjae Yoo & David Keetae Park (BNL)

- 模型页：[[omnifield|OmniField]] — 框架总览、核心能力与关键结果

## 核心论点

真实科学观测数据面临两个实际挑战：**(1) 数据挑战**——各模态内测量值稀疏、不规则且带噪声；(2) **模态挑战**——不同模态的传感器位置、覆盖密度和噪声特性各异，且可用模态集合随时空变化[^src-omnifield]。现有方案要么依赖预处理（gridding/kriging/imputation）引入平滑偏置和不确定性塌缩，要么假设固定观测算子与共享采样索引导致似然误设（likelihood misspecification）[^src-omnifield]。

OmniField 提出了一种**连续性感知（continuity-aware）框架**，基于条件化神经场（Conditioned Neural Field, CNF），在不做 gridding 或代理预处理的情况下，直接从稀疏、不规则、多模态观测中学习连续时空场，统一处理重建、空间插值、预测和跨模态预测四类任务[^src-omnifield]。

## 方法要点

1. **编码器-处理器-解码器架构**：以 Perceiver IO 和 SCENT 为基础，编码器从输入上下文构建 query-local 置换不变摘要，处理器融合多分辨率坐标编码与上下文摘要形成潜变量场，解码器按模态输出预测[^src-omnifield]。

2. **Gaussian Fourier Features (GFF) + 正弦初始化**：替代固定正弦傅里叶特征，从高斯分布采样频率构建更丰富的频谱表示；同时对可学习 query token 做对数间隔多尺度正弦初始化，稳定训练并改善高频学习——消融中 CIFAR-10 重建提升 ×2.74，气候预测提升 30%[^src-omnifield]。

3. **多模态串扰块（MCT）**：串联各模态 CNF token 并通过全局特征 $z$ 注入跨模态信息，处理器基于自注意力实现跨模态融合[^src-omnifield]。详见 [[multimodal-crosstalk|MCT]]。

4. **迭代跨模态精炼（ICMR）**：以全局特征 $z$ 作为通信桥梁，多轮迭代执行 MCT → 池化更新 $z$ → 再次 MCT，逐步对齐异构模态信号，在噪声鲁棒性实验中显著优于 Mid-Fusion[^src-omnifield]。详见 [[iterative-cross-modal-refinement|ICMR]]。

5. **Fleximodal Fusion**：通过模态存在掩码 $\pi_m$ 对缺失通道零门控，使同一模型适应任意输入子集，优于仅训练时的 ModDrop 策略[^src-omnifield]。详见 [[fleximodal-fusion]]。

## 实验与贡献

- **数据集**：ClimSim-THW（3 模态，21,600 个传感器位置，3.87% 采样率）、EPA-AQS（6 种空气污染物，1987–2017 真实记录）、CIFAR-10、RainNet[^src-omnifield]。
- **8 个基线对比**：UNet, ResNet, FNO, OFormer, CORAL, PROSE-FD, MIA, SCENT[^src-omnifield]。
- **关键结果**：跨基准平均相对误差降低 22.4%；严重模拟传感器噪声下性能接近干净输入水平；EPA-AQS 上随模态数增加（2→4→6）性能单调提升[^src-omnifield]。
- **数据贡献**：提供 ClimSim-LHW（反映真实观测稀疏性）和 ML-ready EPA-AQS 数据集[^src-omnifield]。

## 局限性

计算和内存随 token 数量（及潜容量）增长；解码器仅提供点估计缺乏校准不确定性；跨域泛化（季节变化、传感器重定位）未充分量化；更长预测视界需额外时序结构[^src-omnifield]。

[^src-omnifield]: [[source-omnifield]]
