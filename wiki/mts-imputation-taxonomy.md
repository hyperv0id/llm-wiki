---
title: "MTSI 深度学习分类框架（mts-imputation-taxonomy）"
type: concept
tags:
  - taxonomy
  - data-imputation
  - time-series
  - survey
  - classification-framework
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# MTSI 深度学习分类框架

Wang & Du 等人的 MTSI 综述（arXiv:2402.04059v3）提出一个双视角分类框架，用于组织深度多元时间序列插补（MTSI）方法：**插补不确定性**（imputation uncertainty）× **神经网络架构**（neural network architecture）[^src-mts-imputation-survey]。综述作者自称这是首个对深度 MTSI 方法的系统性综述[^src-mts-imputation-survey]。

## 视角一：插补不确定性（预测式 vs 生成式）

综述以"方法能否对同一缺失位置给出反映不确定性的多样化估计"为界划分两大类[^src-mts-imputation-survey]：

- **预测式（predictive）**：对同一缺失分量一致地输出确定值，不刻画插补不确定性；学习目标是观测位置上的重构损失（绝对/平方误差，综述式 2）[^src-mts-imputation-survey]。
- **生成式（generative）**：从观测数据学习概率分布，对缺失位置采样生成略不同的值，从而量化插补不确定性；学习目标是观测数据的对数似然（综述式 3）[^src-mts-imputation-survey]。

## 视角二：网络架构家族

综述将大模型单列为第三类，理由是其应用策略与一般神经网络算法差异较大[^src-mts-imputation-survey]：

| 综述类别 | 架构家族 | 综述列出的代表（节选） |
|---------|---------|----------------------|
| 预测式 | RNN-based | GRU-D、M-RNN、BRITS |
| 预测式 | CNN-based | TimesNet |
| 预测式 | GNN-based | [[grin\|GRIN]]、SPIN |
| 预测式 | Attention-based | SAITS、DeepMVI、[[imputeformer\|ImputeFormer]] |
| 生成式 | VAE-based | GP-VAE（综述称首个 VAE 插补方法）、V-RIN、supnotMIWAE |
| 生成式 | GAN-based | GRUI-GAN（综述称首个 GAN 插补方法）、E2GAN、NAOMI、SSGAN |
| 生成式 | Diffusion-based | [[csdi\|CSDI]]、SSSD、CSBI、MIDM、[[pristi\|PriSTI]]、SPD、[[sadi\|SADI]]、FGTI、MTSCI |
| 大模型 | Foundation model | MOMENT、Timer、TimeMixer++、[[nuwats\|NuwaTS]] |
| 大模型 | LLM-based | GPT4TS、LLM-TS Integrator |

综述对各家族的利弊归纳（综述认为）：RNN 擅长序列信息但受串行处理与记忆约束，长序列可扩展性差；CNN 核尺寸与工作机制限制其作为时序骨干的表现；attention 方法因长程依赖与并行能力通常优于 RNN/CNN 方法；GNN 对时空动态理解更深但计算复杂度更高。VAE 概率建模显式但生成容量有限；GAN 保真度高但训练不稳定（梯度消失等）；扩散模型表达能力强但计算开销大，且缺失与观测部分之间存在边界一致性问题（综述引 RePaint）[^src-mts-imputation-survey]。

## 附带维度：缺失机制标注

综述 Table 1 还为每个方法标注其缺失机制假设（Rubin 分类）：33 个方法中绝大多数标注 MCAR，GRIN/SPIN/GP-VAE/V-RIN 标注 MCAR/MAR，supnotMIWAE 标注 MNAR，SADI 标注 MCAR/MAR/MNAR[^src-mts-imputation-survey]。该标注为综述作者的二手归类，与原论文自述可能不完全一致，引用时应保持"综述 Table 1 标注"的归因口径（详见 [[missing-not-at-random]]）。

## 本 wiki 方法在框架中的位置

以下归类均来自综述 Table 1 或正文（综述口径）[^src-mts-imputation-survey]：

- [[csdi|CSDI]]：生成式-扩散类；综述称其为首个专门为 MTSI 设计的扩散模型，并指出其双 Transformer 去噪网络对变量数与序列长度呈二次复杂度（该局限也是 SSSD 用结构化状态空间模型替换 Transformer 的动机）。
- [[pristi|PriSTI]]：生成式-扩散类（架构标注 Diffusion+Attention+GNN+CNN）；综述将其条件机制概括为"以时空依赖作为条件信息，用条件特征计算的时空注意力权重引导去噪网络"。
- [[grin|GRIN]]：预测式-GNN 类；综述称其为首个基于图的循环 MTSI 架构，并提到 SPIN 通过稀疏时空注意力缓解 GRIN 的误差传播。
- [[imputeformer|ImputeFormer]]：预测式-Attention 类；注意综述对其的转述未涉及其低秩归纳偏置核心，细节以原论文口径为准。
- [[sadi|SADI]]：生成式-扩散类；综述概述其为利用自注意力捕获病患间相似性的相似度感知扩散模型。
- [[nuwats|NuwaTS]]：大模型-PFM 类；综述将其概括为"复用预训练语言模型做时序插补，用专用嵌入与对比学习处理跨域缺失模式"。
- [[timesnet|TimesNet]]：预测式-CNN 类；综述归因于其 FFT 将 1D 序列重组为 2D 格式以适配 CNN 处理。

综述发表于 2024 年前后，未覆盖其后的流匹配插补路线（如 [[loft|LOFT]]、[[giflow|GiFlow]]）；按其不确定性视角，这些方法属于生成式一类，但这是 wiki 的分析性外推，非综述原文论断。后继论文引用情况：[[fence|FENCE]]（AAAI 2026）参考文献含该综述（arXiv:2402.04059，raw PDF 已核实）[^src-fence]；[[loft|LOFT]]（KDD 2026）是否引用该综述未在仓库内核实（raw/ 无 LOFT PDF）。

## 与其他页面

- 各方法页面的原文口径（架构细节、实验数字）保持不变，本页只记录综述的分类归类。
- 综述"impute-then-predict vs encode-and-predict"的下游集成讨论与 [[two-stage-imputation]]（模型内部双阶段精炼）是不同概念，注意区分。
- 工具箱与基准见 [[pypots]]。

[^src-mts-imputation-survey]: [[source-mts-imputation-survey]]
[^src-fence]: [[source-fence]]
