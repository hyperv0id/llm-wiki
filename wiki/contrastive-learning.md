---
title: "Contrastive Learning"
type: technique
tags:
  - representation-learning
  - self-supervised
  - multimodal-alignment
created: 2026-05-03
last_updated: 2026-10-13
source_count: 10
confidence: medium
status: active
---

# Contrastive Learning

对比学习（contrastive learning）是一种无显式标签的表征学习范式，核心机制是通过编码器拉近正样本对（positive pairs）的嵌入表示，同时推远负样本对（negative pairs）的嵌入表示[^src-multimodal-ts-anomaly-detection]。

## In Multimodal Time Series

在多模态时间序列（TS）场景下，对比学习用于在共享嵌入空间中对齐时序与文本表征：以同时间段的时序-文本对作为正样本对，跨时间段样本作为负样本对[^src-multimodal-ts-anomaly-detection]。

### Trimodal limits (TS · vision · language)

[[ts-vl-alignment|Yashwante & Yu (2026)]] 探讨了跨时间序列、可视化图表与文本三模态的 CLIP 式事后（post-hoc）对比对齐机制，采用冻结预训练编码器配合共享投影头。未经耦合的独立预训练表征空间接近正交；经过 InfoNCE 对齐后呈现不对称性（TS–IMG 强于 TS–TXT），且性能随规模扩展改善不均，在文本信息密度增加时趋于饱和。实证表明单靠对比投影无法保证数值时序获得细粒度的多模态对齐收敛[^src-ts-vl-alignment]。

### Cross-modal misalignment under MMCL

[[cross-modal-misalignment|Cai, Liu et al. (NeurIPS 2025)]] 形式化了图文数据生成中的选择偏差（selection bias，语义遗漏）与扰动偏差（perturbation bias，语义失真）。在多模态对比学习（MMCL）的渐进对齐-熵目标下，编码器仅能块识别（block-identify）无偏共享语义子集，模态特异噪声与缺失/扰动因素被滤除。跨模态非对齐本质上构成语义过滤器：在全覆盖预训练中需予缓解，但在与环境敏感因素重合的分布外（OOD）泛化场景中可直接利用[^src-cross-modal-misalignment]。

### InfoNCE Loss

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_k \exp(\text{sim}(z_i, z_k) / \tau)}$$

## 对比方法的两个代价与替代路线

LeCun（2022）在立场论文中提出对比训练的两项核心局限，并将 [[jepa|JEPA]] 的非对比训练作为替代方案[^src-autonomous-machine-intelligence]：

1. **对比样本来源依赖专门设计**：必须人为设计 $\hat{y}$ 的生成或采样策略。该分类谱系涵盖广泛：通过蒙特卡洛/MCMC/对比散度生成 $\hat{y}$ 的概率模型、通过生成器产生 $\hat{y}$ 的 GAN，以及通过损坏 y 获得负样本的去噪自编码器与掩码自编码器（[[mae|MAE]]、[[videomae]]），均属于广义对比范式。
2. **高维空间的维度灾难**：当 y 处于高维空间且基于能量的模型（EBM）具备充分表达能力时，为确保无数据区域的能量值严格高于数据流形，最坏情况下所需对比样本数量随表征维度呈指数增长。

由此得到收缩低能量区域体积的正则化（非对比）路线。VICReg（Bardes et al., ICLR 2022）通过在嵌入空间约束 batch 内变量分量相互去相关（contrastive over components），替代了传统对比方法要求样本向量相互排斥（contrastive over vectors）的做法，避免了对负样本采样机制的依赖（详见 [[non-contrastive-ssl]]）[^src-autonomous-machine-intelligence]。VICReg 联合优化三项损失：均方欧氏距离不变性项、维持各维度标准差高于阈值（$\gamma=1$）以防止常数塌缩的方差项，以及驱使协方差矩阵非对角元素归零以消除冗余的协方差项；其分支间无需参数共享或归一化技巧，在 ImageNet 线性评估（ResNet-50，1000 epochs）上达到 73.2% Top-1 准确率，且在 batch size 从 128 至 4096 范围内（100 epochs 下准确率处于 67.3%–68.6%）保持稳健，消除了对比方法对超大 batch 的依赖[^src-vicreg]。

同谱系的 [[jepa]] 在存在固定干扰噪声时会出现性能劣化，原因在于非对比目标本身并不阻止模型过度聚焦时间变化最慢的特征[^src-jepa-focus-on-slow-features]。

## In Imputation: Mask-Invariant Representations

[[nuwats|NuwaTS]]（arXiv 2024）将对比学习应用于缺失模式（missing patterns）鲁棒性建模，而非模态对齐[^src-nuwats]。模型对每个输入样本生成两个具有不同掩码率（mask ratios）的视图并送入 PLM；同一 patch 在两种掩码下的表征构成正样本对，其余 patch 及其他序列的表征作为负样本。NuwaTS 采用基于双线性内积 $q^\top W k_+$（$W$ 为可学习参数）的 InfoNCE 损失，并与 MSE 重构损失联合优化[^src-nuwats]。该设计构建了掩码不变 patch 嵌入（mask-invariant patch embeddings），使模型在 patch 缺失程度变化时保持表征一致性；消融实验表明该机制是实现跨域零样本插补的必要条件[^src-nuwats]。

## Dual-level Hard Negative Mining in TRACE

[[trace|TRACE]]（NeurIPS 2025）在跨模态对齐中提出 [[dual-level-hard-negative-mining|双级硬负采样]]，分别在 sample-level（[CLS] 嵌入对齐样本文本）和 channel-level（[[channel-identity-token|CIT]] 嵌入对齐通道文本）两个粒度动态挖掘硬负样本。在 channel-level 引入 intra-instance（同实例其他通道）与 inter-instance（跨实例同通道）两类干扰项（distractor），促使模型区分形态相似但语义不同的通道模式[^src-trace-neurips2025]。针对时序文本常指代特定物理变量（如温度峰值、风速）导致单一全局嵌入难以精准对齐的局限，该方法替代了传统 CLIP 式仅在 sample-level 进行随机负采样的机制[^src-trace-neurips2025]。

## In Spatio-Temporal Graph Forecasting

在时空图（STG）连续数值回归任务中，[[stgcl|STGCL]]（SIGSPATIAL 2022）对照实验显示：直接采用两阶段对比预训练会使 PEMS-04 数据集上 GWN 的 MAE 从 19.33 恶化至 20.22/20.67，原因为对比学习优化的 uniformity 特征更有利于分类而非回归。论文改用端到端联合学习，将对比目标作为图级正则项，在四个骨干网络上取得一致提升；同时引入基于 time-of-day 差值阈值 $r_f$（实验显示 30~60 分钟最优）过滤时间邻近的假硬负样本[^src-stgcl]。

## Applications

- [[fine-grained-time-text-semantic-alignment]] — MindTS 基于 patch 级的时序-文本对比对齐
- [[endogenous-text-alignment]] — VoT 基于解耦趋势/季节项的对比学习
- [[multi-level-alignment]] — VoT 基于多级对比损失的多粒度对齐框架
- [[nuwats]] — NuwaTS 跨缺失模式的掩码不变 patch 表征
- [[mtsci]] — MTSCI 基于互补掩码双视图对比损失的插补一致性机制（[[imputation-consistency]]）[^src-mtsci]
- [[ts-vl-alignment]] — 时序、视觉与语言跨模态事后对比对齐的收敛界限
- [[cross-modal-misalignment]] — 选择与扰动偏差下多模态对比学习（MMCL）对非对齐样本的表征保留边界
- [[trace]] — TRACE 用于跨模态时序-文本检索的双级硬负采样
- [[stgcl]] — STGCL 时空图对比学习实证基准（端到端联合训练与时空负样本过滤）

## Related

- [[mutual-information]] — 模态间交互的替代信息论目标
- [[content-condenser-reconstruction]] — 采用重构替代对比学习的互补路径
- [[non-contrastive-ssl]] — 正则化自监督路线与 VICReg，对比学习的高维替代方案
- [[vicreg]] — 防塌缩正则的具体实现与实验证据
- [[jepa]] — 非生成式表征空间预测架构及其在固定噪声下的失效实证
- [[fine-grained-traffic-prediction]] — 细粒度交通预测中用于高效图划分的对比聚类
- [[ssdl]] — ST-SSDL 使用 prototype triplet loss 实现潜在空间离散化，作为自监督偏差学习的基础
- [[source-cross-modal-misalignment]] — MMCL 中非对齐数据价值的理论形式化（NeurIPS 2025）

[^src-st-ssdl]: [[source-st-ssdl]]

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
[^src-minitraffic]: [[source-minitraffic]]
[^src-nuwats]: [[source-nuwats]]
[^src-ts-vl-alignment]: [[source-ts-vl-alignment]]
[^src-cross-modal-misalignment]: [[source-cross-modal-misalignment]]
[^src-trace-neurips2025]: [[source-trace-neurips2025]]
[^src-stgcl]: [[source-stgcl]]
[^src-mtsci]: [[source-mtsci]]
[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
[^src-jepa-focus-on-slow-features]: [[source-jepa-focus-on-slow-features]]
[^src-vicreg]: [[source-vicreg-variance-invariance-covariance-regularization]]
