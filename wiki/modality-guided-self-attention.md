---
title: "Modality-Guided Self-Attention"
type: technique
tags:
  - multimodal-time-series
  - attention-mechanism
  - modality-fusion
  - arxiv-2026
created: 2026-05-03
last_updated: 2026-08-05
source_count: 1
confidence: high
status: active
---

# Modality-Guided Self-Attention

**Modality-Guided Multi-head Self-Attention** 是 [[aurora|Aurora]] 中提出的核心技术，用于将多模态领域知识注入时间序列表示建模[^src-aurora]。

## 动机

传统时间序列自注意力机制仅基于数值时间序列计算注意力权重，无法利用文本或图像中包含的领域特定知识。Aurora 通过模态引导的方式，将多模态领域知识作为注意力计算的引导信号[^src-aurora]。

## 机制

### 前置：token 蒸馏

文本/图像 token 经预训练 Bert/ViT 编码后，先由 `TextDistiller`/`VisionDistiller`（多头交叉注意力，query 为 $K^{\text{text}}$、$K^{\text{image}}$ 个可学习语义聚类质心，且 $K < n$）蒸馏为少量精华 token——关键领域描述往往只值几个词，冗余信息被过滤。蒸馏后的文本表示 $\tilde{X}^{\text{text}}$ 与图像表示 $\tilde{X}^{\text{image}}$ 是后续引导信号的来源[^src-aurora]。

### 跨模态相关桥接（Corr 矩阵）

Aurora 通过 `VisionGuider`/`TextGuider`（交叉注意力）分别捕获时间模态与其他模态的未归一化相关性分数：

$$V_{\text{Attn}} = \text{VisionGuider}(X^{\text{time}}, X^{\text{image}}) \in \mathbb{R}^{n^{\text{time}} \times K^{\text{image}}}$$

$$T_{\text{Attn}} = \text{TextGuider}(X^{\text{time}}, X^{\text{text}}) \in \mathbb{R}^{n^{\text{time}} \times K^{\text{text}}}$$

再以可学习度量 $W \in \mathbb{R}^{K^{\text{image}} \times K^{\text{text}}}$ 调谐语义距离，桥接出**时间模态内部的**相关性：

$$\text{Corr} = V_{\text{Attn}} \cdot W \cdot T_{\text{Attn}}^\top \in \mathbb{R}^{n^{\text{time}} \times n^{\text{time}}}$$

随后注入自注意力打分：

$$S = \frac{QK^\top + \text{Corr}}{\sqrt{d}}, \quad O = \text{Softmax}(S) \cdot V$$

效果是领域知识（"该路段发生事故"）直接调节时间 token 间的注意力权重，让模型聚焦于关键片段而非平均用力[^src-aurora]。最后三路表示经 Cross-Attention Fuser 融合为 $X^{\text{fuse}} = X^{\text{time}} + \tilde{X}^{\text{image}} + \tilde{X}^{\text{text}}$。

与 [[multi-modality-refinement|MoST 的 SNR 自适应模态选择]]不同，Aurora 采用注意力引导的方式——多模态知识不直接替换或筛选时间序列特征，而是以 Corr 矩阵形式作为注意力打分的额外偏置，引导时间表示的建模方向[^src-aurora]。

## 实验证据

附录 C.4 的可视化（Agriculture/Climate/Economy 等）表明：无引导时时间 patch（T1–T4）间相关性近乎均匀；加入模态引导后，与"未来值相关性相似"的 patch 对（如 Agriculture 的 T1↔T2、Climate 的 T3↔T4）被显著聚焦，预测更准。消融上，论文报告去掉该模块（退化为普通 MSA）后 Economy 域 MSE 从 0.033 升至 0.277（约 8.4 倍），是论文报告的消融中变化幅度最大的单点结果[^src-aurora]。

## 与其他模态融合方法的对比

| 方法 | 模型 | 融合策略 |
|------|------|----------|
| **Modality-Guided Attention** | Aurora | 多模态知识引导自注意力计算 |
| SNR-based Modality Selection | [[most|MoST]] | 估计 SNR 后 Gumbel-Sigmoid 门控 |
| Covariate Homogenization | [[unica|UniCA]] | 投影到统一空间后 Pre/Post-Fusion |
| Cross-view Text Fusion | [[mindts|MindTS]] | 内生文本为 query，外生文本为 key/value |
| Adaptive Frequency Fusion | [[vot|VoT]] | 频域分解后自适应加权融合 |

## 相关页面

- [[aurora]] — Aurora 模型
- [[prototype-guided-flow-matching]] — 解码阶段的流匹配技术
- [[multi-modality-refinement]] — MoST 的模态选择技术
- [[covariate-fusion-module]] — UniCA 的协变量融合模块

[^src-aurora]: [[source-aurora]]
