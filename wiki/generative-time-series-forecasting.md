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
last_updated: 2026-05-13
source_count: 4
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

**[[simdiff|SimDiff]]** (AAAI 2026) 是首个纯端到端扩散模型用于时间序列点预测，使用 DDPM 框架并通过 Median-of-Means 将概率样本聚合为点估计[^src-simdiff]。SimDiff 仅支持单模态数值输入。

**[[specstg|SpecSTG]]** (arXiv 2024) 是首个在图谱域执行扩散过程的概率时空图预测框架。核心创新是将扩散过程转移到图傅里叶域——生成未来时间序列的傅里叶表示而非原始序列，使得空间依赖关系自然融入扩散基中。通过 [[fast-spectral-graph-convolution|Fast Spectral GC]] 将图卷积复杂度从 $O(N^2)$ 降至 $O(N)$，训练速度达 GCRDD 的 3.33 倍，点估计最高提升 8%[^src-2401-08119-specstg]。

### 流匹配方法

**[[freqflow-ts|FrèqFlow/SpectFlow]]** (NeurIPS 2025) 首次将条件流匹配引入频域进行确定性 MTS 预测。通过复值线性层在频域中插值频谱，配合流匹配头进行残差学习，仅 89k 参数即达到 SOTA。采用 ODE 单次确定性采样，推理速度远超扩散方法[^src-2511-16426]。

**[[aurora|Aurora]]** (arXiv 2026) 提出 Prototype-Guided Flow Matching，使用多模态领域知识生成条件和原型来引导流匹配过程，实现生成式概率预测[^src-aurora]。Aurora 支持多模态输入（文本、图像、数值）和零样本推理。

### 方法对比

| 方法 | 生成框架 | 模态支持 | 零样本 | 输出类型 | 操作域 |
|------|---------|---------|--------|---------|--------|
| SimDiff | Diffusion (DDPM) | 仅数值 | ✗ | 点估计（MoM 聚合） | 原始域 |
| SpecSTG | Diffusion (谱域) | 仅数值 | ✗ | 概率分布 + 点估计 | **谱域** |
| **FrèqFlow** | **Flow Matching (频域)** | **仅数值** | **✗** | **点估计（确定性）** | **频域** |
| Aurora | Flow Matching (OT) | 文本 + 图像 + 数值 | ✓ | 概率分布 | 原始域 |

## 优势

1. **不确定性量化**：生成式方法自然输出预测分布，无需额外的不确定性建模
2. **多模态条件化**：Flow Matching 和扩散模型天然支持条件生成，便于融入多模态信息[^src-aurora]
3. **灵活采样**：可从预测分布中采样多个实现，支持风险分析和决策

## 挑战

1. **计算成本**：生成式方法通常需要多步采样（扩散模型）或 ODE 求解（流匹配）
2. **训练稳定性**：扩散/流匹配训练比判别式回归更复杂
3. **评估指标**：概率预测需要 CRPS、NLL 等分布级指标，而非简单的 MSE/MAE

## 相关页面

- [[aurora]] — 流匹配生成式预测模型
- [[specstg]] — 谱域扩散时空图预测模型
- [[simdiff]] — 扩散式生成预测模型
- [[freqflow-ts|FrèqFlow/SpectFlow]] — 频域流匹配确定性预测（NeurIPS 2025）
- [[flow-matching]] — Flow Matching 理论基础
- [[diffusion-model]] — 扩散模型理论基础
- [[multimodal-time-series-forecasting]] — 多模态时间序列预测

[^src-aurora]: [[source-aurora]]
[^src-simdiff]: [[source-simdiff]]
[^src-2401-08119-specstg]: [[source-2401-08119-specstg]]
[^src-2511-16426]: [[source-2511-16426]]
