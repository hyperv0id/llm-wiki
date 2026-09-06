---
title: "Hybrid Lead-Lag Graph Construction"
type: technique
tags:
  - graph-learning
  - llm-for-time-series
  - lead-lag-dependency
  - adaptive-graph
created: 2026-09-05
last_updated: 2026-09-05
source_count: 1
confidence: medium
status: active
---

# Hybrid Lead-Lag Graph Construction

**Hybrid Lead-Lag Graph Construction（HLLG）** 是 [[lagllm|LagLLM]] 提出的 lead–lag 图构建方法：用可学习嵌入参数化初始图，再用空间掩码与 frozen LLM 判出的语义掩码精炼，统一数据驱动与知识驱动两条线索[^src-lagllm]。

## 初始数据驱动图

沿用自适应图学习思路（AGCRN、MillGNN 一系），用可学习嵌入的内积相似度参数化邻接。空间嵌入 $E_S\in\mathbb{R}^{N_g\times d_e}$（组级）与时间嵌入 $E_T\in\mathbb{R}^{N_p\times d_e}$（patch 位置级）按广播逐元素相加得 $E_F$，初始图为[^src-lagllm]：

$$A_D = \mathrm{softmax}(E_F E_F^\top),\quad A_D\in\mathbb{R}^{N_g\times N_g\times N_p\times N_p}$$

其中 $A_{ij,nm}\neq 0$ 表示组 $i$ 的 patch $n$（lead）到组 $j$ 的 patch $m$（lag）的依赖。论文认为这种稠密参数化无方向先验、有过拟合风险，因此需要精炼[^src-lagllm]。

## Spatial Mask

$M_S = S A_S S^\top$：把节点级空间结构 $A_S$（如距离矩阵）经分配矩阵 $S$ 聚合到组级，强化空间邻近且强连通的组间 lead–lag 依赖、抑制无关组对[^src-lagllm]。

## Semantic Mask

用 frozen LLM 判断组对之间是否存在"稳定且可解释"的 lead–lag 依赖。lead–lag prompt 四组件[^src-lagllm]：

1. **Task Instruction**：角色设定（时序 lead–lag 分析专家）+ 输出格式（仅答 yes/no，证据弱/不稳定/矛盾/歧义一律 no）
2. **Time Series Statistics**：两组序列的观测值、min、max、mean、std、trend
3. **Lead-Lag Criteria**：可学习部分——$M_S$ 的组连通强度与 $A_D$ 的 lead–lag 权重；预定义部分——交叉相关、Granger 因果、稳定性、可解释性（只是提及以引导推理，不预计算）
4. **Question**：是否存在稳定可解释的 lead–lag 依赖

为减少 LLM 调用，每组只按 $M_S$ 取 Top-$K$ 个最相关邻组，逐对组批并行。LLM 输出经软分类头（线性 + softmax）解码为 $M_K\in[0,1]^{N_g\times N_g}$。该掩码随输入样本动态变化[^src-lagllm]。

## 组合

$$A_L = A_D \otimes (M_S + M_K)$$

$\otimes$ 为广播逐元素乘。消融显示组合优于任何单一来源：PEMS08 上完整版 MAE 13.21，w/o dm（去数据驱动掩码）13.58，w/o km（去语义掩码）13.63，w/o llg（去整个图）14.14[^src-lagllm]。

## 设计要点

- **frozen vs fine-tuned 的分工**：图构建阶段的 LLM 冻结（语义判断不训练），预测骨干阶段的 LLM 用 LoRA 微调——同一论文中 LLM 出现两次、角色不同。
- **软语义掩码**：LLM 的 yes/no 经分类头变成 $[0,1]$ 连续值，而非硬 0/1 过滤。
- **组级稀疏化**：Top-$K$（PEMS08 最优 $K{=}3$、PEMS04 最优 $K{=}2$）控制 LLM 调用量与掩码稀疏度，$K$ 过大引入噪声相关[^src-lagllm]。

[^src-lagllm]: [[source-lagllm-icml2026]]
