---
title: "聚类感知引导"
type: technique
tags:
  - diffusion-models
  - clustering
  - dynamic-guidance
  - spatial-temporal
  - aaai-2026
created: 2026-06-08
last_updated: 2026-08-26
source_count: 1
confidence: medium
status: active
---

# 聚类感知引导 (Cluster-Aware Guidance)

**聚类感知引导**是 [[fence|FENCE]]（AAAI 2026）中提出的引导尺度计算策略，通过在扩散去噪过程中对节点进行动态聚类来获得更稳定、更准确的引导尺度估计[^src-fence]。

## 动机

在时空扩散插补中，不同节点对条件观测的符合程度差异很大[^src-fence]：

- **全局统一引导尺度**：对所有节点使用相同的 $\lambda$，无法区分不同节点的条件满足程度——高缺失率节点需要更强的引导，而低缺失率节点可能被过度引导
- **逐节点引导尺度**：每个节点独立计算 $\lambda$，但在稀疏观测下统计不稳定，估计方差大

聚类感知引导在两者之间取得平衡：利用空间相关性将节点分组，组内共享引导尺度，既获得了比全局更精细的区分度，又比逐节点更稳定[^src-fence]。

## 机制

### 动态聚类

在每一步去噪 $k$ 中，FENCE 从条件去噪网络的**空间注意力分数** $A_{\text{attn}} \in \mathbb{R}^{N \times N}$ 提取节点间的动态相关性，然后使用 k-means 聚类将 $N$ 个节点划分为 $K_c$ 个聚类 $\{C_1, C_2, \dots, C_{K_c}\}$[^src-fence]。

注意力分数在去噪过程中动态演化，反映了不同扩散步下节点间关系的变化[^src-fence]。

### 聚类级后验聚合

对每个聚类 $C_j$，计算聚类级对数后验的均值[^src-fence]：

$$\log p_{\theta,k-1,C_j}(c|x_{k-1}) = \frac{1}{|C_j|} \sum_{l \in C_j} \log p_{\theta,k-1,l}(c|x_{k-1})$$

聚类级后验用于计算该聚类内所有节点的共享引导尺度 $\lambda_{C_j}$[^src-fence]。

### 聚类数的影响

FENCE 实验表明，聚类数 $K_c = N/20$ 时性能最优[^src-fence]：
- $K_c = 1$（全局统一）：退化为忽略节点差异，性能退化
- $K_c = N$（逐节点）：统计不稳定，性能退化
- $K_c = N/20$：在区分度和稳定性之间取得最佳平衡[^src-fence]

## 与相关机制的关系

聚类感知引导是 [[feedback-diffusion-guidance|反馈扩散引导]] 的空间维度扩展——反馈引导提供时间维度（跨去噪步）的动态调整，聚类感知提供空间维度（跨节点）的差异化调整[^src-fence]。

[^src-fence]: [[source-fence]]