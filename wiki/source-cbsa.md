---
title: "Towards Interpretable and Efficient Attention: Compressing All by Contracting a Few"
type: source-summary
tags:
  - attention
  - interpretability
  - efficiency
  - algorithm-unrolling
  - mcr2
  - neurips-2025
created: 2026-04-29
last_updated: 2026-04-29
source_count: 1
confidence: medium
status: active
---

# Source: CBSA — Compressing All by Contracting a Few

Wen, Huang & Li (BUPT) 提出 Contract-and-Broadcast Self-Attention (CBSA)，一种通过算法展开（algorithm unrolling）从优化目标推导出的可解释且高效的注意力机制。发表于 NeurIPS 2025。

## 核心贡献

1. **统一优化目标**：基于 MCR²（Maximal Coding Rate Reduction）目标，引入代表性 token（representatives）概念，将"压缩所有 token"转化为"收缩少数代表性 token"，实现线性复杂度[^src-cbsa]。

2. **CBSA 机制**：通过算法展开推导出 Contract-and-Broadcast Self-Attention，包含两个阶段：(a) 代表性 token 提取（通过 cross-attention 近似 coding rate）；(b) 代表性 token 收缩 + 收缩广播回所有输入 token[^src-cbsa]。

3. **统一注意力公式**：CBSA 通过改变代表性 token 的数量和结构，可实例化为 softmax attention（自表达代表）、linear attention（正交代表）、channel attention（固定正交代表）和 agent attention（移除收缩步骤），揭示了不同注意力机制之间的根本联系[^src-cbsa]。

## 方法论

### 优化目标

论文从 MCR² 目标出发，引入代表性 token $Q = q(Z) \in \mathbb{R}^{d \times m}$（$m \ll N$），将压缩项中的输入 token 替换为代表，并施加不等式约束 $|R(U_k^\top Q) - R(U_k^\top Z)| \leq \tau$，确保收缩代表性 token 等价于压缩所有输入 token[^src-cbsa]。

### CBSA 的梯度步骤

对压缩项执行梯度下降步，得到 CBSA 算子：

$$Z \leftarrow Z - \kappa \sum_{k=1}^{K} U_k \underbrace{U_k^\top Q \left(I_m + \frac{p}{m\epsilon^2}(U_k^\top Q)^\top (U_k^\top Q)\right)^{-1}}_{\text{Contraction}} \underbrace{A_k^\top}_{\text{Broadcast}}$$

其中 Contraction 项给出代表性 token 的收缩方向，Broadcast 项通过复用提取阶段的注意力矩阵 $A_k$ 将收缩广播回所有输入 token[^src-cbsa]。

实际实现中，矩阵逆通过 Gram 矩阵 + softmax 近似，收缩项变为 self-attention 形式：$U_k^\top Q \cdot \text{softmax}((U_k^\top Q)^\top (U_k^\top Q))$[^src-cbsa]。

### 计算复杂度

固定代表性 token 数量 $m = p = d/K$ 时，CBSA 复杂度为 $O(2Nd^2 + 3Nmd + 2m^2d)$，与输入 token 数 $N$ 呈线性关系。当 $N > 2d/K$ 时，FLOPs 低于 MSSA[^src-cbsa]。

## 实验验证

- **ImageNet-1K 分类**：CBT-Small 以 ViT-S 30% 的参数量和 40% 的 FLOPs 达到可比精度（71.4 vs 72.4）。CBT-Base 达到 73.4%[^src-cbsa]。
- **语义分割（ADE20K）**：CBT decoder 以 Segmenter 20% 的 FLOPs 和 0.06% 的 pairwise similarity 计算量，mIoU 提升 1.5%[^src-cbsa]。
- **可解释性验证**：合成数据上 CBSA 迭代确实将 token 压缩到低维子空间；[CLS] attention map 展示语义分割特性涌现；参数扰动下 CBSA 极为鲁棒[^src-cbsa]。
- **吞吐量**：512×512 分辨率下，CBT 训练/推理速度显著优于 ViT 和 CRATE[^src-cbsa]。

## 局限性

- 子空间结构假设（union of linear subspaces）可能不适用于所有模态（论文排除了 NLP 任务）[^src-cbsa]。
- 从预训练 ViT 适配 CBSA 时性能略低于 linear attention 适配（因结构差异更大）[^src-cbsa]。
- 早期层的"解压缩"现象原因不明，被标记为"known unknown"[^src-cbsa]。

## 引用

[^src-cbsa]: [[source-cbsa]]
