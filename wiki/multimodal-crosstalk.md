---
title: "Multimodal Crosstalk (MCT)"
type: technique
tags:
  - cross-modal-fusion
  - neural-field
  - self-attention
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Multimodal Crosstalk (MCT)

**Multimodal Crosstalk (MCT)** 是 [[omnifield|OmniField]] 中的跨模态信息交换模块，用于在条件化神经场（CNF）中将来自不同模态的异构传感器信号融合为统一的潜表示[^src-omnifield]。

## 机制

给定输入时间 $t_{\text{in}}$ 的可用模态集合 $\mathcal{M}_{\text{in}}$，MCT 的输出为[^src-omnifield]：

$$h := \text{MCT}(\{U_m^{t_{\text{in}}}\}_{m \in \mathcal{M}_{\text{in}}}, z) = \mathcal{P}\left(\bigodot_{m=1}^{M} \left[\mathcal{E}_m(U_m^{t_{\text{in}}}) \oplus z\right]\right)$$

其中：
- $\mathcal{E}_m(U_m^{t_{\text{in}}}) \in \mathbb{R}^{n \times d}$ 为第 $m$ 个模态的编码特征
- $\odot$ 表示跨模态拼接
- $z \in \mathbb{R}^{1 \times d}$ 为全局特征，携带跨模态聚合信息
- $\oplus$ 为广播加法注入全局上下文
- $\mathcal{P}$ 为多层自注意力 multimodal processor

## 全局特征 $z$ 的双重角色

$z$ 既是**跨模态通信的全局信息聚合**，也是**紧凑的信息瓶颈**，随网络层演化：来自上一层 ICMR 迭代的池化输出，向所有模态广播，实现轻量级全局条件化[^src-omnifield]。

## 与标准 Mid-Fusion 的区别

标准 Mid-Fusion 先对各模态独立编码再中途融合，缺乏跨模态前瞻性的信号对齐。MCT 在编码阶段即通过 $z$ 注入全局跨模态上下文，使每模态编码能感知其他模态的存在和结构[^src-omnifield]。

## 相关

- [[iterative-cross-modal-refinement]] — MCT 的迭代版本
- [[omnifield]] — 使用了 MCT 的完整模型

[^src-omnifield]: [[source-omnifield]]
