---
title: "Iterative Cross-Modal Refinement (ICMR)"
type: technique
tags:
  - cross-modal-fusion
  - neural-field
  - iterative-refinement
created: 2026-07-21
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# Iterative Cross-Modal Refinement (ICMR)

**Iterative Cross-Modal Refinement (ICMR)** 是 [[omnifield|OmniField]] 中的迭代跨模态对齐策略。通过多轮 [[multimodal-crosstalk|MCT]] 执行，逐步精炼跨模态信号的融合质量[^src-omnifield]。

## 机制

给定 $\ell$ 个 MCT 块，ICMR 定义为[^src-omnifield]：

$$\text{For } k = 0, \dots, \ell-1:$$
$$h^{(k)} := \text{MCT}(\{U_m^{t_{\text{in}}}\}_{m \in \mathcal{M}}, z^{(k)}) \in \mathbb{R}^{n \times d}$$
$$z^{(k+1)} := \frac{1}{n}\sum_{i=1}^{n} h^{(k)}_{i,:}$$

最终多模态神经场输出为 $g = h^{(\ell-1)}$。初始全局特征 $z^{(0)}$ 以零填充[^src-omnifield]。

## 核心洞见

$z$ 在此充当**通信桥梁**：每轮 MCT 融合后，通过平均池化将更新后的全局结构提炼为紧凑编码，供下一轮 MCT 作为跨模态条件注入。这一循环使模型逐步从不完美的初始对齐收敛到细粒度跨模态对应[^src-omnifield]。

## 鲁棒性优势

ICMR 的关键价值体现在**噪声鲁棒性**：在 ClimSim-THW 实验中，当 1–2 个模态被注入不同程度的高斯噪声时，ICMR 保持接近干净输入的性能，而 Mid-Fusion 随噪声增加持续衰减[^src-omnifield]。原因是 ICMR 的迭代交换机制能将信息路由到更干净的通道并抑制受污染通道，而 Mid-Fusion 缺乏预条件交换，会将单模态噪声放大传播到共享表示中[^src-omnifield]。

## 对比

四种融合策略的特点对比[^src-omnifield]：

| 策略 | 特点 |
|------|------|
| Co-Location | 限制到共同传感器，空间覆盖最差 |
| Interpolation | 插值到全部传感器，引入代理误差 |
| Mid-Fusion | 保留稀疏性但无跨模态前瞻对齐 |
| **ICMR** | 保留稀疏性 + 迭代跨模态信号对齐 |

## 相关

- [[multimodal-crosstalk]] — ICMR 的每轮子模块
- [[omnifield]] — 使用 ICMR 的完整模型
- [[fleximodal-fusion]] — 控制可用模态子集的配套机制

[^src-omnifield]: [[source-omnifield]]
