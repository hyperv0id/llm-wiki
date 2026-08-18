---
title: "Multimodal Mixture-of-Experts (MMoE)"
type: technique
tags:
  - mixture-of-experts
  - multimodal
  - time-series
  - llm
  - forecasting
created: 2026-07-25
last_updated: 2026-08-11
source_count: 1
confidence: medium
status: active
---

# Multimodal Mixture-of-Experts (MMoE)

**MMoE**：[[timi|TiMi]] 插件，换 Transformer FFN；TMoE + SMoE[^src-timi]。

## 动机

数值与外生文本难对齐时，用文本表示门控选 experts，而不是拼接文本特征[^src-timi]。

## 架构

### TMoE：Text-Informed Mixture of Experts

- **输入**：LLM 推理生成的文本因果知识 token $\bar{H}$ + 时序 patch token $h$
- **门控**：$s_{i,t} = \text{Softmax}_i(W_t \bar{H})$，基于文本表示的稀疏 Top-K routing
- **输出**：$\text{TMoE}(h, \bar{H}) = \sum_{i\in\tau_t} s_{i,t} \text{FFN}_i(h)$
- **作用**：从文本外推到未来，注入长程因果信号[^src-timi]

### SMoE：Series-Aware Mixture of Experts

- **输入**：所有时序 patch token 拼接为全局序列表示 $[h_1, \cdots, h_N]$
- **门控**：$s_{i,s} = \text{Softmax}_i(W_s [h_1, \cdots, h_N])$，基于序列全局趋势的 Top-K routing
- **输出**：$\text{SMoE}(h) = \sum_{i\in\tau_s} s_{i,s} \text{FFN}_i(h)$
- **作用**：捕获历史序列全局趋势，为 TMoE 提供互补引导[^src-timi]

### 联合公式

$$\text{MMoE}(h, \bar{H}) = \sum_{i\in\tau_x} s_{i,x}\text{FFN}_i(h) + \sum_{i\in\tau_s} s_{i,s}\text{FFN}_i(h)$$

两路 experts 共享 FFN 池但使用独立门控，分别从文本知识和序列历史两个视角提供引导[^src-timi]。

## 与标准 MoE 的区别

TMoE → 标准 MoE 或 Cross-Attention：更差[^src-timi]。

## Backbone（文内）

相对 backbone 平均 MSE：PatchTST −18.2%，TimeXer −12.5%，Autoformer −12.4%[^src-timi]。

[^src-timi]: [[source-timi]]
