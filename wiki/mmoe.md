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
last_updated: 2026-07-25
source_count: 1
confidence: medium
status: active
---

# Multimodal Mixture-of-Experts (MMoE)

**MMoE** 是 [[timi|TiMi]] (ICML 2026) 提出的轻量级即插即用模块，用于将外生文本因果知识注入 Transformer-based 时间序列模型[^src-timi]。它替换标准 FFN 层，由两个互补的 MoE 子模块组成——TMoE 和 SMoE。

## 设计动机

多模态时间序列预测的核心挑战是数值与文本模态缺乏语义对应。传统融合方法（Early/Late Fusion）试图在表示层面对齐，效果有限。MMoE 放弃融合，改用 MoE 路由机制实现**知识引导**：文本提供未来趋势推理 → 门控选择相关 experts → experts 处理时序 token[^src-timi]。

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

消融实验显示：将 TMoE 替换为标准 MoE（仅基于时序 token routing）或 Cross-Attention 均导致次优结果，验证了基于文本表示的门控设计是性能关键[^src-timi]。

## 通用性

MMoE 可注入任意 Transformer-based backbone（仅替换 FFN），已验证在 PatchTST（+18.2%）、TimeXer（+12.5%）和 Autoformer（+12.4%）上的平均 MSE 提升[^src-timi]。Autoformer 是 point-wise 的 encoder-decoder 架构，仍能受益于 MMoE，进一步验证了模块的架构无关性[^src-timi]。

[^src-timi]: [[source-timi]]
