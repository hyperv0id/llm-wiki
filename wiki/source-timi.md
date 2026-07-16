---
title: "TiMi: Empower Time Series Transformers with Multimodal Mixture of Experts"
type: source-summary
tags:
  - time-series
  - multimodal
  - mixture-of-experts
  - llm
  - forecasting
  - icml-2026
created: 2026-07-25
last_updated: 2026-07-25
source_count: 1
confidence: medium
status: active
---

# TiMi: Empower Time Series Transformers with Multimodal Mixture of Experts

**Authors**: Jiafeng Lin, Yuxuan Wang, Huakun Luo, Jianmin Wang, Zhongyi Pei (Tsinghua University). **Venue**: ICML 2026.

## 核心论点

TiMi 提出了一种范式转换式的多模态时间序列预测方法：用 LLM 从外生文本中**推理**未来趋势的因果知识来**引导**预测，而非传统的跨模态特征对齐或融合[^src-timi]。核心组件是 **Multimodal Mixture-of-Experts (MMoE)**，一个轻量级即插即用模块，可注入任意 Transformer-based 时间序列模型。

## 关键贡献

1. **Non-Fusion Guidance 范式**：不同于 Early Fusion（将 TS 嵌入 LLM 文本空间）和 Late Fusion（分别编码后映射），TiMi 让 LLM 独立推理生成未来趋势的结构化知识，通过 MMoE 引导时间序列 backbone 预测，无需显式模态对齐[^src-timi]。
2. **MMoE 模块**：包含两个互补专家系统——**TMoE**（Text-Informed MoE，基于文本表示路由 experts）注入外生因果知识；**SMoE**（Series-Aware MoE，基于全局序列表示路由 experts）提供历史趋势视角[^src-timi]。
3. **可解释性**：SMoE 的专家选择展现出清晰的趋势依赖性——上升趋势序列路由至同一专家，下降趋势路由至另一专家（通过 Mann-Kendall 检验验证）[^src-timi]。

## 实验

- 在 **16 个真实世界多模态基准**（9 Time-MMD + 7 Time-IMM 不规则数据集）上一致 SOTA[^src-timi]。
- 通用性：MMoE 使 PatchTST 平均提升 18.2%，TimeXer 提升 12.5%，Autoformer 提升 12.4%[^src-timi]。
- 消融：TMoE+SMoE 联合 > 单独模块 > 替换为标准 MoE/Cross-Attention；随机初始化 LLM 显著下降，验证预训练 LLM 因果知识的关键作用[^src-timi]。
- 不规则多模态数据：TiMi 在不规则 Time-IMM 上平均 MSE 降低 29.57%（vs PatchTST backbone），远超 Time-MMD 的 11.26%[^src-timi]。

## 局限与展望

使用 Qwen2.5-7B-Instruct 作为标准 LLM，对 LLM 推理质量有依赖。展望：将 MMoE 嵌入基础模型进行大规模预训练或领域微调[^src-timi]。

[^src-timi]: [[source-timi]]
