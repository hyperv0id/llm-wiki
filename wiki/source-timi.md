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
last_updated: 2026-08-11
source_count: 1
confidence: medium
status: active
---

# TiMi: Empower Time Series Transformers with Multimodal Mixture of Experts

**Authors**: Jiafeng Lin, Yuxuan Wang, Huakun Luo, Zhongyi Pei, Jianmin Wang (Tsinghua). **Venue 标注**: ICML 2026. **arXiv**: 2602.21693v1（2026-02-25，Preliminary work）。

## 问题

外生文本（政策、公告等）可携带波动原因，但与数值序列难对齐；直接 alignment / fusion 易引入噪声[^src-timi]。

## 方法

- 冻结 LLM 从文本推理结构化因果知识（趋势等），作为预测引导而非特征拼接[^src-timi]。
- **MMoE** 插件换 FFN：**TMoE** 文本门控 + **SMoE** 全局序列门控，共享 expert 池[^src-timi]。
- 方法标签：**Non-Fusion Guidance**（相对 Early / Late Fusion）[^src-timi]。
- 可解释性：SMoE 路由与 Mann–Kendall 趋势方向相关[^src-timi]。

## 实验（文内）

- 16 基准：9 Time-MMD + 7 Time-IMM[^src-timi]。
- 相对 backbone 平均 MSE：PatchTST −18.2%，TimeXer −12.5%，Autoformer −12.4%[^src-timi]。
- 不规则子集相对 PatchTST：−29.57%；Time-MMD 子集 −11.26%[^src-timi]。
- 消融：TMoE+SMoE > 单模块；标准 MoE / Cross-Attention / 随机初始化 LLM 更差[^src-timi]。

LLM：Qwen2.5-7B-Instruct；依赖推理质量[^src-timi]。

[^src-timi]: [[source-timi]]
