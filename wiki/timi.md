---
title: "TiMi"
type: entity
tags:
  - time-series
  - multimodal
  - mixture-of-experts
  - llm
  - forecasting
  - icml-2026
created: 2026-07-25
last_updated: 2026-07-28
source_count: 2
confidence: high
status: active
---

# TiMi

**TiMi**（**Ti**me Series Transformers with **Mi**xture of Experts）是清华大学提出的多模态时间序列预测框架，发表于 ICML 2026[^src-timi]。核心思想：用 LLM 的因果推理能力从外生文本中提取未来趋势的结构化知识，通过 **[[mmoe|Multimodal Mixture-of-Experts (MMoE)]]** 模块注入 Transformer-based 时间序列模型进行预测引导[^src-timi]。

## 与其他多模态方法的核心区别

TiMi 提出 **[[non-fusion-guidance|Non-Fusion Guidance]]** 范式，区别于两类主流方法[^src-timi]：

| 范式 | 代表方法 | 机制 | 局限性 |
|------|---------|------|--------|
| Early Fusion | [[time-llm|Time-LLM]], UniTime, AutoTimes | 将 TS 嵌入 LLM 文本空间，LLM 作为 backbone | 计算开销大，未利用外部因果知识 |
| Late Fusion | Time-MMD, IMM-TSF, [[vot|VoT]] | 分别编码 TS 和文本，后融合映射为预测 | 文本与数值缺乏语义对应，对齐困难 |
| **Non-Fusion Guidance** | **TiMi** | LLM 独立推理 → 结构化因果知识 → 通过 MoE 引导 TS backbone | — |

TiMi 与 [[vot|VoT]] 都使用 LLM 推理文本，但 VoT 仍依赖 feature-level 对齐（多级对齐），而 TiMi 完全放弃模态融合，改为 MoE routing 实现知识引导[^src-timi]。

与 [[constrained-text-fusion|Constrained Text Fusion / CFA]]（Lee et al., arXiv:2603.22372）对照：CFA 同样发现 Time-MMD 上 **naive 融合常低于 unimodal**，但用 **低秩残差 / 门控等受控融合** 保留特征注入；TiMi 则彻底 **Non-Fusion**，用 MoE 路由代替表示融合——问题诊断相近，机制正交[^src-timi][^src-constrained-text-fusion]。

## 架构

1. **Text Reasoning** — 冻结 LLM (Qwen2.5-7B-Instruct) 处理外生文本，通过平均池化生成含因果知识的文本 token[^src-timi]。
2. **Series Embedding** — Patch-based tokenization（同 PatchTST）将历史序列分割为重叠 patch 并线性投影[^src-timi]。
3. **MMoE Plugin** — 替换 Transformer backbone 中的标准 FFN：

   - **TMoE**（Text-Informed MoE）：文本 token 经线性门控 → sparse Top-K routing → 选中的 experts 处理所有时序 token，注入文本推导的未来趋势[^src-timi]。
   - **SMoE**（Series-Aware MoE）：所有时序 patch token 拼接为全局序列表示 → 门控 → Top-K routing → 基于全局趋势的互补引导[^src-timi]。

输出为两路加权和：$\text{MMoE}(h, \bar{H}) = \sum_{i\in\tau_x} s_{i,x}\text{FFN}_i(h) + \sum_{i\in\tau_s} s_{i,s}\text{FFN}_i(h)$[^src-timi]。

## 关键结果

- 16 个多模态基准上一致 SOTA（9 Time-MMD + 7 Time-IMM 不规则数据集）[^src-timi]。
- 即插即用：PatchTST +18.2%、TimeXer +12.5%、Autoformer +12.4%（平均 MSE 提升）[^src-timi]。
- 不规则数据：TiMi 降低平均 MSE 29.57%（vs PatchTST backbone），远超 Time-MMD 11.26%[^src-timi]。
- SMoE 可解释性：专家选择与 MK 趋势检验强相关，上升/下降趋势自动路由至不同专家[^src-timi]。


## 相关页面

- [[source-timi]] · [[non-fusion-guidance]] · [[mmoe]] · [[vot]] · [[time-mmd]] · [[constrained-text-fusion]] · [[multimodal-time-series-forecasting]]

[^src-timi]: [[source-timi]]
[^src-constrained-text-fusion]: [[source-constrained-text-fusion]]
