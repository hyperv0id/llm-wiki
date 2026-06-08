---
title: "Patch Reprogramming"
type: technique
tags:
  - time-series
  - llm
  - reprogramming
  - cross-modality
created: 2026-06-04
last_updated: 2026-06-08
source_count: 2
confidence: medium
status: active
---

# Patch Reprogramming

Patch Reprogramming 是 Time-LLM 的核心技术，通过多 head 交叉注意力将时间序列 patch embeddings 对齐到 LLM 的预训练词嵌入空间，使冻结的语言模型能够理解并推理时间序列数据 [^src-time-llm]。

## 机制

### 问题

时间序列是连续的，LLM 操作在离散 token 上。直接编辑或无损描述时间序列为自然语言不可行 [^src-time-llm]。

### 解决方案

1. **Text Prototypes 提取**：从 LLM 预训练词嵌入 $E \in \mathbb{R}^{V \times D}$ 通过线性探测得到 $E' \in \mathbb{R}^{V' \times D}$（$V' \ll V$）——一组紧凑的语言线索集合
2. **Cross-Attention 对齐**：对每个 head $k$，$\text{ATTENTION}(Q_k=X_P W_k^Q, K_k=E' W_k^K, V_k=E' W_k^V)$，其中 $X_P$ 是时序 patch embeddings
3. **输出**：聚合多头注意力结果 → 线性投影 → 对齐到 LLM 隐藏维度 $D$

### 关键特性

- 每个 patch 被少数 text prototypes 的加权组合表示，而非全量词表 [^src-time-llm]
- 可视化显示 prototypes 收敛到描述时序属性的词，如 "periodic", "seasonal", "quantile", "average" [^src-time-llm]
- 不同 patches 因语义差异被分配给不同的 prototype 组合 [^src-time-llm]

## 实验证据

- 移除 Patch Reprogramming → 全量预测平均 9.2% MSE 退化，few-shot 场景超过 17% [^src-time-llm]
- 仅 ~6.6M 可训练参数即可激活 LLM 的时序预测能力 [^src-time-llm]

> [!warning] 对不完整序列的反例
> [[nuwats|NuwaTS]] (arXiv 2024) 报告：在**插补**任务中，Patch Reprogramming 式的文本对齐**劣于简单线性嵌入**[^src-nuwats]。其论点是——不完整 patch 缺失比例高且缺失位置多变，patch 与 text prototype 的语义匹配被破坏，模态对齐无法有效表征复杂缺失序列。NuwaTS 表 14 在 6 个数据集上验证（如 ETTh1 MSE：线性 0.164 vs 文本对齐 0.250；ECL：0.085 vs 0.205）[^src-nuwats]。这界定了 Patch Reprogramming 的适用边界：对**完整**序列有效，对**高缺失**序列可能适得其反。

## 与相关技术的区别

| 方法 | 对齐策略 | 修改 LLM？ |
|------|---------|-----------|
| Time-LLM Patch Reprogramming | Cross-attention to text prototypes | 否 |
| GPT4TS | 隐式（直接 fine-tune） | 是 |
| LLMTime | 数值文本化 | 否 |
| Voice2Series | 编辑输入到声学模型格式 | 否 |

## Connections

- 属于：[[time-llm]] — Time-LLM 框架的核心组件
- 基于：[[model-reprogramming]] — 跨域模型重编程范式
- 对比：[[prompt-as-prefix]] — Time-LLM 的另一核心组件
- 反例：[[nuwats]] — NuwaTS 证明文本对齐对不完整序列不如线性嵌入

[^src-time-llm]: [[source-time-llm]]
[^src-nuwats]: [[source-nuwats]]
