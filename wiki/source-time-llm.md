---
title: "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"
type: source-summary
tags:
  - time-series
  - llm
  - reprogramming
  - transformer
  - foundation-model
  - forecasting
  - iclr
created: 2026-06-04
last_updated: 2026-06-04
source_count: 1
confidence: high
status: active
---

# Time-LLM: Time Series Forecasting by Reprogramming Large Language Models

Jin, Wang, Ma, Chu, Zhang, Shi, Chen, Liang, Li, Pan & Wen (Monash / Ant Group / IBM / Griffith / Alibaba / HKUST-GZ, ICLR 2024). Time-LLM 是首个通过 reprogramming 而非 fine-tuning 将冻结 LLM 重用于通用时间序列预测的框架 [^src-time-llm]。

## 核心方法

Time-LLM 包含三个组件 [^src-time-llm]：

1. **Input Embedding**：RevIN 归一化 → patching（$L_p$ 长度，$S$ 步长）→ 线性 patch embedder 映射到 $d_m$ 维
2. **Patch Reprogramming**：多 head 交叉注意力将 patch embeddings 对齐到预训练词嵌入空间。核心技巧是用少量 text prototypes $E' \in \mathbb{R}^{V' \times D}$（$V' \ll V$，通过线性探测源词嵌入 $E$ 得到）作为 key/value，让每个 patch 被少数 text prototypes 的加权组合表示
3. **Prompt-as-Prefix (PaP)**：将自然语言提示前缀（数据集背景、任务指令、输入统计信息等）与 reprogrammed patches 拼接输入 LLM，输出 representations 经 flatten + linear 投影生成预测

LLM backbone **完全冻结**，仅训练轻量级 input transformation 和 output projection（~6.6M 参数，Llama-7B 的 0.2%）[^src-time-llm]。

## 关键实验结果

- **长时预测**（8 数据集，H∈{96,192,336,720}）：7/8 数据集 MSE 最优，平均超越 GPT4TS 12%、TimesNet 20%、PatchTST 1.4% [^src-time-llm]
- **短时预测**（M4 benchmark）：SMAPE=11.983，超越所有 baseline [^src-time-llm]
- **10% few-shot**：超越 GPT4TS 5% MSE；vs PatchTST/DLinear/TimesNet 平均提升 8%/12%/33% [^src-time-llm]
- **5% few-shot**：超越 GPT4TS 5%；vs PatchTST/DLinear/TimesNet 平均 20% [^src-time-llm]
- **零样本跨域**：超越 GPT4TS 22%（MSE↓14.2%），超越 LLMTime（同规模 7B backbone）75%+ [^src-time-llm]

## 消融关键发现

- Patch Reprogramming 移除 → 平均 9.2% 性能退化，few-shot 场景 >17% [^src-time-llm]
- PaP 移除 → 8% 退化，few-shot 场景 >19% [^src-time-llm]
- 输入统计信息（trend, lags）是 PaP 三个组件中最关键的，移除后 MSE 增加 10.2% [^src-time-llm]
- 不同 LLM backbone：Llama-7B > Llama-8 层 > GPT-2 (12) > GPT-2 (6)，scaling law 在 reprogramming 后保留 [^src-time-llm]

## 局限性

- LLM backbone 推理开销大（Llama-7B ~32GB 显存），尽管可训练参数极少仍依赖大模型 [^src-time-llm]
- Reprogramming 空间的可解释性仅限于 text prototypes 对应的词集分析 [^src-time-llm]
- 未探索时序专用预训练或多模态联合推理 [^src-time-llm]

[^src-time-llm]: [[source-time-llm]]
