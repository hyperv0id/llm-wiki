---
title: "Time-LLM"
type: entity
tags:
  - time-series
  - llm
  - reprogramming
  - forecasting
  - iclr
created: 2026-06-04
last_updated: 2026-06-09
source_count: 4
confidence: high
status: active
---

# Time-LLM

Time-LLM 是 Jin et al. (ICLR 2024) 提出的框架，首次通过 **model reprogramming** 将冻结的大语言模型重用于通用时间序列预测，无需任何 backbone fine-tuning [^src-time-llm]。

## 核心设计

### 三组件架构

| 阶段 | 操作 | 可训练 |
|------|------|--------|
| 输入变换 | Patching + 线性 embedder + Patch Reprogramming (cross-attention to text prototypes) | ✓ |
| LLM Backbone | 冻结的 Llama/GPT-2，接收 [PaP prompts \| reprogrammed patches] | ✗ |
| 输出投影 | Discard prefix → flatten → linear → forecasts | ✓ |

### Patch Reprogramming

将时序 patch embeddings 映射到 LLM 预训练词嵌入空间的关键机制 [^src-time-llm]：
- 从 LLM 的预训练词嵌入 $E \in \mathbb{R}^{V \times D}$ 线性探测出一小组 text prototypes $E' \in \mathbb{R}^{V' \times D}$（$V' \ll V$）
- 以时序 patch 为 query、text prototypes 为 key/value 执行 multi-head cross-attention
- 每个 patch 被少数 text prototypes 的组合表示（如 "short up then down steadily"）
- 可视化显示 prototypes 收敛到描述时序属性的词（"periodic", "seasonal", "quantile", "average"）[^src-time-llm]

### Prompt-as-Prefix (PaP)

三部分 prompt 前置在 reprogrammed patches 前 [^src-time-llm]：
1. **Dataset Context**：输入时序的背景信息（如 ETT 数据集的电力变压器描述）
2. **Task Instruction**：预测任务说明（"Predict the next H steps given the previous T steps"）
3. **Input Statistics**：趋势、top-5 lags、min/max/median 等统计量

## 关键性能

- **长时预测**：7/8 数据集 MSE 最优（ETTh1/ETTh2/ETTm1/ETTm2/Weather/ECL/ILI），仅 Traffic 上 PatchTST 稍优 [^src-time-llm]
- **短时预测**（M4）：SMAPE=11.983，MASE=1.595，OWA=0.859 [^src-time-llm]
- **Few-shot**：10% 数据 5% MSE↓ vs GPT4TS；5% 数据 20%+ MSE↓ vs PatchTST/TimesNet [^src-time-llm]
- **Zero-shot 跨域**：22% MSE↓ vs GPT4TS；75%+ vs LLMTime [^src-time-llm]

## 效率

- 可训练参数仅 ~6.6M（Llama-7B 的 0.2%）
- 总内存和时间主要由 backbone LLM 决定，而非 reprogramming 网络 [^src-time-llm]
- 兼容量化等轻量化技术 [^src-time-llm]

## 与其他 LLM+TS 方法的区别

| 维度 | Time-LLM | GPT4TS | LLMTime | LLM4TS |
|------|----------|--------|---------|--------|
| LLM 处理 | 冻结 | Fine-tune | 冻结 | 两阶段 fine-tune |
| 时序输入 | Patch Reprogramming | 直接输入 | 数值文本化 | 直接输入 |
| 输出 | Projection | 直接预测 | Token 解码 | 直接预测 |
| 模态对齐 | Cross-attention to text prototypes | 隐式 | 文本化 | 隐式 |

## 局限与未来

- LLM backbone 推理开销大（Llama-7B ~32GB 显存）[^src-time-llm]
- 未利用时序专用预训练或图文等多模态知识 [^src-time-llm]
- 未来方向：最优 reprogramming 表示、时序知识持续预训练、多模态联合推理 [^src-time-llm]

> [!warning] 反例：文本对齐不适合不完整序列
> [[nuwats|NuwaTS]] (arXiv 2024) 在**插补**场景中给出了与 Time-LLM 相反的证据：对于缺失比例高、缺失位置多变的不完整 patch，把时序 patch 通过 [[patch-reprogramming|Patch Reprogramming]] 对齐到 LLM 词嵌入的"文本对齐"策略**不如简单线性嵌入**[^src-nuwats]。NuwaTS 的表 14 显示，简单线性层在全部 6 个数据集上均优于文本对齐（如 ETTh1 MSE 0.164 vs 0.250）。NuwaTS 据此摒弃硬文本提示，改用统计嵌入 + 缺失嵌入直接编码序列信息[^src-nuwats]。这提示 Time-LLM 的文本对齐增益可能依赖**完整序列**——缺失会破坏 patch 与文本原型的语义匹配。

## Connections

- 基于：[[patch-based-tokenization]] — Patching + 线性 embedder
- 基于：[[instance-normalization]] — RevIN 归一化
- 基于：[[channel-independence]] — 逐变量独立处理
- 增强：[[cvpe]] — 在 Time-LLM 的 patch embedding 注入跨变量上下文
- 对比：[[timesfm]] — 解码器专用时序基础模型
- 对比：[[chronos]] — 时序 tokenization + 语言模型
- 关系：[[multimodal-time-series-forecasting]] — LLM+TS 多模态预测
- 关系：[[timecap]] — 双 LLM agent 时序事件预测
- 演化：[[streasoner]] — STReasoner, 首个时空推理 TS-LM, 在 Time-LLM 范式上增加 graph + multi-step CoT + spatial-aware RL
- 演化：[[cogencast]] — CoGenCast (ICML 2026), 在 Time-LLM 的 LLM+TS 思路上将冻结 LLM 重构为 encoder-decoder，增加流匹配条件化生成，实现一步概率预测
- 概念：[[model-reprogramming]] — 跨域模型重编程范式
- 反例：[[nuwats]] — NuwaTS 证明文本对齐对不完整序列不如线性嵌入，且复用 NLP 权重做插补基础模型
- 演化/对比：[[fstllm|FSTLLM]] (ICML 2025) — 针对 Time-LLM 等方法"通用任务提示 + 数值微调"未充分利用 LLM 推理的批评，FSTLLM 改用节点专属描述、时序模式总结与 STGNN 预测 token 构成的六段 prompt 对 LLaMA-2-7B 做 QLoRA 微调，并以 LLM 构图增强空间相关性，专攻 few-shot 时空预测[^src-fstllm]
- 对比：[[st-vision-llm|ST-Vision-LLM]] (arXiv 2025) uses Time-LLM as an LLM baseline and frames it as a representative 1D-sequence reprogramming approach that, despite strong temporal modeling, lacks mechanisms for the 2D topological/spatial structure of grid-based traffic; on the Telecom Italia mobile-traffic benchmark ST-Vision-LLM substantially outperforms Time-LLM (which uses a Qwen2.5-7B backbone)[^src-st-vision-llm]

[^src-time-llm]: [[source-time-llm]]
[^src-nuwats]: [[source-nuwats]]
[^src-fstllm]: [[source-fstllm]]
[^src-st-vision-llm]: [[source-st-vision-llm]]
