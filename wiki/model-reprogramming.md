---
title: "Model Reprogramming"
type: concept
tags:
  - reprogramming
  - cross-modality
  - transfer-learning
created: 2026-06-04
last_updated: 2026-06-08
source_count: 2
confidence: medium
status: active
---

# Model Reprogramming (模型重编程)

Model Reprogramming 是一种资源高效的跨域机器学习范式——保持预训练源模型（如 LLM、CV 模型）完全冻结，通过学习输入变换和输出投影将模型重用于不同模态/领域的任务，无需 fine-tuning [^src-time-llm]。

## 形式化

给定预训练源模型 $f$（在源域 $\mathcal{S}$ 上训练），目标是在目标域 $\mathcal{T}$ 上执行任务。Reprogramming 学习：
- 输入变换 $\phi: \mathcal{T} \to \mathcal{S}$，将目标数据映射到源模型可处理的表示
- 输出投影 $\psi: \mathcal{S} \to \mathcal{T}$，将源模型输出映射回目标域

$f$ 本身保持冻结，仅 $\phi$ 和 $\psi$ 可训练 [^src-time-llm]。

## 在时间序列中的应用

### Time-LLM
[[time-llm|Time-LLM]] (ICLR 2024) 将 reprogramming 引入时间序列预测 [^src-time-llm]：
- $\phi$：Patching → Patch Reprogramming (cross-attention to text prototypes) + Prompt-as-Prefix
- $f$：冻结的 Llama/GPT-2
- $\psi$：Flatten + Linear projection

仅 ~6.6M 可训练参数（Llama-7B 的 0.2%），在 long-term/short-term/few-shot/zero-shot 预测上全面超越 fine-tuning 方法 [^src-time-llm]。

### 早期工作
- Voice2Series (2021)：将声学模型重编程用于时间序列分类
- 视觉模型重编程：跨域图像分类

### NuwaTS（部分微调 + 即插即用前缀）
[[nuwats|NuwaTS]] (arXiv 2024) 与纯重编程**不同但相关**：它复用 PLM（GPT-2 前 6 层）做插补基础模型，但预训练阶段**部分微调** backbone（LayerNorm、FFN、嵌入层、输出层），而非全冻结[^src-nuwats]。其领域适配则采用真正的冻结-backbone PEFT——[[plug-and-play-prefix-tuning|即插即用前缀]]在冻结 PLM 每层注入可学习 Key/Value。消融发现：**完全冻结 backbone 表现最差**，而**不加载 NLP 预训练权重（从零训练）会显著削弱零样本跨域能力**，证明跨模态知识迁移确实有益[^src-nuwats]。这为"重编程 vs 部分微调"提供了一个数据点：对不完整序列，保留 NLP 权重 + 适度微调优于纯冻结。

## 与相关范式的对比

| 范式 | 源模型 | 目标数据 | 训练开销 |
|------|--------|---------|---------|
| Task-Specific Learning | 无 | 小规模域内 | 从零训练 |
| Fine-Tuning | 冻结 → 解冻 | 域内 | 中等 |
| Model Reprogramming | **全冻结** | **跨域** | **极低** |

## Connections

- 实例化：[[time-llm]] — 首次将 reprogramming 用于时序预测
- 技术：[[patch-reprogramming]] — Time-LLM 的输入变换核心
- 技术：[[prompt-as-prefix]] — Time-LLM 的上下文增强
- 对比：[[nuwats]] — 复用 PLM 但部分微调 backbone + 即插即用前缀做插补基础模型
- 技术：[[plug-and-play-prefix-tuning]] — NuwaTS 的冻结-backbone 前缀 PEFT

[^src-time-llm]: [[source-time-llm]]
[^src-nuwats]: [[source-nuwats]]
