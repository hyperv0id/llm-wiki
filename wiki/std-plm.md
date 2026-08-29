---
title: "STD-PLM"
type: entity
tags:
  - traffic-forecasting
  - data-imputation
  - pretrained-language-model
  - spatial-temporal
  - few-shot
  - zero-shot
  - aaai
created: 2026-06-15
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# STD-PLM

STD-PLM（Spatial-Temporal Data understanding with Pre-trained Language Model）是 Huang et al. (Beijing Jiaotong University, AAAI 2025) 提出的**统一时空预测与插补框架**，基于 GPT-2 预训练语言模型（PLM），通过显式设计的空间-时间 tokenizer 和 Sandglass Attention 实现准确的少样本与零样本时空泛化[^src-std-plm]。

## 动机

现有时空模型面临三个局限[^src-std-plm]：
1. 预测和插补模型各自定制，缺乏统一
2. 零样本和少样本学习能力弱
3. PLM-based 方法仅从空间维度构造 token，忽视时间维度，拓扑信息利用不充分

## 架构

```
ST Data (T×N×C) → Spatial-Temporal Embedding → ST Tokenizer → SGA → PLM → Output Projection
```

### 四个核心模块

| 模块 | 功能 |
|------|------|
| [[topology-aware-node-embedding|Spatial-Temporal Embedding]] | 拉普拉斯特征向量 + 周期嵌入，归纳式节点表示 |
| [[spatial-temporal-tokenizer|Spatial-Temporal Tokenizer]] | 空间 token（内在+动态+mask）+ 时间 token（状态+趋势） |
| [[sandglass-attention|Sandglass Attention (SGA)]] | precoder 聚合 → PLM → decoder 恢复，捕获高阶相关性 + 降开销 |
| Unified Output Projection | 残差连接 → MLP 输出 |

### 训练策略

- Backbone：GPT-2 前 3 层
- 微调：注意力层 LoRA + position embedding / layer norm 全更新
- 约束损失 $L_C = L_G + L_R$：结构感知损失（利用邻接矩阵指导 SGA）+ 正则化项（防止注意力坍塌）

## 关键性能

- **统一任务**：同一模型同时处理预测和插补，均达 competitive 水平[^src-std-plm]
- **Few-shot**：5% 数据匹配全量 LSTM，20% 超越全量 ASTGCN[^src-std-plm]
- **Zero-shot**：PEMS04→PEMS08（MAE 29.52），无需目标域训练[^src-std-plm]
- **效率**：SGA 使推理加速 ~2.5×（PEMS04 17.96s→7.40s）[^src-std-plm]

## 与其他 PLM-based 方法的关系

| 方法 | Token 构造 | 空间建模 | 任务 |
|------|-----------|---------|------|
| [[time-llm|Time-LLM]] (ICLR 2024) | 纯时序 patch | 无 | 预测 |
| [[nuwats|NuwaTS]] (arXiv 2024) | 统计+缺失 patch | 无（CI） | 插补 |
| STLLM (arXiv 2024) | 空间 token | 图聚合 | 预测 |
| STGLLM (arXiv 2024) | 空间 token | 图聚合 | 预测 |
| **STD-PLM** (AAAI 2025) | **空间+时间 token** | 图拉普拉斯特征向量 + SGA | **预测+插补** |

STD-PLM 是首个同时从空间和时间两个维度构造 token 并整合拓扑信息的 PLM-based 时空模型[^src-std-plm]。

## 统一评测口径（Guo et al. 2025）

Guo、Wei 等人的统一评测（11 模型 × 4 个交通数据集 × 20 个缺失场景，官方代码复现 + 网格搜索调参）将 STD-PLM 列入评测清单[^src-guo-imputation-evaluation]。该评测复现口径下：STD-PLM 列 top-4 深度插补模型，在 PEMS04/PEMS08 上与 GCASTN 报告相当水平；PEMS04、SRTR、0.5 设置的挑战/稳定时段分组中，STD-PLM 与 GCASTN、ImputeFormer 同属表现最好的三个模型（挑战期 MAE 23.57/RMSE 35.38/MAPE 9.14，稳定期 MAE 7.64/RMSE 12.00/MAPE 18.63），评测者归因于其分节点建模时空关系并利用 LLM 语言推理能力学习两种时段的交通模式[^src-guo-imputation-evaluation]。效率上（同设置）内存 8744MB 为表内最大，训练 12206.02s、推理 18.44s[^src-guo-imputation-evaluation]。以上数字均为评测者的复现口径（协议见 [[st-traffic-imputation-benchmark]]），与本页原论文实验数字分立。

## Connections

- 基于：[[pretrained-language-model-for-ts]] — PLM 在时间序列/时空数据的跨模态应用范式
- 核心组件：[[spatial-temporal-tokenizer]] — 空间+时间双维度 token 生成
- 核心组件：[[sandglass-attention]] — precoder-decoder 高效注意力模块
- 核心组件：[[topology-aware-node-embedding]] — 拉普拉斯特征向量节点嵌入
- 对比：[[time-llm]] — 纯时序 PLM reprogramming
- 对比：[[nuwats]] — PLM-based 插补基础模型，CI 范式
- 关系：[[traffic-forecasting]] — 交通预测是 STD-PLM 的主要评估场景
- 关系：[[few-shot-traffic-forecasting]] — 时空 FSL 的核心挑战
- 关系：[[st-traffic-imputation-benchmark]] — Guo et al. 统一评测：STD-PLM 的评测口径排名与效率结论
- 关系：[[traffic-missing-patterns]] — 评测所用缺失模式四分类（SRTR/SRTC/SCTR/SCTC）

[^src-std-plm]: [[source-std-plm]]
[^src-guo-imputation-evaluation]: [[source-guo-imputation-evaluation]]
