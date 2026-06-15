---
title: "STD-PLM: Understanding Both Spatial and Temporal Properties of Spatial-Temporal Data with PLM"
type: source-summary
tags:
  - traffic-forecasting
  - data-imputation
  - pretrained-language-model
  - spatial-temporal
  - few-shot
  - zero-shot
  - aaai
created: 2026-06-15
last_updated: 2026-06-15
source_count: 1
confidence: high
status: active
---

# STD-PLM

Huang, Mao, Guo, Chen, Shen, Li, Lin & Wan (Beijing Jiaotong University, AAAI 2025). STD-PLM 是一个基于预训练语言模型（PLM）的统一时空数据预测与插补框架，首次通过显式设计的空间和时间 tokenizer 激活 PLM 对时空数据中空间、时间及耦合时空相关性的理解能力[^src-std-plm]。

## 核心方法

STD-PLM 由四个主要模块组成[^src-std-plm]：

1. **Spatial-Temporal Embedding**：包含拓扑感知节点嵌入（Topology-aware Node Embedding）和周期感知时间嵌入。节点嵌入基于图拉普拉斯矩阵前 $K$ 个最大特征值对应的特征向量，具备跨图结构的归纳学习能力[^src-std-plm]。

2. **Spatial-Temporal Tokenizer**：分别从空间和时间两个维度生成 token[^src-std-plm]：
   - **Spatial Tokenizer**：为每个节点生成空间 token，分解为内在状态（拓扑+周期）和动态状态（历史数据），并融入 mask token（统一预测与插补任务）
   - **Temporal Tokenizer**：聚合所有节点的信息，生成整体状态 token $Z_{state}$ 和整体趋势 token $Z_{trend}$

3. **Sandglass Attention (SGA)**：通过 precoder 将节点级空间 token 聚合为更少的区域级 token（$M < N$），送入 PLM 后在 decoder 恢复为节点级表示——既捕获非 pairwise 高阶时空相关性，又降低计算开销[^src-std-plm]。

4. **Unified Output Projection**：将 PLM 隐藏表示加上趋势 token 并与节点级空间 token 做残差连接，经 MLP 映射到预测/插补输出[^src-std-plm]。

PLM backbone 使用 GPT-2 的前 3 层，对注意力层做 LoRA 微调，position embedding 和 layer norm 完全更新[^src-std-plm]。

## 关键结果

- **预测**：PEMS03/04/07/08 四个数据集上达 SOTA 或次优水平，优于 STGLLM、STLLM、OFA 等 PLM-based 基线[^src-std-plm]
- **插补**：PEMS08 上 RM 70% 和 CM 70% 均达 SOTA，大幅超越 PriSTI、BRITS 等[^src-std-plm]
- **Few-shot**：仅用 5% 训练数据即匹配全量 LSTM；20% 数据超越全量 ASTGCN[^src-std-plm]
- **Zero-shot**：直接跨数据集迁移保持可接受性能（如 PEMS04→PEMS08 MAE 29.52）[^src-std-plm]
- **效率**：SGA 显著降低推理时间和 GPU 内存（PEMS04：7.40s / 8554MiB vs 无 SGA 17.96s / 15366MiB）[^src-std-plm]

## 消融发现

- 去除 Temporal Tokens 或约束损失 $L_C$ 均导致预测和插补性能下降[^src-std-plm]
- PLM fine-tune 优于同等规模从头训练的 Transformer encoder——证实 PLM 嵌入的知识对时空任务有益[^src-std-plm]

## 局限

- PLM backbone 引入额外推理开销（尽管只有 GPT-2 前 3 层）[^src-std-plm]
- 仅评估四个 PEMS 流量数据集，未在更广的时空场景验证[^src-std-plm]
- Sandglass Attention 的 $M$ 个区域 token 为固定值，未做自适应[^src-std-plm]

[^src-std-plm]: [[source-std-plm]]
