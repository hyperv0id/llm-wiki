---
title: "TS-Memory: Plug-and-Play Memory for Time Series Foundation Models"
type: source-summary
tags:
  - time-series-forecasting
  - foundation-model
  - knowledge-distillation
  - retrieval-free
  - plug-and-play
  - kdd-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 0
confidence: low
status: active
---

# TS-Memory: Plug-and-Play Memory for Time Series Foundation Models

**源文件**: `raw/ts-memory-time-series-foundation-models-kdd26.pdf` | **出处**: KDD 2026 | **代码**: [github.com/sisuolv/TS-Memory](https://github.com/sisuolv/TS-Memory)

## 核心论点

TS-Memory 提出 Parametric Memory Distillation 范式——将在线检索的预测分布知识离线蒸馏为轻量参数化记忆模块（PlugMem），使冻结的 [[time-series-foundation-model|TSFM]] 在推理时无需检索即可获得检索增强的适应能力。该方法在冻结 backbone 上实现 $O(1)$ 推理，避免了参数适配的灾难性遗忘和非参数检索的推理延迟。

## 方法

两阶段设计：

1. **Stage I — 特权监督构建**：在训练集上构建泄漏安全的 kNN 知识库，用冻结 TSFM 编码器提取嵌入，检索 $K$ 个最近邻的未来轨迹。对每个邻居做 shift alignment（尾部均值偏移校正），按 $\ell_1$ 距离重排，softmax 加权聚合为经验分位数目标 $\hat{Q}_t^{teach}$，并计算检索置信度 $\text{Conf}_t = \max_k w_k$。

2. **Stage II — 置信门控记忆蒸馏**：PlugMem 是轻量 encoder-decoder Transformer，接收原始上下文 $X_t$，输出分位数预测 $\hat{Q}_t^{mem}$。联合训练损失包含：任务损失（pinball loss 对真值）、对齐损失（仅当检索教师优于 backbone 且通过 advantage gate $\chi_t = \mathbb{I}[\text{err}_T + \epsilon_{\text{gate}} < \text{err}_{\text{base}}]$ 时以置信加权 $\omega_t = \chi_t \cdot \text{Conf}_t^\gamma$ 蒸馏 Huber loss）、稳定性正则（锚定中位数至 backbone + 分位数交叉惩罚）。

推理时通过线性融合 $\hat{Q}_t^{final} = (1-\alpha)\hat{Q}_t^{base} + \alpha\hat{Q}_t^{mem}$，$\alpha$ 在验证集上调优。

## 实验结果

- **跨 backbone 一致提升**：在 [[chronos|ChronosBolt]]/Chronos2/[[sundial|Sundial]]/[[timesfm|TimesFM]] 四种冻结 backbone 上，32 对 dataset–backbone 平均 MSE 降 5.8%、MAE 降 2.1%，峰值 MSE 降 16.0%（ETTm2 + TimesFM）。排除已知预训练重叠的 27 对上，降幅略增至 MSE 6.22%/MAE 2.18%。
- **优于适配基线**：在 ChronosBolt 上与 RAFT、TS-RAG、LoRA 对比，TS-Memory 是唯一在全部 8 数据集上同时降低 MSE/MAE/CRPS 的方法。LoRA 的 CRPS 行为不一致，部分数据集退化。
- **推理效率**：TS-Memory 总延迟仅比冻结 backbone 高 3.8–4.7%，而 TS-RAG 因检索占 51–66% 总延迟导致近 2 倍放慢。
- **跨模型迁移**：PlugMem 从单一检索教师蒸馏后可迁移至不同 backbone（Chronos2→ChronosBolt/Sundial/TimesFM），平均 MSE 降 6–10%。从小 backbone（205M）蒸馏的 PlugMem 可直接用于更小变体（降至 9M）保持一致提升。
- **域漂移鲁棒性**：跨域训练平均 MSE 降 2.7%，域内监督最佳达 6.8% MSE 降。

## 局限性

1. 仅适用于长程概率预测，未覆盖插补、异常检测等任务。
2. Shift alignment 主要处理加性通道偏移，时序变形、频率偏移或突变 regime 可能降低可靠性。
3. 检索语料与部署域不匹配时教师目标噪声增大；advantage gate 和 backbone 融合可缓解但无法消除。
4. 融合权重 $\alpha$ 需逐 dataset–backbone 对在验证集调优，输入自适应融合留待未来。
