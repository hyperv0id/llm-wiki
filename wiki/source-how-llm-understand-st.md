---
title: "How Can Large Language Models Understand Spatial-Temporal Data?"
type: source-summary
tags:
  - spatio-temporal
  - llm
  - traffic-prediction
  - tokenizer
  - adapter
  - few-shot
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# How Can Large Language Models Understand Spatial-Temporal Data?

**Authors**: Lei Liu, Shuo Yu, Runze Wang, Zhenxun Ma, Yanming Shen (Dalian University of Technology)
**Venue**: arXiv:2401.14192v2 (May 2024)

## 核心论点

LLM 无法直接消费时空数值数据，瓶颈在于 tokenization：论文提出 STG-LLM，核心是 STG-Tokenizer 把每个节点编码为一个 token——token 内部承载该节点完整的 $L \times F$ 历史序列（时间语义），token 之间的注意力捕获空间语义；N 个节点只需 N 个 token，相比把每步每节点展开为独立 token 大幅压缩序列长度[^src-how-llm-understand-st]。

## 方法机制

- **STG-Tokenizer**：节点 token 拼接 TD-Embedding（time-of-day，按日等分 $K_1$ 段）与 DW-Embedding（day-of-week，$K_2$ 天），两维度均设 64，取最后时间步 $t$ 检索后拼接：$T_e^i = T_g^i \| E_{td}^i \| E_{dw}^i$，维度 $N \times (L \cdot F + C_1 + C_2)$[^src-how-llm-understand-st]
- **STG-Adapter**：仅一层线性编码（$T_q = W_1 T_e + b_1$，对齐 LLM 嵌入维度 D）+ 一层带残差的线性解码（$\hat{X}_{t+1:t+P} = W_3((W_2 H + b_2) + T_e)$）[^src-how-llm-understand-st]
- **Prompt 融合**：prompt 经 Text-Tokenizer 生成 $T_p \in \mathbb{R}^{M \times D}$，与数据 token 拼接（$T = T_p \| T_q$）喂给 LLM；prompt 模板含 domain、instruction、事故/天气/日期等补充信息，例如注入"事故发生于节点 i 时刻 j 并持续 k 小时"[^src-how-llm-understand-st]
- **冻结策略**：冻结 GPT2 的多头注意力和 FFN，只微调 positional embeddings 和 layer norm；总参数 60,885,480，可训练仅 1,033,704（1.70%）——STG-Tokenizer 18,880（0.03%）、STG-Adapter 217,640（0.36%）、Position Embeddings 786,432（1.29%）、Layer Norm 10,752（0.02%）[^src-how-llm-understand-st]
- **骨干**：仅用 GPT2 的 3 层；训练 200 epoch，Adam lr $10^{-3}$，Huber Loss，6:2:2 划分，12 步预测 12 步[^src-how-llm-understand-st]

## 实验结果

6 个数据集：PEMS03/04/07/08（Caltrans PeMS 交通流，传感器数分别为 358/307/883/170，时间步数 26,208/16,992/28,224/17,856，5 分钟窗口聚合）、Electricity（321 客户，2012–2014 年逐小时 kWh）、ExchangeRate（8 国汇率，1990–2010 年逐日）[^src-how-llm-understand-st]。基线 13 个，含 PDFormer、DSTAGNN、FOGS、AGCRN 等。

- **PEMS07**（883 传感器）：STG-LLM MAE 19.82 / RMSE 33.06 / MAPE 8.51%，全指标优于 PDFormer（20.62/33.96/8.58）；PEMS04 MAE 18.14、PEMS08 MAE 13.78，均略逊于 PDFormer（18.31/13.58）但无需 DTW 矩阵、kShape 聚类等特征工程[^src-how-llm-understand-st]
- **Electricity**：MAE 214.11，优于 GraphWaveNet 233.72 / AGCRN 224.75 / STNorm 239.85——在无显式空间图的数据上同样成立[^src-how-llm-understand-st]
- **Few-shot**（PEMS04 全量微调 → PEMS08 迁移）：50 个样本（约 0.47% 数据）即接近传统深度方法全量训练性能；1000 样本（9.35%）超过除 PDFormer 外全部基线；2000 样本（18.69%）逼近 PDFormer 与全量训练[^src-how-llm-understand-st]
- **Prompt 有效性**：加时间 prompt 后 PEMS04 MAE 18.81 → 19.26（去掉变差），证明 LLM 预训练知识（周二工作日、晚高峰）可注入时空预测[^src-how-llm-understand-st]
- **Ablation**（PEMS04 MAE）：完整 18.14；去掉 LLM 21.94、去掉 Adapter 19.41、去掉 Position Embeddings 20.51、去掉 Tokenizer 26.91——tokenizer 是最关键组件[^src-how-llm-understand-st]

## 与相关工作的差异

与 [[source-st-llm]]（通过 positional encodings 让 LLM 空间感知、部分冻结 LLM 的路线）不同，STG-LLM 不依赖预定义图结构，用"节点即 token"的图 tokenizer 加双线性层 adapter；与 [[source-opencity]]（2–26M 参数、21 数据集预训练、面向零样本的基础模型路线，但需要路网邻接矩阵做 Laplacian 空间编码）相反，STG-LLM 逐数据集微调冻结 GPT2，且明确在 Electricity/ExchangeRate 等无图数据上验证泛化[^src-how-llm-understand-st]。

## 局限性

- 骨干仅 GPT2 3 层，参数效率高但未验证更大 LLM（GPT-3.5/Llama）是否带来进一步收益；论文仅声称可替换骨干[^src-how-llm-understand-st]
- Few-shot 实验限定在 PEMS04↔PEMS08 同域（均为加州交通流）迁移，未测试跨域（如交通→电力）few-shot[^src-how-llm-understand-st]
- 每个 token 承载完整 $L \times F$ 序列，节点数极多时 token 数线性增长，大网格场景的可扩展性未讨论[^src-how-llm-understand-st]

[^src-how-llm-understand-st]: [[source-how-llm-understand-st]]
