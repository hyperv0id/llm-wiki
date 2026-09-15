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
**Venue**: arXiv:2401.14192v2 (2024-05)

## 问题

LLM 只接受序列文本输入：把时空数据转成自然语言描述，PEMS07 一小时的交通数据（883 节点 × 12 步）就需至少 10,596 token，会撑爆上下文窗口；而现有时间序列 patch/token 化方法又捕获不了空间依赖。论文提出 STG-LLM，把瓶颈定位在 tokenization[^src-how-llm-understand-st]。

## 方法

- **STG-Tokenizer**：节点即 token，token 内承载该节点完整 $L \times F$ 历史序列（时间语义），token 间注意力承载空间语义，N 个节点只需 N 个 token。每个 token 拼接按最后时间步 $t$ 检索的 TD-Embedding 与 DW-Embedding（维度均设 64）：$T_e^i = T_g^i \| E_{td}^i \| E_{dw}^i$[^src-how-llm-understand-st]。
- **STG-Adapter**：一层线性编码 $T_q = W_1 T_e + b_1$ 对齐 LLM 维度 $D$，prompt token $T_p$ 与其拼接后喂给 LLM；一层线性解码 $\hat{X}_{t+1:t+P} = W_3((W_2 H + b_2) + T_e) + b_3$，含残差与偏置[^src-how-llm-understand-st]。
- **冻结策略**：冻结 GPT2 的多头注意力与 FFN，只微调 positional embeddings 和 layer norm；总参数 60,885,480，可训练 1,033,704（1.70%）——Tokenizer 18,880（0.03%）、Adapter 217,640（0.36%）、Position Embeddings 786,432（1.29%）、Layer Norm 10,752（0.02%）[^src-how-llm-understand-st]。
- **训练设置**：GPT2 用 3 层，6:2:2 划分，12 步预测 12 步，Adam lr $10^{-3}$，weight decay 0.05，Huber Loss，200 epoch（验证集 50 iteration 早停）[^src-how-llm-understand-st]。

## 证据

6 个数据集：PEMS03/04/07/08（Caltrans PeMS，传感器 358/307/883/170，时间步 26,208/16,992/28,224/17,856）、Electricity（321 客户，逐小时，26,304 步）、ExchangeRate（8 国逐日，7,588 步）[^src-how-llm-understand-st]。

与 PDFormer 逐项对比（Table 2）：PEMS04 MAE 18.14 < 18.31，STG-LLM 更优；PEMS08 MAE 13.78 > 13.58，略逊；PEMS07 19.82/33.06/8.51 对 19.83/32.87/8.53，互有胜负。论文自述是"超过多数基线、仅 PDFormer 接近"，且不依赖 DTW 矩阵、kShape 聚类等特征工程。Electricity 上 MAE 214.11 优于 GraphWaveNet 233.72 / AGCRN 224.75 / STNorm 239.85，在无显式图结构数据上同样成立[^src-how-llm-understand-st]。

Few-shot（PEMS04 全量微调 → PEMS08 迁移）：50 样本（约 0.47%）即接近传统深度方法；1000 样本（9.35%）超过除 PDFormer 外全部基线；2000 样本（18.69%）逼近 PDFormer 与全量训练。消融（PEMS04 MAE）：把 STG-Tokenizer 换成 GPT4TS 的 tokenizer 退化最重（18.14→26.91），其次去 LLM 21.94、去 Position Embeddings 20.51、去 Adapter 19.41。Prompt：注入时间提示使 PEMS04 MAE 19.26→18.81[^src-how-llm-understand-st]。

## 页面评述（非论文内容）

与 [[source-st-llm]] 的区别：STG-LLM 不做图结构嵌入进 LLM 的位置编码，而是"节点即 token"加双线性层 adapter；与 [[source-opencity]] 的基础模型路线（多数据集预训练、面向零样本）不同，STG-LLM 逐数据集微调冻结 GPT2，并在 Electricity/ExchangeRate 这类无图数据上验证了泛化。

## 局限性

论文承认：无 prompt 时性能下降，说明其收益依赖额外信息注入（页面评述）。骨干仅 GPT2 3 层，更大 LLM 是否带来收益未验证；few-shot 限于 PEMS04↔PEMS08 同域迁移，未测跨域。

[^src-how-llm-understand-st]: [[source-how-llm-understand-st]]
