---
title: "Spatial-Temporal Large Language Model for Traffic Prediction (ST-LLM)"
type: source-summary
tags:
  - spatio-temporal
  - llm
  - traffic-prediction
  - gpt2
  - parameter-efficient-finetuning
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# ST-LLM: Spatial-Temporal Large Language Model for Traffic Prediction

**Authors**: Chenxi Liu, Sun Yang, Qianxiong Xu, Zhishuai Li, Cheng Long, Ziyue Li, Rui Zhao (NTU S-Lab, PKU, SenseTime, U. Cologne)
**Venue**: MDM 2024
**Code**: https://github.com/ChenxiLiu-HNU/ST-LLM

## 核心问题与动机

LLM 时间序列方法（OFA、Time-LLM、TEMPO-GPT）只沿时间轴建模，忽略交通数据的空间维度；而传统注意力/GNN 模型结构日趋复杂，精度提升却放缓[^src-st-llm]。ST-LLM 的做法：把每个地点的每个时间步当作一个 token——N 个站点 → N 个 token，把 LLM 的 attention 序列维度从传统的时间轴**反转为空间轴**，用 GPT2 (6 层) 直接对 N 个空间 token 做 self-attention 以捕获全局空间依赖[^src-st-llm]。没有文本 prompt、没有指令微调——纯数值嵌入。

## 方法机制

三个嵌入 + 融合卷积：token embedding 经 1×1 pointwise convolution 得 $E_P \in \mathbb{R}^{N \times D}$；temporal embedding 在 day ($T_d$=48，30 分钟粒度) 和 week ($T_w$=7) 两个分辨率做 absolute positional encoding，经可学习 $W_{day}$、$W_{week}$ 后相加得 $E_T$；spatial embedding 用可学习的 adaptive embedding $E_S = \sigma(W_s \cdot X_P + b_s)$，不依赖邻接矩阵[^src-st-llm]。Fusion convolution 把三者拼接投影为 $H_F \in \mathbb{R}^{N \times 3D}$ 输入 LLM[^src-st-llm]。

**PFA (Partially Frozen Attention)**：GPT2 前 F 层的 MHA 和 FFN 全部冻结以保留预训练知识，最后 U 层的 FFN 仍冻结、但 **MHA 解冻**——论点是一层中预训练知识主要在 FFN，而 attention 负责适配时空依赖[^src-st-llm]。沿用 GPT2 pre-LN 结构加一层 learnable positional encoding；末端 regression convolution (RConv) 输出未来 S 步预测，损失为 L1 + λ·L2 正则，Ranger21 优化器 (lr=0.001, batch 64, 100 epochs)[^src-st-llm]。消融显示 U 从 0 增到 6 时误差持续下降（MAE 2.070→2.010），说明解冻越多 attention 收益越大[^src-st-llm]。

## 实验结果

数据集仅两个 NYC 数据集：NYCTaxi（3500 万次行程，266 个虚拟站点）和 CHBike（260 万次 Citi Bike 订单，250 个站点），均为 2016-04-01 至 06-30、4368 个 30 分钟时间步，6:2:2 划分，P=S=12[^src-st-llm]。对比 10 个传统模型（DCRNN、STGCN、GWN、AGCRN、STG-NCDE、DGCRN、ASTGCN、GMAN、ASTGNN、STSGCN）和 4 个 LLM 基线（OFA、GATGPT、GCNGPT、LLAMA2 7B, 8 层）[^src-st-llm]。

**全量训练**（Table II）：NYCTaxi Pick-up MAE 5.29 / RMSE 9.42 / MAPE 33.55% / WAPE 20.03%（次优 LLAMA2 MAE 5.35、GMAN WAPE 20.42%）；Drop-off MAE 5.07 / RMSE 9.07；CHBike Pick-up MAE 1.99 / RMSE 3.08；Drop-off MAE 1.89 / RMSE 2.81 / MAPE 49.50% / WAPE 38.27%[^src-st-llm]。相对 OFA 平均 MAE 降低 22.5%，相对 LLAMA2 降低 20.8%[^src-st-llm]。**Few-shot**（仅 10% 训练数据）：NYCTaxi Pick-up 上比 LLAMA2 MAE 低 7.06%，CHBike Drop-off 上比 OFA 低 9.15%；比 GATGPT 平均低 39.21%、比 GCNGPT 低 7.80%[^src-st-llm]。**Zero-shot** 定义为域间迁移（只用 NYCTaxi 训练、不接触 CHBike 直接预测），ST-LLM 在 intra-domain（pick-up→drop-off）和 inter-domain（NYCTaxi→CHBike）迁移中误差均最低，LLAMA2 次之，OFA 最差[^src-st-llm]。

## 与 OpenCity / UrbanGPT 的定位差异

三者路径完全不同：OpenCity 不用 LLM，在 21 个数据集上预训练 foundation model，实现跨城市 zero-shot；UrbanGPT 用 LLaMA + 时空指令微调，靠文本指令实现 unseen city/time zero-shot；ST-LLM 用最小的 GPT2（6 层），每个数据集单独全量微调（80% 训练数据），其"zero-shot"只是 NYC 内部的 pick-up→drop-off / taxi→bike 迁移，没有跨城市泛化实验[^src-st-llm]。

## 局限

- 只在 2 个 2016 年 NYC 数据集上验证，没有 PEMS 公路速度/流量数据集，跨城市泛化未测试[^src-st-llm]
- 非基础模型：换城市/换数据类型需重新微调，与 OpenCity 的 zero-shot 目标相反[^src-st-llm]
- 纯数值嵌入，未利用 LLM 的语言知识和文本上下文（对比 UrbanGPT 的指令微调路线）[^src-st-llm]
- LLAMA2 7B 只用 8 层、GPT2 只用 6 层的截断策略缺少分析；U 层数选择在 0-6 内单调变好但未解释上限[^src-st-llm]

[^src-st-llm]: [[source-st-llm]]
