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

## 核心问题与机制

针对 LLM 时序方法（OFA、Time-LLM、TEMPO-GPT）只沿时间轴建模、忽略空间维度的问题，论文提出 ST-LLM，把 attention 序列维从时间反转为空间：论文称 token 为某位置的各时间步，框架实现中历史数据 $X_P \in \mathbb{R}^{P\times N\times C}$ 被视为 **N 个空间位置的 token**（L227-228），GPT2（取 6 层，L683-684）对 N 个空间 token 做 self-attention 以捕获全局空间依赖（L459-461）。

三嵌入（式 2-6）：token embedding $E_P$ 用 1×1 pointwise 卷积；temporal embedding 在 day（$T_d$=48）与 week（$T_w$=7）两个分辨率做 absolute positional encoding，可学习投影后相加；spatial embedding 是 adaptive embedding $E_S=\sigma(W_s\cdot X_P+b_s)$，不依赖邻接矩阵。FConv 将三者拼接投影为 $H_F\in\mathbb{R}^{N\times 3D}$（式 7）。

**PFA（Partially Frozen Attention）**：前 F 层的 MHA 与 FFN 全冻结；最后 U 层解冻 MHA、FFN 仍冻结（LN 均可训练，式 8-10）。末端 RConv 输出未来 S 步，损失为误差项加 λ·L2 正则（式 12）；Ranger21 优化器、lr 0.001、batch 64、100 epochs（L680-686）。

## 证据

数据集：NYCTaxi（3500 万+ 行程、266 虚拟站点）与 CHBike（约 260 万 Citi Bike 订单、过滤后 250 站），均 2016-04-01 至 06-30、4368 个 30 分钟步（L623-631）；6:2:2 划分、P=S=12（L674-676）。基线以 Table II 列头为准：6 个 GNN + 3 个 attention + 4 个 LLM（含 LLAMA2 8 层）共 14 个；正文却称 "10 baselines" 且漏 STSGCN（L636-639）。

Table II（ST-LLM 四场景 MAE 均为最优）：NYCTaxi Pick-up MAE 5.29 / RMSE 9.42 / WAPE 20.03%（次优 LLAMA2 MAE 5.35）；Drop-off MAE 5.07 / RMSE 9.07；CHBike Pick-up MAE 1.99 / RMSE 3.08；Drop-off MAE 1.89 / RMSE 2.81 / WAPE 38.27%。正文自述"比 OFA 平均 MAE 提升 22.5%、比 LLAMA2 提升 20.8%"（L700-701），但按表内四场景 MAE 均值核算（ST-LLM 3.56 vs OFA 3.86、LLAMA2 3.78）仅约 8% 与 6%（不自洽，如实并列）。

**U 敏感性（§G）非单调**（L1468-1494）：NYCTaxi Pick-up 上随 U 增至 2 改善、超过 2 后退化（"this positive effect inverts when U exceeds 2 ... starts to degrade"），最优 U=2；CHBike Pick-up 最优 U=1；论文总结解冻更多层可能反而退化。图 3 纵轴刻度 2.070/2.010（L1193/L1225）非数据点。

**Few-shot**（仅 10% 训练数据，Table IV）：NYCTaxi Pick-up MAE 5.40 vs LLAMA2 5.81，降 7.06%（与表一致）；正文另称对 OFA 改进 9.15%、对 GATGPT/GCNGPT 平均改进 39.21%/7.80%（L1848-1855），按表核算均不自洽。**Zero-shot** 协议：只用 NYCTaxi 训练、不接触 CHBike 直接预测（L1951-1955）；Table V 含 6 组迁移，ST-LLM 六组 MAE 均最低，LLAMA2 次之，OFA 被论文评为"not a good zero-shot predictor"。

## 与 OpenCity / UrbanGPT 的定位差异（页面评述，跨源对比）

论文未讨论二者。OpenCity 不用 LLM，预训练后跨城市 zero-shot；UrbanGPT 用 LLaMA + 时空指令微调实现 unseen city zero-shot；ST-LLM 用 6 层 GPT2 逐数据集训练，其 "zero-shot" 只是 NYC 内部 taxi↔bike 与 pick-up↔drop-off 迁移，无跨城市实验。

## 局限（页面评述；原文无 limitations 章节，仅 conclusion + future work，L1936-1943）

- 仅 2 个 2016 年 NYC 数据集，无 PEMS 类公路数据，跨城市泛化未测试。
- 非基础模型路线，换城市/数据类型需重训；U、F 取值靠网格观察，论文未解释 U 最优值非单调现象的机制。

[^src-st-llm]: [[source-st-llm]]
