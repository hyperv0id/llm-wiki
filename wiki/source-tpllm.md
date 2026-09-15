---
title: "TPLLM: A Traffic Prediction Framework Based on Pretrained Large Language Models"
type: source-summary
tags:
  - llm
  - spatio-temporal
  - traffic-prediction
  - few-shot
  - lora
  - gpt-2
created: 2026-09-15
last_updated: 2026-09-15
source_count: 1
confidence: medium
status: active
---

# TPLLM: A Traffic Prediction Framework Based on Pretrained Large Language Models

**Authors**: Yilong Ren, Yue Chen, Shuai Liu, Boyue Wang, Haiyang Yu, Zhiyong Cui (北京航空航天大学，北京工业大学)
**Venue**: arXiv:2403.02221 (March 2024)

## 核心论点

深度学习交通预测模型的精度随训练数据量上升，但长时序交通数据采集与存储成本高，使数据稀缺地区难以训练。TPLLM 把多元时序交通数据重塑为类似 token embedding 的数值形式输入冻结的 GPT-2（D=768），仅用 LoRA 微调约 0.95% 参数，利用预训练 LLM 的 cross-modal knowledge transfer 与 few-shot 能力做交通流预测，面向全样本和小样本两种场景[^src-tpllm]。

## 方法机制

管线不含自然语言 prompt，全部为数值嵌入：输入 1 小时历史（T=12），预测未来 15/30/60 分钟（T'=3/6/12）[^src-tpllm]。

- **Graph embedding**：$\tilde{A}=A+I$，$GEF(X)=\mathrm{ReLU}(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}XW+b)$，从路网邻接矩阵提取空间特征
- **Sequence embedding**：$SEF(X)=\mathrm{Conv1d}_F(X)$，1-D CNN 提取序列自身特征
- **融合与输出**：$M=\mathrm{Linear}_D(\mathrm{LN}(\mathrm{ReLU}(GEF+SEF)))$，取最后一个特征通道得到 $N\times D$ 的 LLM 输入；LLM 输出经 $Y=\mathrm{ReLU}(\mathrm{Linear}_{T'}(H))$ 投影回预测步数，损失为 L1（MAE）
- **LoRA**：注入 GPT-2 每个 Transformer block 的 attention Query/Key，$h=W_0h_0+\frac{\alpha}{r}BCh_0$（$\alpha=32$，$r\in\{4,...,64\}$）；最优 $r=48$（PeMS08 few-shot 为 32），但 MAE 对 $r$ 不敏感，可取小值省算力[^src-tpllm]

## 实验结果

数据集：PeMS04（307 节点，16992 时间步，2018-01/02）、PeMS08（170 节点，17856 步，2016-07/08），按时序 6:2:2 划分；few-shot 训练集仅为全样本的 10%（6/20/20）。基线：LSTM、STGCN、ASTGCN、STSGCN[^src-tpllm]。

- **全样本**（1 小时内平均）：PeMS04 TPLLM MAE 19.53 vs ASTGCN 21.83，15 分钟 MAPE 12.04% vs 12.85%；PeMS08 TPLLM MAE 15.45 vs ASTGCN 18.33，MAPE 9.88% vs 11.64%，全部指标优于基线
- **Few-shot**：PeMS04 TPLLM 平均 MAE 23.68（较全样本 +4.15），ASTGCN 27.12（+5.29）；PeMS08 TPLLM 18.09（+2.64）vs ASTGCN 22.47（+4.14）——TPLLM 退化幅度最小
- **消融**（平均 MAE）：去 sequence embedding 退化最重（PeMS08 few-shot 27.86 vs 完整 18.09）；去 graph embedding 15.98 vs 15.45（PeMS08 全样本）；去 LoRA 23.76 vs 19.53（PeMS04 全样本），且 LoRA 移除后 LLM 完全冻结仍有一定零样本预测能力[^src-tpllm]

## 局限与定位

- 仅在两个加州 PeMS 流量数据集上验证单一任务（流量），未测速度/需求/缺失填补等下游任务
- 骨干为 GPT-2（1.24 亿参数），未对比 OpenCity 等时空基础模型或 Time-LLM 等同类时序-LLM 方法
- 后续工作方向是纳入更多交通影响因素的 embedding，以及更适配时空任务的 PEFT 技术[^src-tpllm]

与 [[source-st-llm]] 同属 2024 年"时序重塑为 token 输入 LLM"路线：ST-LLM 采用 partial frozen 策略解冻 LLM 的部分层微调，TPLLM 则完全冻结 GPT-2 骨干，只训 Q/K 上的 LoRA 低秩增量（0.95% 可训练参数），且不构造文本 prompt。

[^src-tpllm]: [[source-tpllm]]
