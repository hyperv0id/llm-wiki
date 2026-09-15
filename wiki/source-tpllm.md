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

**Authors**: Yilong Ren, Yue Chen, Shuai Liu, Boyue Wang, Haiyang Yu, Zhiyong Cui
**Venue**: arXiv:2403.02221v2, 2024-03-18

## 问题

深度学习交通预测模型精度依赖大量训练数据，而交通数据采集与保存成本高（原文："Due to the high cost of data collection and preservation, few-shot traffic prediction tasks represent common real-world situations"）[^src-tpllm]。论文提出 TPLLM：数值化路网数据输入冻结的 GPT-2（D=768），仅用 LoRA 微调约 0.95% 的可训练参数，利用预训练 LLM 的 cross-modal knowledge transfer 与 few-shot 能力做交通流预测[^src-tpllm]。

## 方法机制

管线不含自然语言 prompt，输入全为数值嵌入：原文将单传感器 T 时段历史序列视为 word、全路网数据视为 sentence（"Based on the similarity between time-series traffic data and natural language, we consider the historical data sequence of a single sensor during a period T as a word"）[^src-tpllm]。

- **Graph embedding**：$\tilde{A}=A+I$，$GEF(X)=\mathrm{ReLU}(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}XW+b)$，GCN 从邻接矩阵提取空间特征[^src-tpllm]
- **Sequence embedding**：$SEF(X)=\mathrm{Conv1d}_F(X)$，1-D CNN 提取序列自身特征[^src-tpllm]
- **融合与输出**：$M=\mathrm{Linear}_D(\mathrm{LN}(\mathrm{ReLU}(GEF+SEF)))$，取 $M$ 最后一个特征通道得到 $N\times D$ 的 LLM 输入；输出 $Y=\mathrm{ReLU}(\mathrm{Linear}_{T'}(H))$；损失取稳健 L1（MAE）以应对传感器故障产生的离群值[^src-tpllm]
- **LoRA**：注入每个 Transformer block 的 attention Query/Key，$h=W_0h_0+\frac{\alpha}{r}BCh_0$（$B$ 零初始化、$C$ 高斯初始化），$\alpha=32$；rank 网格为 $r=4/8/16/32/48/64$[^src-tpllm]

## 证据

数据集：PeMS04（307 节点、16992 时间步，2018-01-01 至 02-28）、PeMS08（170 节点、17856 步，2016-07-01 至 08-31），5 分钟粒度，三特征仅取周期性最明显的流量[^src-tpllm]。输入 1 小时历史 $T=12$，预测 15/30/60 分钟（$T'=3/6/12$）[^src-tpllm]。基线：LSTM、STGCN、ASTGCN、STSGCN[^src-tpllm]。

- **全样本**（1 小时内平均 MAE）：PeMS04 TPLLM 19.53 vs ASTGCN 21.83；PeMS08 15.45 vs 18.33；平均 MAPE 9.88% vs 11.64%，全指标优于基线（Table III）[^src-tpllm]
- **Few-shot**：训练集仅为全样本训练集的 10%，验证/测试与全样本相同，占全数据 6%/20%/20%。平均 MAE：PeMS04 TPLLM 23.68（+4.15）vs ASTGCN 27.12（+5.29）；PeMS08 18.09（+2.64）vs 22.47（+4.14）——TPLLM 退化最小（Table IV）[^src-tpllm]
- **rank 敏感性**：三个实验最优 $r=48$，PeMS08 few-shot 为 32，但 MAE 曲线整体平坦，论文推测可取小 $r$ 省算力[^src-tpllm]。
- **消融**（平均 MAE）：去 sequence embedding 退化最重，PeMS08 few-shot 27.86 vs 完整 18.09；去 graph embedding（PeMS08 全样本）15.98 vs 15.45；去 LoRA（PeMS04 全样本）23.76 vs 19.53；LoRA 移除后 LLM 完全冻结仍有一定零样本能力（论文原话 "which reflects some extent the zero-shot learning capability"）[^src-tpllm]

## 范围与局限

页面评述：仅验证两个加州 PeMS 数据集的流量单一任务，未测速度/需求/缺失填补，也未实验对比同类时序-LLM 或时空基础模型[^src-tpllm]。原文后续方向：设计纳入更多交通影响因素的 embedding、探索更适配时空预测的 PEFT[^src-tpllm]。

页面评述：与 [[source-st-llm]] 同属 2024 年"时序重塑为 token 输入 LLM"路线；ST-LLM 的 PFA 是前若干层全冻结、后若干层仅解冻多头注意力（FFN 仍冻结），TPLLM 则完全冻结骨干、只训 Q/K 的 LoRA 低秩增量，且不构造文本 prompt[^src-tpllm]。

[^src-tpllm]: [[source-tpllm]]
