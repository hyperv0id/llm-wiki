---
title: "LagLLM: LLM-empowered lead–lag dependency learning for spatial-temporal time series forecasting"
type: source-summary
tags:
  - traffic-forecasting
  - air-quality
  - lead-lag-dependency
  - llm-for-time-series
  - graph-learning
  - icml-2026
created: 2026-09-05
last_updated: 2026-09-05
source_count: 0
confidence: medium
status: active
---

# LagLLM: LLM-empowered lead–lag dependency learning for spatial-temporal time series forecasting

**Wu, Zhou, Shang & Chen (2026), ICML 2026, PMLR 306**

完整论文（11 页正文）：`raw/lagllm-icml2026.pdf`（pdftotext 抽取全文阅读；首页脚注 "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026"，camera-ready 版式，venue 从原文核实）。代码：https://github.com/w2obin/LagLLM-2026 。四位作者均属浙江大学计算机科学与技术学院暨区块链与数据安全全国重点实验室，通讯作者 Ling Chen。

## 核心论题

论文提出 LagLLM，自称首个显式建模 lead–lag 依赖的 LLM 赋能时空预测框架，统一数据驱动动力学建模与知识驱动语义推理。论文将 lead–lag 依赖定义为跨空间与时间的依赖：位置 $i$ 在 $t_1$ 时刻的观测依赖位置 $j$ 在 $t_2$ 时刻的观测，时滞 $\tau_{ij}=|t_1-t_2|$；空间依赖（$t_1=t_2$）与时间依赖（$i=j$）是其一维退化形式。论文指出两个应用 LLM 的非平凡障碍：(1) lead–lag 先验稀缺——时空数据集几乎不带"谁 lead 谁 lag"的文本标注；(2) LLM 缺乏拓扑感知——预训练语料不含空间结构与方向性传播。

## 方法

四个模块：Spatial-Temporal Tokenizer（节点序列 patch 化，可学习分配矩阵 $S$ 将 $N$ 节点聚为 $N_g$ 组）；Hybrid Lead-Lag Graph Construction（数据驱动图 $A_D=\mathrm{softmax}(E_F E_F^\top)$ 由空间/时间嵌入内积构成，经 spatial mask $M_S=SA_S S^\top$ 与 frozen LLM 判出的 semantic mask $M_K$ 精炼，$A_L=A_D\otimes(M_S+M_K)$，其中 lead–lag prompt 四组件为 Task Instruction / Time Series Statistics / Lead-Lag Criteria / Question）；Structural Token Sorting（特征向量中心度 + 时间位置初始化重要性，lead–lag 图上消息传递后按分数重排 token，lead 在前）；GPT-2 骨干（layer norm 与 position embedding 全更新，注意力层 LoRA，FFN 冻结）。

## 实验

8 个数据集（PEMS03/04/07/08 流量、METR-LA/PEMS-BAY 速度、NYCTaxi 出租车需求、ChinaAQI 空气质量），对照 22 个基线（15 个 STGNN/Attention + 7 个预训练模型）。论文报告：27 例中 22 例最优，MAE 平均超最强基线 STD-PLM 1.76%；ChinaAQI RMSE 超 CrossST 2.27%；PEMS07 5% 数据 few-shot 下 MAPE 提升 17.20%。消融（表 5）：PEMS08 上去掉整个 lead-lag 图 MAE 13.21→14.14，退化最大。效率（表 7）：36.24 s/epoch vs STD-PLM 16.78，但收敛更快（约 80 vs 120 epoch）。

## 论文自述与课程评估

- **论文自述**：METR-LA 上相对较弱——交通速度由局地相邻模式主导，跨区 lead–lag 效应有限，MegaCRN 更适配；每 epoch 计算开销高于 STD-PLM；未来工作包括更丰富元数据（道路容量、POI、气象）与 semantic mask 缓存、蒸馏 LLM。
- **本课程评估**：实验未在论文内给出与 GNN 基线的训练时间对比（表 7 仅对比 LLM-based 方法），"每 epoch 更贵"的完整代价谱系需读者自行补齐；semantic mask 随输入样本动态变化，意味着推理时每次前向都要调用 LLM，论文未单独量化这一推理时开销；消融只在 PEMS08 与 ChinaAQI 两个数据集上做，代表性有限。

## 关键术语

- **Lead–lag dependency**：跨空间-时间的延迟依赖，$\tau_{ij}$ 为时滞
- **HLLG（Hybrid Lead-Lag Graph Construction）**：数据驱动图 + 空间/语义双掩码精炼
- **STS（Structural Token Sorting）**：按图上重要性重排 token，对齐 LLM 自回归注意力
- **Semantic mask $M_K$**：frozen LLM 按提示判断组间是否存在稳定可解释 lead–lag 依赖所得的 $[0,1]$ 掩码，随输入动态变化
