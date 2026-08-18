---
title: "QuiZSF: A Retrieval-Augmented Framework for Zero-Shot Time Series Forecasting (WWW 2026)"
type: source-summary
tags:
  - time-series-forecasting
  - retrieval-augmented-generation
  - zero-shot-forecasting
  - time-series-pre-trained-models
  - www-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: low
status: active
---

# QuiZSF: A Retrieval-Augmented Framework for Zero-Shot Time Series Forecasting

**来源**：Shichao Ma, Zhengyang Zhou, Qihe Huang, Binwu Wang, Yang Wang (USTC). *QuiZSF: A Retrieval-Augmented Framework for Zero-Shot Time Series Forecasting*. WWW 2026. (DOI: 10.1145/3774904.3792141)

## 核心论点

论文提出 [[quizsf|QuiZSF]]，将检索增强生成（RAG）从 NLP 扩展到时间序列预测领域[^src-quizsf-zero-shot-forecasting-www26]。现有时序预训练模型（TSPMs）虽在零样本预测中展现潜力，但面临两大限制：无法高效整合新知识（需高成本微调）、缺乏跨域结构相似性的检索复用机制[^src-quizsf-zero-shot-forecasting-www26]。QuiZSF 通过构建大规模时序数据库、多粒度交互学习和模态适配三组件，使模型能主动检索并利用相似序列的辅助知识，降低零样本场景中的幻觉[^src-quizsf-zero-shot-forecasting-www26]。

## 方法

QuiZSF 由三个组件构成[^src-quizsf-zero-shot-forecasting-www26]：

1. **[[chronorag-base|ChronoRAG Base (CRB)]]**：层次化树结构时序数据库，整合 27 个数据集覆盖 7 个领域（Web/Energy/Health/IoT/Nature/Transport/Environment），提供 34M/48M/143M 三档规模。配合 Hybrid and Hierarchical Time-series Retrieval (HHTR) 策略，结合域内局部匹配与全局原型比较，使用余弦相似度+欧氏距离逆数的复合度量[^src-quizsf-zero-shot-forecasting-www26]。树结构通过 k-means 聚类构建层次索引（每簇最多 256 条序列），支持动态更新和局部重聚类[^src-quizsf-zero-shot-forecasting-www26]。

2. **Multi-grained Series Interaction Learner (MSIL)**：计算两种互补模式——Interaction Pattern（逐元素乘积+非线性投影，捕获细粒度依赖）和 Average Pattern（均值池化，编码全局趋势），通过多粒度交叉注意力以目标序列为 Query 融合输出[^src-quizsf-zero-shot-forecasting-www26]。

3. **Model Cooperation Coherer (MCC)**：双分支适配器，Numerical Coherer 用残差连接融合表示并输入 Non-LLM TSPMs（TTM-Base）；Language Coherer 将数值表示转化为结构化文本提示输入 LLM-based TSPMs（[[time-llm|Time-LLM]] + LLaMA-7B）[^src-quizsf-zero-shot-forecasting-www26]。

## 实验设置

针对两类 TSPMs 的不同输入模态，分别设计两种零样本设置[^src-quizsf-zero-shot-forecasting-www26]：
- **QuiZSF_T**（Non-LLM，多源泛化）：基座 TTM-Base，训练集 38.7M time points，评测 ETT/Weather，与 TTM/Moirai/TimesFM 及 full-shot 模型（iTransformer/Crossformer/DLinear/TimesNet/PatchTST/TiDE/FEDformer）对比[^src-quizsf-zero-shot-forecasting-www26]。
- **QuiZSF_L**（LLM-based，单源迁移）：基座 Time-LLM + LLaMA-7B，遵循 TimeLLM 零样本设置，与 Time-LLM/LLMTime/GPT4TS 及 DLinear/PatchTST/TimesNet/Autoformer 对比[^src-quizsf-zero-shot-forecasting-www26]。

## 结果

- **QuiZSF_T**：零样本设置下 Top1 占 75%（ETTh1/ETTh2/ETTm2/Weather Avg MSE 最优），部分数据集超越 full-shot 模型。Weather 全长度最优（Avg 0.231 vs TTM 0.239）。ETTm1 表现有限（Avg 0.395 vs Moirai 0.383），论文归因于细粒度数据与粗粒度检索知识的对齐困难[^src-quizsf-zero-shot-forecasting-www26]。
- **QuiZSF_L**：8 个设置中 7 个最优（87.5%）。ETTh1→ETTh2 0.352（优于 Time-LLM 0.356）。唯一非最优为 ETTh2→ETTh1（0.535 vs Time-LLM 0.521）[^src-quizsf-zero-shot-forecasting-www26]。
- **消融**：去 RAG 下降 3.14%–5.34%，去 MSIL 下降，去 Coherer 下降约 2%[^src-quizsf-zero-shot-forecasting-www26]。
- **效率**：模型大小和推理时间具竞争力，检索模块引入的开销可控[^src-quizsf-zero-shot-forecasting-www26]。
- **超参数**：CRB-Medium, K=8, ρ=60% 为最佳配置[^src-quizsf-zero-shot-forecasting-www26]。

## 贡献与局限

贡献：首次将 RAG 系统性扩展到时序预测并构建 CRB；设计 HHTR 混合检索与 MSIL 多粒度交互学习；提出 MCC 双分支适配器同时支持 Non-LLM 与 LLM-based TSPMs；在 5 个基准上取得 SOTA，Non-LLM 设定 75% Top1，LLM 设定 87.5% Top1[^src-quizsf-zero-shot-forecasting-www26]。
局限：短时细粒度数据集上检索知识与目标波动对齐困难；评测仅覆盖 ETT/Weather；检索与交互引入额外开销[^src-quizsf-zero-shot-forecasting-www26]。

[^src-quizsf-zero-shot-forecasting-www26]: [[source-quizsf-zero-shot-forecasting-www26]]
