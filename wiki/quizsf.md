---
title: "QuiZSF"
type: technique
tags:
  - time-series-forecasting
  - retrieval-augmented-generation
  - zero-shot-forecasting
  - time-series-pre-trained-models
  - www-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# QuiZSF: Retrieval-Augmented Zero-Shot Time Series Forecasting

**QuiZSF**（Quick Zero-shot time-series Search and Forecasting）是 Ma, Zhou, Huang, Wang & Wang（USTC, WWW 2026）提出的检索增强零样本时序预测框架[^src-quizsf-zero-shot-forecasting-www26]。核心思想是将检索增强生成（RAG）从文本扩展到时间序列：从大规模时序数据库中检索与目标序列结构相似的历史序列，融合后输入预训练模型完成零样本预测[^src-quizsf-zero-shot-forecasting-www26]。

## 动机

现有时序预训练模型（TSPMs）存在两个关键限制[^src-quizsf-zero-shot-forecasting-www26]：

1. **无法动态整合新知识**：真实世界序列持续演化，TSPMs 若无高成本微调则难以吸收新知识[^src-quizsf-zero-shot-forecasting-www26]。
2. **缺乏检索和复用机制**：跨域时间序列常展现结构相似性（周期性、季节偏移、突变过渡），但 TSPMs 缺少检索和复用这些辅助模式的机制[^src-quizsf-zero-shot-forecasting-www26]。

## 框架三组件

QuiZSF 由三个主要组件构成[^src-quizsf-zero-shot-forecasting-www26]：

### 1. ChronoRAG Base (CRB) + HHTR

[[chronorag-base|ChronoRAG Base]] 是一个层次化树结构时序数据库，整合 27 个时序数据集覆盖 7 个领域（Web、Energy、Health、IoT、Nature、Transport、Environment），提供三个规模版本：CRB-Small（34M）、CRB-Medium（48M）、CRB-Large（143M time points）[^src-quizsf-zero-shot-forecasting-www26]。实验中选用 CRB-Medium 平衡精度与效率[^src-quizsf-zero-shot-forecasting-www26]。

**Hybrid and Hierarchical Time-series Retrieval (HHTR)** 策略结合域内局部原型匹配与全局原型比较[^src-quizsf-zero-shot-forecasting-www26]：
- 当目标域已知且存在于 CRB 中：Top-K = ρ·Top-K_local + (1-ρ)·Top-K_global
- 当域未知：仅全局检索 Top-K_global

距离度量使用复合相似度[^src-quizsf-zero-shot-forecasting-www26]：
$$\text{Sim}(X_T, X_i) = \cos(X_T, X_i) + \frac{1}{\text{dist}(X_T, X_i)}$$
结合余弦相似度（趋势对齐）与欧氏距离逆数（几何邻近）[^src-quizsf-zero-shot-forecasting-www26]。

### 2. Multi-grained Series Interaction Learner (MSIL)

MSIL 捕获目标序列与检索序列间的细粒度和粗粒度依赖[^src-quizsf-zero-shot-forecasting-www26]：

- **Interaction Pattern (P_int)**：通过逐元素乘积 + 非线性投影捕获细粒度依赖[^src-quizsf-zero-shot-forecasting-www26]。
- **Average Pattern (P_avg)**：通过均值池化与变换编码全局趋势[^src-quizsf-zero-shot-forecasting-www26]。
- **Multi-grained Cross-Attention**：目标序列为 Query，交互模式和平均模式为 Key/Value，融合后输出 R_fused[^src-quizsf-zero-shot-forecasting-www26]。

### 3. Model Cooperation Coherer (MCC)

MCC 是双分支适配器，将检索知识整合进两类 TSPMs[^src-quizsf-zero-shot-forecasting-www26]：

- **Numerical Coherer（Non-LLM TSPMs）**：残差连接融合归一化目标序列 T_norm 与 MSIL 融合表示 R_fused，输入数值预训练模型（实验中用 TTM-Base）[^src-quizsf-zero-shot-forecasting-www26]。
- **Language Coherer（LLM-based TSPMs）**：将 MSIL 输出（P_int, P_avg, T_norm）转化为结构化文本摘要，结合指令提示引导语言模型生成预测（实验中用 Time-LLM + LLaMA-7B）[^src-quizsf-zero-shot-forecasting-www26]。

## 实验设置

由于 Non-LLM 与 LLM-based TSPMs 输入模态不同，QuiZSF 分别设计两种零样本设置[^src-quizsf-zero-shot-forecasting-www26]：

- **QuiZSF_T**（多源泛化设置）：在多个源数据集上训练，直接应用到未见目标数据集。基座模型 TTM-Base，训练集 38.7M time points，评测在 ETT 和 Weather 上[^src-quizsf-zero-shot-forecasting-www26]。
- **QuiZSF_L**（单源迁移设置）：在单个源数据集训练，迁移到不同域。基座模型 Time-LLM + LLaMA-7B，遵循 TimeLLM 零样本实验设置[^src-quizsf-zero-shot-forecasting-www26]。

评测数据集：ETTh1, ETTh2, ETTm1, ETTm2, Weather，指标 MSE[^src-quizsf-zero-shot-forecasting-www26]。

## 主要结果

### QuiZSF_T（Table 2）

- 在零样本设置下 Top1 占 75%（ETTh1/ETTh2/ETTm2/Weather 的 Avg MSE 均最优或并列最优）[^src-quizsf-zero-shot-forecasting-www26]。
- 在 ETTh2 上 Avg MSE 0.345，优于 TTM-Base 0.347 和 Moirai 0.346[^src-quizsf-zero-shot-forecasting-www26]。
- 在 Weather 上全预测长度最优（Avg 0.231 vs TTM 0.239, Moirai 0.239）[^src-quizsf-zero-shot-forecasting-www26]。
- 在 ETTm1 上表现有限（Avg 0.395 vs Moirai 0.383），论文归因于细粒度分钟级序列与检索信息的对齐困难[^src-quizsf-zero-shot-forecasting-www26]。
- 论文报告 QuiZSF_T 在部分数据集上甚至超越 full-shot 模型（如 DLinear Avg ETTh2 0.559, PatchTST ETTh1 0.454）[^src-quizsf-zero-shot-forecasting-www26]。

### QuiZSF_L（Table 3）

- 在 8 个单源迁移设置中 7 个最优（87.5%）[^src-quizsf-zero-shot-forecasting-www26]。
- 例如 ETTh1→ETTh2 MSE 0.352（优于 Time-LLM 0.356），ETTh1→ETTm2 0.272（优于 Time-LLM 0.277）[^src-quizsf-zero-shot-forecasting-www26]。
- ETTh2→ETTh1 是唯一未取得最优的设置（0.535 vs Time-LLM 0.521）[^src-quizsf-zero-shot-forecasting-www26]。

## 消融实验

消融在 QuiZSF_L 上进行（Table 4, ETTm1→ETTh2 和 ETTm1→ETTm2）[^src-quizsf-zero-shot-forecasting-www26]：

- **去 RAG**（退化为 LLMTime）：性能下降 3.14%–5.34%，确认检索外部知识的重要性[^src-quizsf-zero-shot-forecasting-www26]。
- **去 MSIL**（用简单均值替代）：性能下降，证明多粒度交互学习的必要性[^src-quizsf-zero-shot-forecasting-www26]。
- **去 Coherer**（去掉结构化提示模板，直接拼接数值特征）：性能下降约 2%，说明模态适配对 LLM-based TSPMs 至关重要[^src-quizsf-zero-shot-forecasting-www26]。

## 效率分析

QuiZSF_T 在模型大小和 CPU 推理时间上保持竞争力，仅略逊于 TTM-Base（因引入轻量检索和交互模块）[^src-quizsf-zero-shot-forecasting-www26]。检索和特征提取依赖点积计算，效率高，不显著延长推理时间[^src-quizsf-zero-shot-forecasting-www26]。

## 超参数分析

在 ETTm1→ETTm2 上分析关键超参数[^src-quizsf-zero-shot-forecasting-www26]：
- CRB-Medium, K=8, ρ=60% 在精度与效率间取得最佳平衡[^src-quizsf-zero-shot-forecasting-www26]。

## 局限性

- 对短时细粒度数据集（如 ETTm1）效果有限，因粗粒度检索知识难以与目标序列的细粒度波动对齐[^src-quizsf-zero-shot-forecasting-www26]。
- 检索和交互模块引入额外参数和计算开销（论文称「minimal」）[^src-quizsf-zero-shot-forecasting-www26]。
- 仅在 ETT/Weather 数据集上评测，未覆盖更多领域[^src-quizsf-zero-shot-forecasting-www26]。

## 关联页面

- [[chronorag-base]] — ChronoRAG Base 时序数据库
- [[source-quizsf-zero-shot-forecasting-www26]] — 完整源文件摘要
- [[time-llm]] — QuiZSF_L 的基座模型
- [[timesfm]] — 对比基线 TSPM
- [[ratd]] — 先前的检索增强时序预测工作
- [[gtr]] — 全局时序检索模块
- [[pir]] — 检索增强预测后处理方法
- [[patchtst]] — 对比基线
- [[timesnet]] — 对比基线
- [[itransformer]] — 对比基线
- [[crossformer]] — 对比基线
- [[ltsf-linear|DLinear]] — 对比基线（DLinear 对比在 source-summary）
- [[fedformer]] — 对比基线
- [[tide]] — 对比基线
- [[retrieval-augmented-spatio-temporal-forecasting]] — RAG-for-STF 范式

[^src-quizsf-zero-shot-forecasting-www26]: [[source-quizsf-zero-shot-forecasting-www26]]
