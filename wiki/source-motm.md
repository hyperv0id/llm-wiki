---
title: "MoTM: Towards a Foundation Model for Time Series Imputation based on Continuous Modeling"
type: source-summary
tags:
  - time-series
  - data-imputation
  - foundation-model
  - zero-shot
  - implicit-neural-representation
  - continuous-time
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# MoTM: Towards a Foundation Model for Time Series Imputation based on Continuous Modeling

**作者**：Etienne Le Naour, Tahar Nabil, Ghislain Agoua（EDF R&D）。**发表**：第 10 届 AALTD 工作坊（Advanced Analytics and Learning on Temporal Data），ECML 2025，**口头报告**；arXiv:2507.13207[^src-motm]。这是 [[motm|MoTM]] 模型的**原始/主要来源**（区别于评估它的 [[source-time-indexed-imputation|TMLR 2026 基准]]）。

## 核心问题

时间序列基础模型近年蓬勃发展，但**几乎都聚焦预测**，跨域缺失值**插补**仍被严重低估[^src-motm]。已有的插补基础模型尝试（NuwaTS、MOMENT）依赖**固定长度输入段**，无法处理不规则采样或变分辨率数据，难以利用跨数据集的共享模式（如周期性）[^src-motm]。连续时间建模（尤其隐式神经表示 INR）是有前途的方向，但其代表 **TimeFlow（Le Naour et al. 2024）虽在单一分布内表现强，却难以跨分布泛化**[^src-motm]。

## 方法：MoTM = TimeFlow 基 + ridge 编排器

核心思想："**新序列是已见模式的混合**"。MoTM 分三步[^src-motm]：

1. **预训练**：在 $N_{train}$ 个不同数据集上各训练一个 **TimeFlow（生成式 INR，由超网络 + 隐编码调制）**，构成一组**捕获各域特定动态的基**。
2. **推理步骤 1 — 适配基**：对新目标序列，为每个基模型用**少量内循环优化步**（meta-learning）拟合一个潜编码 $z^{(i,j)*}$，得到一组**调制 INR**，每个从自身视角重建序列。
3. **推理步骤 2 — 拟合编排器**：从每个调制 INR 提取隐藏表示（最后隐层），拼接成共享特征空间 $R_{obs}$，再**在观测上下文上拟合一个 ridge 回归**线性组合这些特征（闭式解，可扩展）；对任意目标时刻只需构造 $R_{target}$ 并乘以 $W^*$[^src-motm]。

由连续时间公式天然支持：处理任意缺失模式、不同采样率（10min/30min/1h/2h）、不规则/未对齐时间戳，以及无需重训的 OOD 推理[^src-motm]。

## 贡献与结果

- **能力组合首创**：论文声称 MoTM 是首个同时满足"插补任意缺失模式 + 原生支持不同采样率 + 可做 OOD 推理"三者的模型[^src-motm]。
- **合成实验**：在仅含日周期（ks1D）与周周期（ks1W）的两个数据集上预训练，对**同时含日+周周期且 15min 采样**的未见 ks1D1W 数据集，MoTM 零样本插补 MAE 比单个 TimeFlow 降低约 **75%**，证明超越记忆的泛化[^src-motm]。
- **真实数据**：基于 Electricity/Solar/SpanishW-T 三个 TimeFlow 训练；在 OOD 的 Traffic/ETTh1/ETTh2/Weather/SpanishE 上，零样本超越 MOMENT（平均 40.3%）与监督 TimeFlow（10.2%）、BRITS（24.0%），并在 ID 上略胜 SAITS（12.6%）[^src-motm]。
- **效率**：H100 上对 Traffic 插补 83k 段（长 672）仅 **61 秒（约 0.7ms/段）**，而 SAITS 需重训约 3 小时 16 分[^src-motm]。

## 局限

收益在数据集间**不均匀**（如 Solar 提升有限）；**最优模型组合的选择仍是开放问题**；监督 SAITS 在若干设置（尤其 OOD 点缺失）仍优于 MoTM；编排器（ridge）较简单，作者提出未来用更大统一数据库与更具表达力的编排机制改进[^src-motm]。

## 关联页面

- [[motm]] — 本文提出的模型实体页（详述方法与基准结果）
- [[motm-ridge-orchestrator]] — MoTM 的 ridge 编排器机制
- [[time-indexed-foundation-model]] — MoTM 所属的连续时间零样本插补范式
- [[tabpfn-ts]] — 互逆设计的另一时间索引插补模型
- [[source-time-indexed-imputation]] — 评估 MoTM 的 TMLR 2026 基准（独立第二来源）
- [[source-nuwats]] — 本文作为对比基线的固定段插补基础模型

[^src-motm]: [[source-motm]]
