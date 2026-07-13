---
title: "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
type: source-summary
tags:
  - transformer
  - exogenous
  - time-series
  - 2024
created: 2026-07-07
last_updated: 2026-07-13
source_count: 1
confidence: medium
status: active
---

# TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables

**Authors**: Yuxuan Wang*, Haixu Wu*, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu, Yunzhong Qiu, Jianmin Wang, Mingsheng Long (Tsinghua University, BNRist)

**Venue**: NeurIPS 2024 | **arXiv**: 2402.19072v4 | **Code**: [github.com/thuml/TimeXer](https://github.com/thuml/TimeXer)

## 核心贡献

TimeXer 首次系统性地解决了 **外生变量（exogenous variables）条件下的时间序列预测** 问题。与现有将多变量等同处理或忽略外生信息的范式不同，TimeXer 在无需修改 Transformer 架构的前提下，设计了一套层次化的嵌入与注意力机制，同时捕捉内生的时间依赖和外生到内生的跨变量关联。[^src-timexer]

## 关键设计

### 双粒度表示（Dual-Granularity Representation）

TimeXer 的核心洞见在于：内生变量和外生变量需要不同粒度的表示。[^src-timexer]

- **内生变量**：采用 **patch-wise** 表征。将内生产出时间序列分割为不重叠的 patch，每个 patch 投影为一个 temporal token，保留局部的语义信息和时序变化细节，与 [[source-patchtst|PatchTST]] 的 patch 化思路一脉相承。
- **外生变量**：采用 **variate-wise** 表征。每个外生变量被压缩为一个全局的 variate token，以适应缺失值、时间偏移、频率不一致和不等长等现实场景中的不规则性。patch 级的细粒度表示对于外生变量不仅带来高昂的计算成本，还会引入不必要的噪声。

### 内生全局 Token（Global Endogenous Token）

每个内生序列引入一个 **可学习的全局 token**，作为桥梁连接内生 patch token 与外生 variate token。[^src-timexer] 其双向交互机制包括：

1. **Patch-to-Global**：全局 token 关注所有内生 patch，聚合全序列的 patch 级信息。
2. **Global-to-Patch**：每个内生 patch token 关注全局 token，从全局视角接收跨变量相关性。
3. **Exogenous-to-Endogenous Cross-Attention**：以内生全局 token 为 Query、外生 variate token 为 Key/Value，通过交叉注意力将外生变量信息选择性传播到对应的内生 patch。

### 层级注意力机制

- **内生自注意力（Endogenous Self-Attention）**：patch token 与全局 token 拼接后送入标准 Transformer 自注意力层，同时捕捉 patch 间时序依赖和 patch-全局关系。
- **外生-内生交叉注意力（Exogenous-to-Endogenous Cross-Attention）**：以内生 query 和外生 key/value 建模两类变量间的自适应关联。

## 实验结果

在 **12 个真实世界基准数据集**（涵盖电力价格（PJM/BE/FR/DE/EPF）、电力负载（ETT）、气象、交通等）上达到一致 SOTA。[^src-timexer]

关键结果：
- 在 PJM 电力价格数据集上，MSE 达 **0.090**，显著优于 PatchTST (0.480)、iTransformer (0.520) 等基线。
- 在 EPF 基准上，MSE 平均改善超 **10%** 以对上最强基线。
- 在零样本跨数据集迁移中表现优异，验证了方法的泛化性。
- 对不规则外生变量（缺失值、时间偏移、频率不匹配、长度差异）具有鲁棒性。

## 相关链接

- [[source-exost|ExoST]] — 同期外生变量建模框架，采用 Select-then-Balance 范式
- [[source-patchtst|PatchTST]] — Patch 化时序预测奠基工作
- [[source-dits|DiTS]] — 多模态扩散 Transformer 时序预测，采用类似双流设计
- [[source-kite|KITE]] — 外生条件下的概率流匹配（HCM + KGC + CFG），确定性外生基准上对比 TimeXer
- [[source-gcgnet|GCGNet]] — 将 TimeXer 归为 temporal→channel 两步策略，提出图一致联合相关生成网络 (ICLR 2026)
- [[source-crosslinear|CrossLinear]] — KDD 2025 Linear 外生模型：1D conv 交叉相关嵌入 + patch/linear head，同协议 many-to-one 基准上与 TimeXer 对照且更轻量
- [[source-nbeatsx|NBEATSx]] — 早期可解释外生神经基扩展（IJF 2022 EPF）；TimeXer 在同族 EPF 市场上用 Transformer 外生融合刷新
- [[source-tft|TFT]] — 早期可解释 multi-horizon Transformer（static/known/observed + 分位数，2020）；TimeXer 更聚焦 endo/exo token 粒度与 irregular 外生

[^src-timexer]: [[source-timexer]]
