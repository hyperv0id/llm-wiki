---
title: "Error-based Uncertainty Estimation"
type: technique
tags:
  - time-series-forecasting
  - uncertainty-estimation
  - error-prediction
  - auxiliary-loss
created: 2026-08-06
last_updated: 2026-08-06
source_count: 1
confidence: medium
status: active
---

# Error-based Uncertainty Estimation（误差代理不确定性估计）

**Error-based uncertainty estimation（误差代理不确定性估计）** 是 [[pir|PIR]] 失败识别组件采用的技术：用神经网络直接预测逐实例预测误差（MSE），以"预测的误差"作为不确定性估计，识别可能失效的预测实例[^src-pir]。

## 动机

论文指出两点阻碍[^src-pir]：

1. 点预测回归模型的输出直接由隐状态生成，生成前不存在 token 级分布，基于概率的常用不确定性量化方法难以直接适用；
2. 预测失败同时源于数据不确定性（如缺失值）与模型不确定性（如长尾模式欠拟合），而多数不确定性量化方法只覆盖后者。

论文因此放弃概率框架，改用数据驱动方式：用"预测误差"这一可获得真实值的监督信号量化不确定性——直觉是不确定性越高，误差越大[^src-pir]。

## 机制

1. **估计器**：δ = f_ue(x, ȳ, E)，带非线性激活的两层全连接网络；输入为输入序列 x、骨干中间预测 ȳ 与通道嵌入矩阵 E ∈ R^(N×d)（编码通道身份，帮助捕捉跨通道差异）[^src-pir]。
2. **辅助损失**：L_ue = (1/N)Σ||δ − ||ȳ−y||²||₁，以 MAE 约束 δ 对齐真实 MSE；与修订损失 L_pr 构成多任务目标 L = L_pr + λ·L_ue（λ = 1），与骨干端到端联合优化[^src-pir]。
3. **下游使用**：δ 经 σ(Linear(·)) 得到局部修订权重 α（δ 越大 α 越大），并与检索相似度 w 一起经 MLP 得到全局修订权重 β——不确定性直接决定修订强度（见 [[post-hoc-forecast-revision]]）[^src-pir]。

## 论文报告的证据

- **定性**：SparseTSF 在 Solar/Traffic 上的估计误差与真实误差曲线峰谷一致（图 3），论文据此认为估计器能可靠识别失败实例[^src-pir]。
- **定量**：附录 D 在 ETTm1 上扫描 λ（0 到 1，图 5），MSE 与 L_ue 的 R² 为 0.9067（[[patchtst|PatchTST]] 骨干）/ 0.7500（[[itransformer|iTransformer]] 骨干）；论文认为高相关性支持以预测误差作为不确定性代理[^src-pir]。

## 边界与对照

- **失败模式不解耦**：论文自述失败原因缺乏 ground truth 与度量，无法识别并解耦每个实例的具体失败原因，因此不显式定位失败模式，而是用误差代理统一覆盖[^src-pir]。
- **与分布式的区别**：生成式方法直接输出预测分布（[[generative-time-series-forecasting]]），共形预测提供有限样本覆盖保证（[[conformal-prediction]]）；误差代理是学习式点估计，不提供概率或保证[^src-pir]。
- **与状态空间不确定性的区别**：[[kalmannet-uncertainty-modeling|KalmanNet]] 从线性状态空间模型的滤波协方差得到不确定性，PIR 直接回归误差标量——两种不同的学习式路线。

## 相关页面

- [[pir]] — 框架总览
- [[post-hoc-forecast-revision]] — 识别结果的下游使用
- [[instance-level-variation]] — 误差来源与长尾现象
- [[conformal-prediction]] · [[generative-time-series-forecasting]] · [[kalmannet-uncertainty-modeling]] — 其他不确定性量化路线

[^src-pir]: [[source-pir]]
