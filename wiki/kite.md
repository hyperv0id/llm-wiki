---
title: "KITE"
type: entity
tags:
  - flow-matching
  - probabilistic-forecasting
  - exogenous
  - classifier-free-guidance
  - manifold
  - icml-2026
created: 2026-07-12
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# KITE

**KITE**（Knowledge-GuIded Probabilistic Modeling for Time Series Forecasting with Exogenous Variables）是华东师范大学 Decision Intelligence 组提出的端到端框架，发表于 ICML 2026。它以 [[flow-matching|Flow Matching]] 为生成骨干，把历史/未来外生变量作为条件，输出内生变量的预测分布，而不是单点估计。[^src-kite]

代码：<https://github.com/decisionintelligence/KITE>

## 要解决什么

外生条件下的概率预测同时撞上两堵墙：[^src-kite]

1. **拓扑落差**：从上下文无关的高斯源出发，到高度局域的协变量条件目标，传输路径长而曲折，采样慢且保真度差。
2. **伪相关放大**：纯数据驱动的协变量注意力容易学到训练集里的虚假相关；在迭代生成中误差沿轨迹传播放大，OOD 关系漂移时更脆。

## 三块积木

| 模块 | 角色 | 一句话 |
|------|------|--------|
| [[history-conditional-manifold\|HCM]] | 源分布 | 用历史内生学 \(\mu,\sigma,\delta\)，把起点搬到目标附近 |
| [[knowledge-guided-conditioning\|KGC]] | 条件化 | 统计先验（Pearson/Granger）双线性调制内生–外生注意力 |
| [[classifier-free-guidance\|CFG]] | 可控性 | 联合训练有/无条件速度场，推理用 \(\gamma\) 调节外生强度 |

三者如何串起来见 [[kite-manifold-guidance-chain]]。[^src-kite]

## 任务形式

给定历史内生 \(X_{\text{endo}}\in\mathbb{R}^{N\times T}\)、历史外生 \(X_{\text{exo}}\in\mathbb{R}^{D\times T}\)、未来外生 \(Y_{\text{exo}}\in\mathbb{R}^{D\times F}\)，建模[^src-kite]

\[
p_\theta(Y_{\text{endo}} \mid X_{\text{endo}}, X_{\text{exo}}, Y_{\text{exo}}).
\]

路径：\(Y_s = s\cdot Y_{\text{endo}} + (1-s)\cdot Y_0\)，目标速度 \(\hat v_s = Y_{\text{endo}} - Y_0\)，网络 \(v_\theta(Y_s,s,c)\) 回归该场。[^src-kite]

## 结果速览

- 12 个确定性外生基准（含 EPF 五市场、能源、水库、风电、ETT 等）：有未来外生时 MSE 相对 CrossLinear 降 16.1%，MAE 相对 TFT 降 9.2%。[^src-kite]
- 8 个 ProbTS 概率基准：CRPS 相对 \(K^2\)VAE 降 5.7%、相对 CSDI 降 11.4%。[^src-kite]
- 消融显示 HCM、KGC、CFG 协同增益最大；效率上约 2.3M 参数，在 Weather 长预测设定中推理速度领先多数概率基线。[^src-kite]

## 与邻近工作的位置

- 相对 [[timexer|TimeXer]] / [[source-exotst|ExoTST]]：从确定性外生融合推进到概率生成 + 可控条件强度。[^src-kite]
- 相对 [[tsflow|TSFlow]]：同样用信息源替代各向同性高斯，但 KITE 的源是**可学习历史条件流形**，不是固定 GP 核规则；并原生接未来外生与统计知识注意力。[^src-kite]
- 相对 [[timegrad|TimeGrad]] / [[csdi|CSDI]]：骨干从扩散换成直线路径流匹配，且条件通路按内生/外生语义分离并受先验门控。[^src-kite]

## 相关页面

- [[source-kite]] — 源摘要
- [[history-conditional-manifold]] — HCM 技术
- [[knowledge-guided-conditioning]] — KGC 技术
- [[classifier-free-guidance]] — CFG
- [[kite-manifold-guidance-chain]] — 三模块串联
- [[flow-matching-forecasting]] — 流匹配预测总览
- [[gaussian-process-prior-flow-matching]] — TSFlow 的 GP 源路线
- [[multimodal-exogenous-guided-long-term-st-forecasting]] — 外生引导长期时空预测议程
- [[dag]] / [[source-dag]] — 确定性双相关外生预测对照（IJCAI 2026，同 ECNU 组）
- [[gcgnet]] / [[source-gcgnet]] — 图一致生成外生预测对照（ICLR 2026，同 ECNU 组）

[^src-kite]: [[source-kite]]
