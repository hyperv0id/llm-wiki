---
title: "KITE: Knowledge-Guided Probabilistic Modeling for Time Series Forecasting with Exogenous Variables"
type: source-summary
tags:
  - flow-matching
  - probabilistic-forecasting
  - exogenous
  - classifier-free-guidance
  - manifold
  - 2026
  - icml-2026
created: 2026-07-12
last_updated: 2026-07-13
source_count: 1
confidence: high
status: active
---

# KITE: Knowledge-Guided Probabilistic Modeling for Time Series Forecasting with Exogenous Variables

**Authors**: Hanyin Cheng, Jingrong Zhou, Yang Shu, Chenjuan Guo (East China Normal University)

**Venue**: ICML 2026 (PMLR 306) | **Code**: [github.com/decisionintelligence/KITE](https://github.com/decisionintelligence/KITE)

## 核心贡献

KITE 把**外生变量条件下的概率时序预测**做成端到端生成框架：骨干是 [[flow-matching|Flow Matching]]，外生变量作为条件，专门打两个瓶颈——(1) 无上下文先验与目标分布之间的**拓扑落差**，(2) 迭代生成中**伪相关协变量**被放大。[^src-kite]

相对 [[source-timexer|TimeXer]] / TFT / TiDE 等确定性外生预测，以及 [[source-tsflow|TSFlow]] / [[source-timegrad|TimeGrad]] / [[source-csdi|CSDI]] 等概率模型，KITE 同时要求：未来外生已知时可控条件生成，且输出完整预测分布而非点估计。[^src-kite]

## 关键设计

### 1. History-Conditional Manifold (HCM)

用历史内生序列构造**可学习源分布** \(Y_0 = \mu_{\text{hist}} + \sigma_{\text{hist}} \delta_{\text{hist}}\)，替换标准高斯 \(N(0,I)\)：[^src-kite]

- **Barycenter Mapping**：\( \mu_{\text{hist}} = f_\phi(X_{\text{endo}}) \)，把历史投影到预测空间中心；
- **Uncertainty Estimator**：\( \sigma_{\text{hist}} = \mathrm{Softplus}(g_\psi(X_{\text{endo}})) + \sigma_{\min} \)，异方差覆盖尺度；
- **Manifold Projector**：低秩流形基 \(M\) 与各向同性噪声混合，\(\delta_{\text{hist}} = \alpha \frac{Mz}{\|Mz\|} + (1-\alpha)\epsilon\)；
- **Coverage Constraint**：\(L_{CC}\) 阻止 \(\sigma\) 塌缩，并强制源支撑盖住目标（对 \(\mu\) stop-gradient）。

理论：在历史中心化误差下降足够大时，HCM 的匹配目标尺度更小、路径管更窄，路径局部 Jacobian budget 更低（Proposition 1–2）。[^src-kite]

### 2. Knowledge-Guided Conditioning (KGC)

用统计先验（主实验为 Pearson；消融含 Granger）调制内生–外生注意力，而非简单相加先验权重：[^src-kite]

\[
\mathrm{Attn}(q_i,k_j) = q_i (W_1 + s_{ij} W_2) k_j^\top
\]

实现上拆成两条投影支路 \(A_b\)（数据驱动）与 \(A_g\)（先验注入），再与归一化先验 \(\tilde S\) 融合并 log-gating；历史/未来外生可顺序注入，未来外生缺失时可跳过。[^src-kite]

### 3. Classifier-Free Guidance (CFG)

训练时以概率 \(p_{\text{con}}\) 丢弃条件 \(c=\{X_{\text{exo}}, Y_{\text{exo}}\}\)，同时学协变量无关与协变量条件速度场；推理时[^src-kite]

\[
\hat v_s = (1+\gamma)\, v_\theta(Y_s,s,c) - \gamma\, v_\theta(Y_s,s,\varnothing)
\]

\(\gamma\) 显式控制外生影响强度；敏感度实验中 \(\gamma \approx 1.2\text{–}1.4\) 最优。[^src-kite]

## 实验结果（摘要）

- **确定性 + 历史/未来外生**（12 数据集）：相对 CrossLinear MSE ↓16.1%，相对 TFT MAE ↓9.2%。[^src-kite]
- **确定性 + 仅历史外生**：平均 MSE 相对 TimeXer ↓3.8%，MAE 相对 DUET ↓3.3%。[^src-kite]
- **概率预测 CRPS**（8 数据集，ProbTS 设定）：相对最强单变量 \(K^2\)VAE ↓5.7%，相对最强多变量 CSDI ↓11.4%；覆盖 short/long horizon。[^src-kite]
- **消融**：HCM、KGC、CFG 单独均有增益；三者齐用最优（如 NP MSE 0.395→0.325）。[^src-kite]
- **统计知识**：同阶段 Pearson 优于 Granger；混合时「历史 Granger + 未来 Pearson」优于反向，符合因果/相关的时间语义。[^src-kite]
- **效率**（Weather-L）：参数约 2.3M，在概率基线中推理速度靠前且 CRPS 最优。[^src-kite]

## 局限与边界

- 主设定假设未来外生已知或高精度；作者明确将噪声/缺失/不确定未来外生留作未来工作。[^src-kite]
- 统计先验默认 Pearson，先验质量与非平稳关系漂移仍可能约束 KGC。[^src-kite]
- 概率设定中「最后一维为内生、其余为外生」的划分沿用既有协议，未必覆盖真实多目标联合预测。[^src-kite]

## 相关链接

- [[kite]] — 方法实体
- [[history-conditional-manifold]] — 历史条件流形源分布
- [[knowledge-guided-conditioning]] — 知识引导条件化
- [[classifier-free-guidance]] — 无分类器引导
- [[kite-manifold-guidance-chain]] — 三模块串联分析
- [[tsflow]] / [[source-tsflow]] — GP 先验流匹配前驱
- [[source-timexer]] / [[source-exotst]] / [[source-exost]] — 外生确定性预测谱系
- [[source-dag]] / [[dag]] — DAG 双相关确定性外生预测（IJCAI 2026，同 ECNU 组）
- [[source-gcgnet]] / [[gcgnet]] — GCGNet 图一致生成外生预测（ICLR 2026，同 ECNU 组；确定性联合图对齐）
- [[flow-matching-forecasting]] — 流匹配时序预测总览

[^src-kite]: [[source-kite]]
