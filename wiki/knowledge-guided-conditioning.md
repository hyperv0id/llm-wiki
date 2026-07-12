---
title: "Knowledge-Guided Conditioning"
type: technique
tags:
  - conditioning
  - attention
  - exogenous
  - statistical-prior
  - probabilistic-forecasting
  - time-series
created: 2026-07-12
last_updated: 2026-07-12
source_count: 1
confidence: high
status: active
---

# Knowledge-Guided Conditioning

**Knowledge-Guided Conditioning (KGC)** 是 [[kite|KITE]] 的条件化模块：把从输入上下文提取的统计先验（Pearson 相关、Granger 因果等）注入内生–外生注意力，用双线性子空间插值抑制伪相关，稳住迭代生成中的协变量交互。[^src-kite]

## 动机

纯数据驱动外生建模（如 [[source-timexer|TimeXer]] 类交叉注意、简单拼接）容易过拟合训练集静态伪相关；在概率生成的**多步传输**里，错误相关会沿轨迹放大，OOD 关系漂移时更差。[^src-kite]

朴素地把统计权重加到输入端，无法解决「结构瓶颈」——注意力投影空间本身仍不随统计环境移动。KGC 改在**查询投影子空间**里做先验调制。[^src-kite]

## 核心公式

统计先验矩阵 \(S=\{s_{ij}\}\in\mathbb{R}^{N\times D}\)，\(s_{ij}\) 为第 \(i\) 个内生与第 \(j\) 个外生的先验强度。[^src-kite]

双线性注意：

\[
\mathrm{Attn}(q_i,k_j)=q_i(W_1 + s_{ij} W_2)k_j^\top
\]

在数据驱动子空间 \(W_1\) 与先验注入子空间 \(W_2\) 之间按 \(s_{ij}\) 插值。[^src-kite]

### 实现分解（不显式构造满双线性矩阵）

历史外生注入示例：[^src-kite]

\[
\begin{aligned}
A_b &= (Y_s W_{1,q})(X_{\text{exo}} W_{1,k})^\top / \sqrt{d},\\
A_g &= (Y_s W_{2,q})(X_{\text{exo}} W_{2,k})^\top / \sqrt{d},\\
A &= \mathrm{Softmax}\big(A_b + \tilde S \odot A_g + \log(\tilde S+\delta)\big).
\end{aligned}
\]

- \(\tilde S=\mathrm{Norm}(|S|)\in[0,1]^{N\times D}\)；
- \(\log(\tilde S+\delta)\) 做门控：先验为 0 时强抑对应注意力；
- 条件注入：\( \tilde Y_s = A X_{\text{exo}} W_v + Y_s \)。[^src-kite]

历史与未来外生顺序注入；未来外生不可用时可省略对应步。[^src-kite]

## 先验选择实验

主实验统一用 Pearson。混合先验消融表明：[^src-kite]

- 同阶段：Pearson 整体优于 Granger；
- 混合：历史外生用 **Granger**、未来外生用 **Pearson** 最好——因果更贴「过去驱动现在」，相关更贴「同期/未来协变」。

## 与链路中其他模块

- 输入侧：KGC 读 \(X_{\text{exo}},Y_{\text{exo}}\)（及 \(S\)），写条件化后的速度网络状态；
- 生成侧：条件 \(c\) 可被 [[classifier-free-guidance|CFG]] 随机置空，从而学到「无外生」速度场；
- 源侧：[[history-conditional-manifold|HCM]] 不负责外生，只提供靠近目标的 \(Y_0\)。

串联叙事见 [[kite-manifold-guidance-chain]]。[^src-kite]

## 相关页面

- [[kite]] / [[source-kite]]
- [[cross-attention-conditioning]] — 一般交叉注意条件化
- [[spurious-patterns]] — 伪模式问题
- [[heterogeneous-covariates]] / [[covariate-fusion-module]] — 协变量适配谱系
- [[source-timexer]] / [[source-exotst]] / [[source-exost]]

[^src-kite]: [[source-kite]]
