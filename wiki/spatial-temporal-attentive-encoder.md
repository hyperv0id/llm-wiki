---
title: "Spatial-Temporal Attentive Encoder (STAE)"
type: technique
tags:
  - spatial-temporal
  - attention
  - graph-convolution
  - traffic-forecasting
created: 2026-04-27
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# Spatial-Temporal Attentive Encoder (STAE)

STAE 是 [[hyperd|HyperD]] 里精炼**周期参数表**的编码器：输入是一张 $L\times N$ 的表，输出同形状的加细表。它由 GCN、时间自注意力、空间自注意力三路组成，后两者并行，输出拼接后线性融合[^src-hyperd-hybrid-periodicity-decoupling]。

日表 $P_D$ 与周表 $P_W$ 各自独立套用同一套 STAE，得到 $\tilde P_D$、$\tilde P_W$；STAE 不接收原始序列，也不接收残差[^src-hyperd-hybrid-periodicity-decoupling]。

## 数据流与维度

**维度符号**：$L$ 周期长度（$L_D=288$ 或 $L_W=2016$）；$N$ 节点数；$\hat A\in\mathbb{R}^{N\times N}$ 归一化图 Laplacian。

$$\begin{aligned}
H &= \sigma\!\left(\hat A\,(P W)\right) && \text{// } [L, \bcancel{N}]\times[\bcancel{N}, N]\to[L, N]\xrightarrow{\hat A\text{ 邻域聚合}}[L, N]\ \text{——论文原式；按行作用于每个时间槽}\\
Q_t,K_t,V_t &= H W_t^{q},\,H W_t^{k},\,H W_t^{v} && \text{// } [L,\bcancel{N}]\times[\bcancel{N},N]\to[L,N]\ \text{——时间注意力的投影}\\
H_t &= \operatorname{softmax}\!\left(\frac{Q_tK_t^{\top}}{\sqrt{N}}\right)V_t && \text{// } [L,N]\times[N,L]\to[L,L]\xrightarrow{\times V_t}[L,N]\ \text{——每个时间槽作 query}\\
Q_s,K_s,V_s &= H^{\top}W_s^{q},\,H^{\top}W_s^{k},\,H^{\top}W_s^{v} && \text{// } [N,L]\times[L,L]\to[N,L]\ \text{——转置后每个节点作 query}\\
H_s &= \left(\operatorname{softmax}\!\left(\frac{Q_sK_s^{\top}}{\sqrt{L}}\right)V_s\right)^{\top} && \text{// } [N,L]\times[L,N]\to[N,N]\xrightarrow{\times V_s}[N,L]\xrightarrow{\top}[L,N]\\
\tilde P &= W_o\,[H_t;H_s] && \text{// } [L,\bcancel{2N}]\times[\bcancel{2N},N]\to[L,N]\ \text{——两路拼接后线性融合}
\end{aligned}$$

投影矩阵维度按论文：$W_t^{q,k,v},W\in\mathbb{R}^{N\times N}$，$W_s^{q,k,v}\in\mathbb{R}^{L\times L}$，$W_o\in\mathbb{R}^{2N\times N}$[^src-hyperd-hybrid-periodicity-decoupling]。

## 设计要点

- **表即输入**：STAE 建模的是「同一墙钟槽位在不同节点、不同天之间的相关性」，不是序列内部的 token 交互；因此它能把单节点均值表扩展成带空间、跨周期上下文的表示[^src-hyperd-hybrid-periodicity-decoupling][INFERENCE]。
- **与 STAEformer 的区别**：STAEformer 对原始序列做时空嵌入与注意力；HyperD 沿用该骨架但输入换成周期参数表，且只服务周期支（见 [[hybrid-periodicity-decoupling]]）[^src-hyperd-hybrid-periodicity-decoupling][INFERENCE]。
- **没有双路径**：日、周两张表是并行精炼后**相加**，不是短周期/长周期两套编码器；残差支走的是 [[frequency-aware-residual-representation|FARR]]，与本编码器无关[^src-hyperd-hybrid-periodicity-decoupling]。

## 证据

去掉 STAE 后 PEMS04 / PEMS08 的 MAE 从 18.20 / 13.59 升到 21.07 / 15.97（Table 2）[^src-hyperd-hybrid-periodicity-decoupling]。

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
