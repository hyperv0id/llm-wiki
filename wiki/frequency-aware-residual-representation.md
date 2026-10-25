---
title: "Frequency-Aware Residual Representation (FARR)"
type: technique
tags:
  - fourier-transform
  - complex-mlp
  - traffic-forecasting
  - residual-modeling
created: 2026-04-27
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# Frequency-Aware Residual Representation (FARR)

FARR 是 [[hyperd|HyperD]] 中处理**残差分量**的模块：先由周期剖面减出残差，再用复值 MLP 在空间、时间两轴的频域里编码[^src-hyperd-hybrid-periodicity-decoupling]。它不负责提取周期，也不做频段划分。

## 输入从哪来

$$\begin{aligned}
S &= \operatorname{Gather}(\tilde P_D,i_D)+\operatorname{Gather}(\tilde P_W,i_W) && \text{// } [T_1,N]\ \text{——按时间戳从日/周表取行相加}\\
R^{in} &= X-S && \text{// } [T_1,N]\ \text{——观测减周期剖面，得到残差}
\end{aligned}$$

周期剖面 $S$ 来自 [[spatial-temporal-attentive-encoder|STAE]] 精炼后的参数表，与输入 $X$ 无关；这一步的减法就是论文所谓的「解耦」[^src-hyperd-hybrid-periodicity-decoupling]。

## 数据流与维度

**维度符号**：$T_1$ 输入步；$N$ 节点数；$D$ 升维后的通道维；$D'$ 复值 MLP 隐层维；$N'=\lfloor N/2\rfloor+1$、$T_1'=\lfloor T_1/2\rfloor+1$ 为共轭对称截断后的频点数。

$$\begin{aligned}
\tilde R^{in} &= \operatorname{Proj}_{1\to D}(R^{in}) && \text{// } [T_1,N]\xrightarrow{\text{逐元素升维}}[T_1,N,D]\\
R^s_f &= \operatorname{FFT}_s(\tilde R^{in}) && \text{// } [T_1,N,D]\xrightarrow{\text{沿 } N \text{ 维 FFT}}[T_1,N',D]\ \text{——空间频率表示}\\
R_2 &= \operatorname{CMLP}(R^s_f) && \text{// } [T_1,N',D]\xrightarrow{D\to D'\to D}[T_1,N',D]\ \text{——复值矩阵乘混合实部与虚部}\\
R^s &= \operatorname{IFFT}_s(R_2) && \text{// } [T_1,N',D]\xrightarrow{\text{沿 } N' \text{ 维 IFFT}}[T_1,N,D]\\
R^t &= \operatorname{IFFT}_t(\operatorname{CMLP}(\operatorname{FFT}_t(R^s))) && \text{// } [T_1,N,D]\to[T_1',N,D]\to[T_1',N,D]\to[T_1,N,D]\\
R^{out} &= \operatorname{Proj}(R^t+\tilde R^{in}) && \text{// } [T_1,N,D]\xrightarrow{\text{残差连接 + 沿时间投影}}[T_2,N]
\end{aligned}$$

复值 MLP 的两层写法（式 9）：$\operatorname{Re}(R_1)=\sigma(\operatorname{Re}(R^s_f)W_1^{r}-\operatorname{Im}(R^s_f)W_1^{i}+b_1^{r})$，$\operatorname{Im}(R_1)=\sigma(\operatorname{Im}(R^s_f)W_1^{r}+\operatorname{Re}(R^s_f)W_1^{i}+b_1^{i})$；第二层同构。权重 $W_1\in\mathbb{R}^{D\times D'}$、$W_2\in\mathbb{R}^{D'\times D}$[^src-hyperd-hybrid-periodicity-decoupling]。按卷积定理，频域复值乘法等价于时域全局卷积；论文用这一点把残差的高频波动当作全局模式学习[^src-hyperd-hybrid-periodicity-decoupling]。

## 与其他频域方法的关系

- **FEDformer**：频域注意力 + 频域交叉注意力，随机选固定数量 Fourier 模式；FARR 只做「FFT → 复值 MLP → IFFT」，没有注意力[^src-hyperd-hybrid-periodicity-decoupling]。
- **FreTS**：频域 MLP 直接作用在原始序列上；FARR 作用在减掉周期剖面后的残差上[^src-hyperd-hybrid-periodicity-decoupling]。
- **频段分解**：FARR 不做高/中/低频切分，也没有可学习阈值；全文唯一的频率切分在 [[dual-view-alignment-loss|DVA 损失]]里，切的是预测频谱，不是输入[^src-hyperd-hybrid-periodicity-decoupling]。

## 证据

去掉 STFE（即整个残差支换成普通 MLP 之前的结构）在 PEMS04 / PEMS08 上 MAE 从 18.20 / 13.59 升到 25.24 / 18.47，是 Table 2 中最大的单模块退化；把 STFE 换成普通 MLP 只升到 18.57 / 13.73[^src-hyperd-hybrid-periodicity-decoupling]。

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
