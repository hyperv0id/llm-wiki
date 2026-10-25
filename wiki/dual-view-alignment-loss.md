---
title: "Dual-View Alignment Loss (DVA)"
type: technique
tags:
  - loss-function
  - frequency-domain
  - periodic-residual-decoupling
  - traffic-forecasting
created: 2026-04-27
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# Dual-View Alignment Loss (DVA)

DVA 是 [[hyperd|HyperD]] 的辅助损失：把预测 $\hat Y$ 的频谱按一个**预定义**频点 $F_{low}$ 切成低频、高频两段，各自 IFFT 回时域，低频段对齐周期支输出 $S^{out}$、高频段对齐残差支输出 $R^{out}$，两项都是 MSE[^src-hyperd-hybrid-periodicity-decoupling]。

它约束的是「预测的频段归属」，不是两个编码器表示之间的一致性；论文没有使用 Soft-DTW，也没有余弦相似度[^src-hyperd-hybrid-periodicity-decoupling]。

## 数据流与维度

**维度符号**：$T_2$ 预测步；$N$ 节点数；$T_2'=\lfloor T_2/2\rfloor+1$ 为共轭对称截断后的频点数；$F_{low}$ 为切分频点下标。

$$\begin{aligned}
\hat Y_f &= \operatorname{FFT}(\hat Y) && \text{// } [T_2,N]\xrightarrow{\text{沿 } T_2 \text{ 维 FFT}}[T_2',N]\\
\hat Y_f^{low},\ \hat Y_f^{high} &= \hat Y_f[0{:}F_{low}],\ \hat Y_f[F_{low}{:}] && \text{// } [T_2',N]\xrightarrow{\text{按频点切两段}}[F_{low},N],\ [T_2'-F_{low},N]\\
\mathcal{L}_{dva} &= \operatorname{MSE}\!\left(\operatorname{IFFT}(\hat Y_f^{low}),S^{out}\right)+\operatorname{MSE}\!\left(\operatorname{IFFT}(\hat Y_f^{high}),R^{out}\right) && \text{// } [T_2,N]\times 2\ \text{——低频压向周期剖面，高频压向残差}\\
\mathcal{L} &= \mathcal{L}_{pred}+\alpha\,\mathcal{L}_{dva} && \text{// 标量——}\alpha \text{ 平衡对齐项}
\end{aligned}$$

$\mathcal{L}_{pred}$ 是式 16 的逐元素绝对误差求和；$\alpha$ 与 $F_{low}$ 都在验证集上网格搜索，$F_{low}\in\{1,2,3\}$、$\alpha\in\{0.5,1,1.5,2\}$。PEMS07 最优 $F_{low}=1,\alpha=2.0$，PEMS08 最优 $F_{low}=2,\alpha=0.5$；作者指出这两个超参对 MAE 影响较小、对 RMSE 更明显（附录超参研究）[^src-hyperd-hybrid-periodicity-decoupling]。

## 为什么需要它

周期支与残差支各自输出后直接相加，两支的频段归属没有约束：残差支可以拟合出低频周期形状，周期支也可能带上高频抖动。DVA 用「预测频谱的低频 = 周期剖面、高频 = 残差」这一硬分配来抑制重叠，让减法得到的 $R^{in}$ 与最终预测的频段含义保持一致[^src-hyperd-hybrid-periodicity-decoupling]。

## 证据

去掉 DVA 后 PEMS04 / PEMS08 的 MAE 从 18.20 / 13.59 升到 18.47 / 13.70（Table 2），是六个消融项里最小的退化[^src-hyperd-hybrid-periodicity-decoupling]。$F_{low}$ 的取值直接受预测长度 $T_2=12$ 限制：可切分的频点只有 1–3 个，因此该损失在 12 步预测上是很粗的频段约束[^src-hyperd-hybrid-periodicity-decoupling][INFERENCE]。

[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
