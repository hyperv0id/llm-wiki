---
title: "Temporal Query Network (TQNet)"
type: entity
tags:
  - time-series
  - forecasting
  - model
  - icml-2025
  - multivariate
created: 2026-04-28
last_updated: 2026-10-25
source_count: 2
confidence: medium
status: active
---

# Temporal Query Network (TQNet)

TQNet 是 ICML 2025 接收的多变量时序预测模型，由华南理工大学林伟伟团队提出。核心创新是 **Temporal Query (TQ) 技术**：以周期性偏移的可学习向量作为注意力 Query，在单层注意力与浅层 MLP 架构下建模变量间相关性[^src-tqn]。

## 核心创新

### Temporal Query (TQ) 技术

传统自注意力（如 iTransformer）的 Q、K、V 全部来自输入序列，相关性估计完全依赖单样本，受噪声、极端值与缺失值影响。TQ 技术将 Query 替换为可学习参数：

- **Query**：$\theta_{\mathrm{TQ}}\in\mathbb{R}^{C\times W}$ 的周期性片段，刻画变量间全局相关性
- **Key / Value**：原始输入 $X_t\in\mathbb{R}^{C\times L}$，保留样本级局部信息

论文将该分工记为式 1–式 3（§3.2），并在 Table 3 对比三种 Query/Key 组合[^src-tqn]。

### 周期偏移机制

给定时刻 $t$，按下标 $t \bmod W$ 为起点从 $\theta_{\mathrm{TQ}}$ 循环取长度 $L$ 的片段：

$$\theta_{\mathrm{TQ}}^{t,L} = \theta_{\mathrm{TQ}}[:,\ \mathcal{I}],\qquad \mathcal{I}=(t \bmod W + \operatorname{Range}(L)) \bmod W$$

由此 $\theta_{\mathrm{TQ}}^{t,L}=\theta_{\mathrm{TQ}}^{t+iW,L}$（式 2）：间隔 $W$ 步的样本共用同一组 query。参数复用的同时，同组参数接收多样本梯度更新，等效于对局部噪声做平均[^src-tqn]。

## 架构

TQNet 仅包含三部分（Figure 2）：

1. **TQ-MHA**：单层多头注意力，Query 来自 TQ 向量，K/V 来自输入
2. **MLP**：两层全连接 + GeLU，带残差连接
3. **Instance Normalization**（可选）：输入去均值与方差，输出反归一化

头数固定为 4，dropout 默认 0.5，$\theta_{\mathrm{TQ}}$ 初始化为 0（Algorithm 1、附录 A.2）[^src-tqn]。

## 数据流与维度

形状记作 $[\text{行}, \text{列}]$。**维度符号**：$C$ 为变量（通道）数；$L$ 为回溯长度，论文实验统一定为 $L=96$（Table 2）；$H$ 为预测步长，ETT/Electricity/Solar/Traffic/Weather 取 $\{96,192,336,720\}$，PEMS 取 $\{12,24,48,96\}$（§4.1）；$W$ 为周期长度（Table 1）；$h=4$ 为头数，$L_h=L/h$ 为单头宽度；$d$ 为注意力后线性层隐藏宽度，论文仅给形状 $[C,d]$、未给数值（附录 A.1）。论文按单头书写，分头展开为本页推断[^src-tqn][INFERENCE]。

### Stage 1 · 归一化与周期取段

$$\begin{aligned}
\tilde{X}_t &= (X_t-\mu)\big/\sqrt{\sigma^2+\epsilon} && \text{// } [C, L]\to[C, L]\ \text{——可选 IN，去掉样本的均值与方差}\\
\mathcal{I} &= (t \bmod W + \operatorname{Range}(L)) \bmod W && \text{// } \text{标量}\to[L]\ \text{——长度为 } L \text{ 的下标序列，越过 } W \text{ 回绕}\\
Q &= \theta_{\mathrm{TQ}}[:,\mathcal{I}] && \text{// } [C, W] \xrightarrow{\text{按 } \mathcal{I}\in[L] \text{ 索引列}} [C, L]\ \text{——本样本的 query，与输入数值无关}
\end{aligned}$$

$W<L$ 时索引在窗内多次回绕，$Q$ 的列按周期重复；Table 1 中 $W$ 最小为 24（ETTh1/ETTh2），对应窗内重复 4 次[^src-tqn][INFERENCE]。

### Stage 2 · 三条投影：Q 来自参数，K/V 来自输入

$$\begin{aligned}
Q_h &= QW_h^{Q} && \text{// } [C, \bcancel{L}]\times[\bcancel{L}, L_h]\to[C, L_h]\ \text{——只由周期参数决定，不含样本噪声}\\
K_h &= \tilde{X}_tW_h^{K} && \text{// } [C, \bcancel{L}]\times[\bcancel{L}, L_h]\to[C, L_h]\ \text{——逐样本，保留局部信息}\\
V_h &= \tilde{X}_tW_h^{V} && \text{// } [C, \bcancel{L}]\times[\bcancel{L}, L_h]\to[C, L_h]
\end{aligned}$$

### Stage 3 · 在通道轴上打分与加权

$$\begin{aligned}
S_h &= Q_hK_h^{\top}\big/\sqrt{L} && \text{// } [C, \bcancel{L_h}]\times[\bcancel{L_h}, C]\to[C, C]\ \text{——消去单头宽度，得通道两两打分}\\
A_h &= \operatorname{Softmax}(S_h) && \text{// } [C, C] \xrightarrow{\text{沿 key 通道归一}} [C, C]\ \text{——每通道对全部 } C \text{ 个通道的权重}\\
\mathrm{Head}_h &= A_hV_h && \text{// } [C, \bcancel{C}]\times[\bcancel{C}, L_h]\to[C, L_h]\ \text{——消去通道维，每通道汇总全部变量的局部特征}
\end{aligned}$$

式 3 分母写作 $\sqrt{L}$，按单头宽度应为 $\sqrt{L_h}$[^src-tqn][INFERENCE]。

### Stage 4 · 多头拼回、输出投影、残差

$$\begin{aligned}
U &= \operatorname{Concat}(\mathrm{Head}_1,\dots,\mathrm{Head}_4) && \text{// } 4\times[C, L_h]\to[C, L]\ \text{——沿列拼接，回到回溯长度}\\
h_{\mathrm{attn}} &= UW^{O} + \tilde{X}_t && \text{// } [C, \bcancel{L}]\times[\bcancel{L}, L]\to[C, L]\ \text{——加回输入，形状不变}
\end{aligned}$$

### Stage 5 · 压到隐藏宽度，过 MLP

$$\begin{aligned}
h'_{\mathrm{attn}} &= \operatorname{Linear}(h_{\mathrm{attn}}) && \text{// } [C, \bcancel{L}]\times[\bcancel{L}, d]\to[C, d]\\
h_{\mathrm{mlp}} &= \operatorname{Linear}\!\left(\operatorname{GeLU}\!\left(\operatorname{Linear}\!\left(h'_{\mathrm{attn}}\right)\right)\right) + h'_{\mathrm{attn}} && \text{// } [C, d]\to[C, d]\ \text{——两层 MLP 建时间依赖，残差稳定训练}
\end{aligned}$$

### Stage 6 · 输出投影与反归一化

$$\begin{aligned}
\bar{Y}_t &= \operatorname{Linear}\!\left(\operatorname{Dropout}(h_{\mathrm{mlp}})\right) && \text{// } [C, \bcancel{d}]\times[\bcancel{d}, H]\to[C, H]\\
\bar{Y}_t &\leftarrow \bar{Y}_t\sqrt{\sigma^2+\epsilon}+\mu && \text{// } [C, H]\to[C, H]\ \text{——仅启用 IN 时执行，还原原始尺度}
\end{aligned}$$

### 结构要点

- **Q 与样本无关**：$Q$ 仅由 $(t \bmod W)$ 与共享参数决定，相隔 $W$ 步的样本 query 完全相同（式 2）。K/V 逐样本变化，两者分工为"全局先验 + 局部信息"[^src-tqn]。
- **注意力在通道轴上做**：打分矩阵为 $[C, C]$，单层注意力的开销为 $C^2L_h$ 的打分与 $C^2L_h$ 的加权，即 $O(C^2L)$；$L$ 只作线性因子，通道数以平方进入。
- **参数规模**：$\theta_{\mathrm{TQ}}$ 为 $C\times W$，与 $L$ 无关；其余权重来自投影（$L_h$ 宽）、压缩线性层 $L\to d$ 与输出层 $d\to H$。单层注意力加浅层 MLP 是论文所称轻量的来源（Figure 7 左）[^src-tqn]。

## 复杂度

打分矩阵为 $[C,C]$，注意力在通道维具二次复杂度，论文在 Limitations 中明确列出该顾虑（§5）[^src-tqn]。论文同时报告：实测训练开销随通道数接近线性，在 $C<1000$ 的实际场景接近线性模型 DLinear（Figure 7 右、§5）。两条陈述指向不同对象：前者是渐近阶，后者是 $C<1000$ 范围内的实测曲线。

## 性能（Table 2）

12 个数据集、24 项误差指标中，TQNet 有 22 项进入前二（Table 2 Count 行），其次为 TimeXer 11 项、CycleNet 9 项[^src-tqn]。论文据此称为整体 SOTA，并强调高维数据集（Electricity 321 通道、PEMS 170–883 通道）上的优势（§4.2）[^src-tqn]。

|数据集|TQNet MSE / MAE|该数据集最强基线 MSE / MAE|基线|
|---|---|---|---|
|Electricity|0.164 / 0.259|0.168 / 0.259|CycleNet|
|Solar|0.198 / 0.256|0.210 / 0.261|CycleNet|
|Traffic|0.445 / 0.276|0.428 / 0.282|iTransformer|
|PEMS07|0.075 / 0.171|0.085 / 0.182|TimeXer|

数值为 Table 2 四个预测步长的均值。Traffic 的 MSE 劣于 iTransformer，论文未做单独解释[^src-tqn]。

## 边界（§5、附录 A.3）

- **依赖周期先验**：$W$ 由数据集稳定周期决定，在缺乏明显周期的数据上泛化受限；$W$ 与真实周期不对齐（如 23、167）时性能下降（Figure 6）[^src-tqn]。
- **弱相关数据上未必有益**：变量间相关性弱时，强制多变量建模会引入多余复杂度，甚至降低性能[^src-tqn]。
- **回溯窗变长的收益递减**：$L$ 足够长时多变量建模收益下降——长窗口本身提供时序信息，同时引入更多噪声（Figure 8）[^src-tqn]。
- **单周期假设**：每个数据集仅设一个 $W$，对多周期耦合（如周与月）需退化为日周期或集成多模型；论文自称多周期建模仍是开放问题（附录 A.3）[^src-tqn]。

## 与其他模型的关系

|模型|变量相关性建模方式|
|---|---|
|TQNet|TQ 技术：可学习向量作为 Query|
|iTransformer|样本级自注意力|
|CycleNet|可学习周期参数（无注意力）|
|PENGUIN|周期分组注意力偏置|
|PatchTST|通道独立（无变量相关性）|
|[[unica|UniCA]]|统一的协变量适应框架，可为 TQNet 添加异构协变量支持|

TQ 可学习参数取自 CycleNet 的周期建模思路，区别在 CycleNet 用参数预测残差目标，TQ 把参数放入注意力作 Query[^src-tqn]。

## 与可学习 Query 设计家族的关系

可学习向量充当 Query 在计算机视觉中早有实例：[[detr|DETR]] 将 $N$ 个 learned positional embeddings（[[object-queries|object queries]]）输入 Transformer decoder，逐 query 并行解码目标集合（Sec 3.2）[^src-detr]。TQ 与 object queries 同属「Query 来自可学习参数而非输入」设计家族，但用途不同：TQ 在时序注意力中注入全局变量相关性先验[^src-tqn]，object queries 用于集合预测的非自回归并行解码[^src-detr]。此对照为 wiki 组织口径，非 TQNet 论文定位。

## 相关页面

- [[unified-covariate-adaptation]] — 统一协变量适应概念
- [[instance-normalization|RevIN]] — IN 与 RevIN 类分布漂移处理
- [[temporal-query-technique]] — TQ 技术专项页
- [[detr]] / [[object-queries]] — 可学习 Query 并行解码在目标检测中的出处

## 引用

[^src-tqn]: [[source-tqn]]
[^src-detr]: [[source-detr]]
