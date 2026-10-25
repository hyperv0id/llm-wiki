---
title: "Temporal Query Technique"
type: technique
tags:
  - attention
  - periodicity
  - multivariate
  - time-series
  - global-correlation
created: 2026-04-28
last_updated: 2026-10-25
source_count: 3
confidence: medium
status: active
---

# Temporal Query Technique

Temporal Query (TQ) 技术用**周期性偏移的可学习向量**作为注意力 Query，使注意力同时捕获变量间的稳定全局模式与样本级局部变化[^src-tqn]。

## 核心思想

标准自注意力的 Q、K、V 均由输入序列生成，相关性估计仅依赖单个样本，易受噪声、极端值与缺失值影响。TQ 将 Query 替换为可学习参数：

$$\begin{aligned}
Q_h &= \theta_{\mathrm{TQ}}^{t,L}W_h^{Q} && \text{// 来自周期性偏移的参数片段，刻画全局变量相关性}\\
K_h &= X_tW_h^{K},\quad V_h = X_tW_h^{V} && \text{// 来自输入，保留样本级局部信息}
\end{aligned}$$

## 周期偏移机制

给定周期长度 $W$，参数表为 $\theta_{\mathrm{TQ}}\in\mathbb{R}^{C\times W}$；时刻 $t$ 截取长度 $L$ 的循环片段：

$$\theta_{\mathrm{TQ}}^{t,L}=\theta_{\mathrm{TQ}}[:,\mathcal{I}],\qquad \mathcal{I}=(t \bmod W+\operatorname{Range}(L)) \bmod W$$

由此 $\theta_{\mathrm{TQ}}^{t,L}=\theta_{\mathrm{TQ}}^{t+iW,L}$（式 2），相隔 $W$ 步的样本共享同一组 query。论文将该构造的作用归结为参数复用与噪声平均（§3.2）[^src-tqn]。$W<L$ 时下标在窗内回绕，query 列按周期重复[^src-tqn][INFERENCE]。

## 注意力计算

$$\mathrm{Head}_h=\operatorname{Softmax}\!\left(Q_hK_h^{\top}\big/\sqrt{L}\right)V_h$$

$Q_h,K_h,V_h\in\mathbb{R}^{C\times L_h}$，打分矩阵为 $[C,C]$，softmax 沿 key 通道归一，输出汇聚全部 $C$ 个变量的局部特征。式 3 的分母写作 $\sqrt{L}$，按单头宽度应为 $\sqrt{L_h}$[^src-tqn][INFERENCE]。

## 与其他方法的对比

|方法|Query 来源|特点|
|---|---|---|
|TQ Technique|可学习向量（周期性偏移）|融合全局先验与局部样本|
|标准自注意力|输入序列|仅样本级相关性|
|PENGUIN|输入序列 + 周期偏置|在注意力分数中注入周期偏置[^src-penguin]|
|CycleNet|可学习参数（无注意力）|以周期参数预测残差|
|[[detr\|DETR]]（目标检测）|可学习 object queries|集合预测的非自回归并行解码[^src-detr]|

「Query 取自可学习参数而非输入数据」在目标检测中的对应实例为 [[detr|DETR]] 的 [[object-queries|object queries]]：$N$ 个 learned positional embeddings 输入 Transformer decoder，逐 query 并行解码目标集合（Sec 3.2）[^src-detr]。

与 PENGUIN 的分工：两者均向注意力注入周期先验，TQ 替换 query 来源，PENGUIN 则保留输入序列作 query、在 key 侧打分叠加相位折叠偏置[^src-penguin]。

## Query/Key 组合的实测（Table 3）

七个高维数据集、四个预测步长平均，三种配置的 MSE（Table 3）[^src-tqn]：

|数据集|(Q=Raw, K=Raw)|(Q=TQ, K=Raw)|(Q=TQ, K=TQ)|
|---|---|---|---|
|Electricity|0.175|**0.164**|0.179|
|Solar|0.208|**0.198**|0.213|
|Traffic|**0.426**|0.445|0.429|
|PEMS03|0.114|**0.097**|0.111|
|PEMS04|0.112|**0.091**|0.113|
|PEMS07|0.094|**0.075**|0.092|
|PEMS08|0.170|**0.142**|0.174|

仅将 Query 换为 TQ（K 仍取输入）在七个数据集中六个最优，Query 与 Key 均换为 TQ 则全面退化；Traffic 是唯一例外，全自注意力略优（0.426 vs 0.445），论文未单独解释[^src-tqn]。对比表明增益来自「Q 全局、K 局部」的分工，而非单纯引入可学习参数。

## 超参数 W

$W$ 按数据集的稳定周期设定，与 CycleNet 的取值对齐（Table 1）[^src-tqn]：

|数据集|采样间隔|W|
|---|---|---|
|ETTh1 / ETTh2|1 h|24|
|ETTm1 / ETTm2|15 min|96|
|Electricity|1 h|168|
|Solar|10 min|144|
|Traffic|1 h|168|
|Weather|10 min|144|
|PEMS03 / 04 / 07 / 08|5 min|288|

$W$ 可依领域周期先验或自相关函数（ACF）确定（附录 A.3）。敏感性见 Figure 6：Electricity 上 $W=168$ 最优（MSE 0.164），$W=24$ 仅捕获部分周期，$W=23$、$167$ 等非真实周期取值带来语义不一致、性能下降；$W=336$（两周）仍接近 168[^src-tqn]。每个数据集仅设一个 $W$，多周期耦合时需退化为最长/最短周期或集成多个模型（附录 A.3）[^src-tqn]。

## 效果

> 融合全局（TQ）与局部（输入）优于仅样本、优于仅全局[^src-tqn]

排序依据为 Table 3 在七个高维数据集上的四步长平均与 §4.2 对 Table 3 的归纳；Traffic 的 MSE 例外见上表[^src-tqn]。

## 相关页面

- [[tqn|TQNet]] — 该技术的完整模型页
- [[periodicity-modeling-in-time-series]] — 周期建模路线对照，TQ 属「显式—先验」
- [[source-tqn]] — 论文摘要页

## 引用

[^src-tqn]: [[source-tqn]]
[^src-penguin]: [[source-penguin]]
[^src-detr]: [[source-detr]]
