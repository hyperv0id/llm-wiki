---
title: "周期化相对注意力偏置（Periodic Relative Attention Bias）"
type: technique
tags:
  - attention
  - attention-bias
  - periodicity
  - time-series
  - alibi
  - grouped-query-attention
created: 2026-10-25
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# 周期化相对注意力偏置

把 [[alibi|ALiBi]] 的"距离越大惩罚越重"改成"相位差越大惩罚越重"：对相对位置先取模周期、再折到半周期内，加性写进注意力分数[^src-penguin]。PENGUIN 用它在一个标准 PatchTST 骨架上处理日/周等多重周期，不改变输入布局[^src-penguin]。模型全貌见 [[penguin|PENGUIN]]。

## 构造

设 patch 序列上 query 位置 $i$、key 位置 $j$，第 $r$ 个注意力组绑定的周期为原始周期 $P_r$、patch 长 $P$、步长 $S$，则 patch 后的周期为 $P_S^{(r)}=P_r/S$（要求 $S$ 整除每个 $P_r$）[^src-penguin]。

$$\begin{aligned}
P_S^{(r)} &= P_r/S && \text{// } \text{标量}\ \text{——周期在 patch 序列上的长度，须为整数}\\
u_{ij} &= |i-j| \bmod P_S^{(r)} && \text{// } [N,N]\xrightarrow{\text{逐元素取模}}[N,N]\ \text{——lag 与 lag}+P_S\text{ 落到同一相位}\\
\hat B_{ij} &= \min\!\left(u_{ij},\,P_S^{(r)}-u_{ij}\right) && \text{// } [N,N]\to[N,N]\ \text{——三角折叠到 }[0,P_S/2]\\
B^{(k)}_{ij} &= -m_k\,\hat B_{ij} && \text{// } m_k=2^{-k/8}\ \text{——组内第 }k\text{ 个头的斜率，固定不训练、跨组共享}\\
\Lambda^{(r)}_{\ell} &= Q_\ell K_r^{\top}/\sqrt{d_h}+B^{(k)} && \text{// } [N,\bcancel{d_h}]\times[\bcancel{d_h},N]\to[N,N]\ \text{——与打分同形状相加后再 softmax}
\end{aligned}$$

三处与 ALiBi 的差别[^src-penguin]：

| 项 | ALiBi | 本偏置 |
|---|---|---|
| 距离 | $\lvert i-j\rvert$ | $\min(u,\,P_S-u)$，$u=\lvert i-j\rvert \bmod P_S$ |
| 斜率 | 按头号从几何序列取 | 仍取 $m_k=2^{-k/8}$，但在各组之间共用同一个向量 $\mathbf m$ |
| 移植到多头 | 每个头一张偏置 | 每个组绑定一个周期，组内每头再按斜率分化 |

偏置极大值出现在半周期处（惩罚最重），同相位处为 0；因此注意力相对更偏向"同相位"的 key，而不是"近邻"的 key。无周期数据集改用线性原式 $B_{ij}^{(k)}=-m_k|i-j|$，并保留一个非周期组与周期组并存[^src-penguin]。

## 多周期的处理方式

一个周期一个组，$g$ 组、$h$ 头、$g\mid h$、组内 $n=h/g$ 头；组内 query 各自独立，$K,V$ 按组共享（多查询注意力）。每组一张 $[n,N,N]$ 偏置，组间只差 $P_S^{(r)}$[^src-penguin]。

这样做的直接后果是：$K,V$ 的宽度是 $g\,d_h$ 而不是 $h\,d_h$，多一个周期只多一组偏置与一组 KV，不重排、不复制序列。代价是组内 $n$ 个头被迫共用同一组 KV，头之间的分化只剩偏置斜率。

## 证据

Table 3（$L=336$，Traffic / Electricity，四个 horizon 的 avg MSE）[^src-penguin]：

| 设置 | Traffic | Electricity |
|---|---|---|
| 单组、无偏置 | 0.407 | 0.171 |
| 单组 + 线性偏置 | 0.396 | 0.169 |
| 周期组 $\{24,168\}$ | 0.390 | 0.165 |
| 非周期组 + 周期组 $\{24,168\}$ | 0.387 | 0.166 |

Table 14 在 Weather（$P\in\{6\},\{144\},\{6,144\}$）与 ETTm2（$\{4\},\{96\},\{4,96\}$）上重复同一结论，并给出一个反直觉观察：没有明显多周期性时，**细粒度周期**（$6$ 步 = 1 小时、$4$ 步 = 1 小时）比粗粒度（$144$、$96$ 步 = 1 天）更有效；Weather 的 Both$\{6,144\}$ 得 0.226 / 0.264，No Bias 为 0.234 / 0.270[^src-penguin]。

Table 7 测周期给错的情形（ETTm1）：正确 $\{96\}$ 0.343，错给 $\{72\}$、$\{144\}$ 均 0.349，完全不给 0.349；Traffic 完全不给 0.396（CATS 0.411）[^src-penguin]。即退化到 $\mathcal{P}=\varnothing$ 时仍可用，但错周期比不给周期更糟。

## 边界

- **周期是先验超参**：论文假设周期已知，ACF 只是被引用的取法，未做检测实验；作者自述不适用于周期随时间漂移的信号（ECG）[^src-penguin]。
- **"nested" 全文无定义**：除模型名与"periodic-nested attention bias"之外，没有对"嵌套"一词的说明；可核实的数学内容只有上面的模 + 三角折叠。
- **组与周期的对应未规定**：$p>1$ 时哪个周期进哪组、$g$ 取多少，只给了"一组一周期 + 可留一组非周期"的定性描述；$g$ 没有单独消融[^src-penguin]。
- **实现细节有瑕疵**：式 4 的 $k=\ell\bmod n+1$ 与组内头区间 $[\tau+1,\tau+n]$ 组合后错位一位；正文一处说组内共享单个 $k_r\in\mathbb{R}^{d_h}$，但偏置表是 $N\times N$ 的，实际必须是按 token 的 $K_r,V_r\in\mathbb{R}^{N\times d_h}$；decoder 版在式 9 与 $\S5.4$ 里符号相反；App. A.2 写非重叠切分，而 Fig. 2 与 App. D.4 用的是重叠切分[^src-penguin]。
- **因果性未进公式**：掩码不在式 4 中，只在 $\S5.6$ 用热图论证其必要性[^src-penguin]。

## 引用

[^src-penguin]: [[source-penguin]]
