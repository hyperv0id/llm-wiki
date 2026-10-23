---
title: "Next-Scale Autoregressive Trajectory Decoding"
type: technique
tags:
  - trajectory-prediction
  - autoregressive
  - multi-scale
  - next-scale-prediction
  - interpolation
created: 2026-10-20
last_updated: 2026-10-23
source_count: 3
confidence: medium
status: active
---

# Next-Scale Autoregressive Trajectory Decoding（尺度级自回归轨迹解码）

**尺度级自回归解码**是 [[trajar|TrajAR]] 论文提出的轨迹解码策略：把自回归的单位从「下一个时间步」换成「下一个更细的尺度」，每个自回归步并行输出该尺度上的全部未来位置，并借三次插值从上一尺度构造更细尺度的起点[^src-trajar]。论文把这一改动定位为其贡献 1，来源是图像生成中的 visual autoregressive（VAR）思路[^src-trajar]；VAR 原始范式见 [[next-scale-prediction]]（多尺度 VQVAE token map 由粗到细自回归、尺度内并行）[^src-2404-02905-var]。

## 动机：两种输出框架各自的问题

论文对两种既有框架的批评如下[^src-trajar]：

- **逐步自回归（next-step）**：后段轨迹受前期误差累积影响，偏差随 horizon 放大。
- **非自回归（并行输出全部步）**：每步预测缺少来自前一步的纠正，论文称这会给整条轨迹引入整体偏置，并可能违反运动学约束（加速度突变、转弯半径不真实）。

尺度级自回归的取舍是：粗尺度（位置少、间隔大）先决定长时路径的骨架，且在每个尺度上都被监督；细尺度在骨架的插值上做局部修正[^src-trajar]。图 1(b) 把三者并排：step-by-step、non-autoregressive、以及本文的多尺度预测[^src-trajar]。

## 机制

**尺度分组。**未来位置 $\langle\tau^{\nu_T}_{t_0+1},\dots,\tau^{\nu_T}_{t_0+t_f}\rangle$ 被分成 $K$ 个尺度 $S_1,\dots,S_K$，$S_K$ 与目标预测分辨率一致，$S_0=\langle\tau^{\nu_T}_{t_0}\rangle$ 是当前时刻位置[^src-trajar]。似然分解为

$$
p(S_1,\dots,S_K)=\prod_{k=1}^{K}p(S_k \mid T_1,\dots,T_k)
$$

每步的源序列 $T_k=[S_0\|\hat S_{k-1}]$，第一步 $T_1=[O_{his}\|S_0]$，$O_{his}$ 是编码器输出；第 $k$ 步的整个 $\hat S_k$ 在一次前向中并行生成[^src-trajar]。

**三次插值构造细尺度起点。**$k>1$ 时，在 $T_k$ 的每两个相邻位置之间插入一个新位置、再丢掉 $\tau^{\nu_T}_{t_0}$，得到更细分辨率的 $\bar T_k$，用全连接层 $\phi_k$ 逐位置映射为嵌入 $E_k$[^src-trajar]。插值算子的作用是把「上一尺度的粗预测」变成「本尺度的初始化」，从而让细尺度不必从零开始[^src-trajar]。

**Block-wise causal attention。**解码器是 $l_d$ 层 Transformer，用按尺度分块的因果 mask，保证 $\hat S_k$ 只能看分辨率不细于 $k$ 的源序列 $T_j$（$0<j\le k$），防止细尺度信息回流到粗尺度[^src-trajar]。

**输出与监督。**解码输出经一层 MLP（BatchNorm + Sigmoid）得到位置预测；训练时所有尺度同时与真值比较，损失是各尺度 RMSE 之和

$$
\mathcal{L}=\sum_{k=1}^{K}\sum_{\tau\in S_k}\sqrt{(\tau-\hat\tau)^2}
$$

最终预测取最细尺度 $\hat S_K$[^src-trajar]。论文把监督口径表述为「粗尺度预测获得更频繁的监督」（图 1(b) 的 multi-supervise 一栏）[^src-trajar]。

## 证据

**消融。**表 3 中把 `TrajAR Decoder` 列置 ×（即不用多尺度解码器）时误差上升最大：SinD-Tianjin ADE/FDE 0.409/0.798 → 1.623/3.241，SinD-Xi'an 0.509/0.942 → 1.892/3.912[^src-trajar]。该行同时是表 3 五组设置里最差的一组，论文据此称 next-scale 路线有效[^src-trajar]。

**插值算子对照。**表 7（SinD-Tianjin，按使用者类型，ADE/FDE）比较三次插值、线性插值、Lagrange 插值与「None」：三次插值在机动车（0.303/0.564）、自行车（0.613/1.171）、行人（0.408/0.701）上均最低；线性插值在机动车上退化到 1.321/2.212，Lagrange 在行人上退化到 1.723/3.345[^src-trajar]。论文的解释是三次插值在位置、速度、曲率上连续，线性插值对快速使用者（机动车）不稳，Lagrange 在高阶多项式上出现 Runge 效应[^src-trajar]。

## 与 VAR 及其它移植的关系

- 论文说明该结构「inspired by VAR [Tian et al., 2024]」（图像生成的 next-scale 预测）[^src-trajar]。
- VAR 原文对 raster-scan next-token 图像 AR 的四点批评（单向假设违反 VQVAE 双向相关性、零样本双向任务受限、flatten 破坏空间局部性、O(n⁶) 计算低效）是这一移植路线的问题动机[^src-2404-02905-var]。
- 同仓库中 [[climatear]] 也把 VAR 用于概率预测，但它先在 VQ 分词器里把连续场量化成多尺度离散 token、再用分类交叉熵逐尺度生成[^src-climatear]。TrajAR 的方法描述里没有量化或码本环节：尺度间靠三次插值上采样、目标是连续坐标的 RMSE（上式），并行生成的是一个尺度上的全部位置[^src-trajar]。两者的共同点是「逐尺度而非逐 token 自回归」，差别在连续/离散表示与监督方式。

## 适用范围与未覆盖项

- 只在四个路口数据集、$t_h=t_f=32$ 步的确定性单模态设定下验证[^src-trajar]；论文把概率多模态预测列为未来工作[^src-trajar]。
- 尺度数 $K$、各尺度含哪些位置（图 2(b) 只给示意）、插值算子的计算开销，正文实现设置均未列出；表 7 的「None」一栏对应的具体上采样方式也未说明[^src-trajar]。
- 表 3 的 `TrajAR Decoder` 行只标 ✓/×，未说明替换成什么解码器（逐步自回归、单步并行或其它），因此该行不能直接读成「同 backbone 下 next-step 的对照」[^src-trajar]。

## Related Pages

- [[trajar]] — 模型页
- [[source-trajar]] — 论文源摘要
- [[climatear]] — VAR 在概率气候预测中的另一条移植路线
- [[next-scale-prediction]] — VAR 原始范式（图像生成）
- [[source-2404-02905-var]] — VAR 论文源摘要
- [[intersection-trajectory-prediction]] — 任务页（三种输出框架对照）

[^src-trajar]: [[source-trajar]]
[^src-climatear]: [[source-climatear]]
[^src-2404-02905-var]: [[source-2404-02905-var]]
