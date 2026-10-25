---
title: "WITRAN"
type: entity
tags:
  - time-series
  - forecasting
  - rnn
  - periodicity
  - recurrent
  - long-range-forecasting
  - NeurIPS-2023
created: 2026-10-25
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# WITRAN

WITRAN（Water-wave Information Transmission and Recurrent Acceleration Network）是 Jia et al.（北京交通大学，NeurIPS 2023）提出的长时序预测模型：把输入按人工设定的自然周期重排成 $R\times C$ 二维网格，在网格上沿水平（周期内）、垂直（周期间）两个方向做门控递推，再用反对角线分片把串行递推步数从 $L$ 压到 $R+C-1$（论文记作时间复杂度 $O(\sqrt L)$，内存 $O(L)$）[^src-witran-jia-2023]。

## 问题定位

论文把 long-range 预测要捕获的语义信息分成两类，并主张既有方法只能分别解决其一（§1）[^src-witran-jia-2023]：

1. **global-local correlations**：短程变化与长程趋势；
2. **long- / short-term repetitive patterns**：多时间尺度的重复模式（小时级、日级周期）。

对既有路线的批评（Table 1 / Table 2 口径）[^src-witran-jia-2023]：

- **RNN**：递归结构天然由局部到全局积累语义、线性复杂度，但梯度消失/爆炸与信息遗忘使其难以直接用于长序列。信息遗忘有实验支撑：ECL 数据集上只把最后 $m$ 个点喂给 LSTM/GRU，$m$ 超过阈值后误差不再下降（Appendix B.2，Figure 7）。
- **CNN**：受单层感受野限制（[[micn|MICN]] 用下采样+等距卷积解决）或受 1D 输入限制（[[timesnet|TimesNet]] 用 2D 化解决）——各解决一端。
- **Transformer**：point-wise 注意力难以从单点 token 抽出隐含语义；复杂度上多数变体（LogTrans/Autoformer $O(L\log L)$、Full Attention $O(L^2)$）并未低于 RNN 的 $O(L)$（Appendix C 逐项推导 $4Ld_{model}^2+2L^2d_{model}$ 等公式支撑此论点）。
- **FiLM / DLinear** 等其他 DNN：难以捕获序列中的重复模式。

论文的切入思路承接 [[patchtst|PatchTST]]/TimesNet 已验证的洞察——把信息传递切成相邻几步的 patch 分开处理，可同时保留全局与局部语义——并把这一思路迁移到 RNN：将 RNN 输入切成大量子序列（即周期重排）分别递推（§1）[^src-witran-jia-2023]。

## 机制

### Input Module：自然周期重排（§3.1）

与 TimesNet 的差别被论文明确写出：**不用 FFT 自适应发现多周期，而是分析数据自然周期后以超参设定重排**（$C\in\{12,24,48\}$，要求 $H\bmod C=0$），论文自述因此"much simpler"。归一化是 NLinear 式开关 `norm∈{0,1}`：开启时减去序列最后一个值 $x_H$，预测后加回；时间特征（HourOfDay/DayOfWeek/DayOfMonth/DayOfYear）与序列在特征维拼接后再重排[^src-witran-jia-2023]。

### WIT：双向"水波"传递

重排后信息沿两个方向流动（Figure 2）：水平方向沿行内时间步（intra-period，粒度=小时），垂直方向跨行（inter-period，粒度=天，同列同相位）。"water-wave"类比的确切所指是并行性：**某点算完后，其右邻与下邻互不依赖即可开始**——这既是信息传递图像，也是 [[recurrent-acceleration-network|RAN]] 反对角线并行的物理依据（§3.3）[^src-witran-jia-2023]。

### HVGSU：双向门控单元

每个网格点放一个 HVGSU，内含两个方向的 Gated Selective Cell（GSC）。GSC 借 GRU 的门降低开销（只留 selection + output 两个门），但把门计算改成与输入并行、避免 GRU 中 $r_t\to\tilde h_t$ 的串行等待（Appendix B.3）。核心设计是**主/副状态**：横向 cell 的主状态来自同行左邻、副状态来自上一行同列（反之亦然），于是每个时间步上两类周期语义在门控输入处融合，selection gate 逐通道凸组合决定保留多少原主状态、注入多少融合信息（§3.2）[^src-witran-jia-2023]。

### RAN 加速与预测模块

递推按 [[recurrent-acceleration-network|RAN]] 的反对角线波前并行执行（详见该页）。预测模块非自回归：取横向隐藏态最后一行（含全局+最新短期语义）与纵向各列配对，两层 FC 一次输出全部 $P$ 步，避免自回归误差累积（§3.4，做法承接 Informer/Pyraformer）[^src-witran-jia-2023]。

## 数据流与维度

**维度符号**：$H$ 输入长度，$H=R\times C$（$R$ 行=周期实例数，$C$ 列=周期长度）；$P$ 预测步数，$R_{de}\times C=P$；$N$ HVGSU 层数（论文记作 $L$，与其序列长度记号冲突，本页改用 $N$）；$c_{in}/c_{time}$ 输入/时间特征维；$d_{model}$ 隐藏维；首层 $d_{in}=c_{in}+c_{time}$，深层 $d_{in}=2d_{model}$。形状记 $[\,\cdot\,]$，按论文单通道口径书写。

### Stage 1 · 按自然周期重排

$$\begin{aligned}
X'_{1D} &= \begin{cases}X_{1D}, & \text{norm}=0\\ X_{1D}-x_H, & \text{norm}=1\end{cases} && \text{// } [H, c_{in}]\ \text{减最后一步 } x_H\in[c_{in}]\text{——NLinear 式抑制分布漂移}\\
X_{2D} &= \operatorname{Rearrange}([\,X'_{1D},\,TF_{en}\,]) && \text{// } [H,\,c_{in}{+}c_{time}] \xrightarrow{\text{按 } H=R{\times}C \text{ 折叠}} [R, C, c_{in}{+}c_{time}]\ \text{——列=周期内步，行=周期实例}
\end{aligned}$$

### Stage 2 · 双向 GSC 递推（逐层逐格）

**横向支路（principal=本向，subordinate=纵向）**：

$$\begin{aligned}
h^{hor}_{r,c,l} &= \mathrm{GSC}_{hor}\big(input_{r,c,l},\; h^{hor}_{r,c-1,l},\; h^{ver}_{r-1,c,l}\big) && \text{// 主状态取同行左邻}[d_{model}]\text{、副状态取上一行同列}[d_{model}]\text{——周期内传递为主、注入周期间语义}\\
h^{ver}_{r,c,l} &= \mathrm{GSC}_{ver}\big(input_{r,c,l},\; h^{ver}_{r-1,c,l},\; h^{hor}_{r,c-1,l}\big) && \text{// 主副对调——周期间传递为主、注入周期内语义；}r{=}1\text{ 或 }c{=}1\text{ 处副/主状态以全零张量填充}\\
o_{r,c,l} &= [\,h^{hor}_{r,c,l},\;h^{ver}_{r,c,l}\,] && \text{// } [d_{model}]\oplus[d_{model}]\to[2d_{model}]\ \text{——两向状态拼接，作为该格输出与下一层 } input
\end{aligned}$$

**GSC 内部门控（以横向 cell 为例，$h^{pri}=h^{hor}$，纵向同构；式 4）**：

$$\begin{aligned}
z &= [\,h^{pri}_{t-1},\;h^{sub}_{t-1},\;x\,] && \text{// } [d_{model}]\oplus[d_{model}]\oplus[d_{in}]\to[2d_{model}{+}d_{in}]\ \text{——融合只发生在门控输入的 concat}\\
S_t,\;O_t &= \sigma(W_{s}z+b_{s}),\;\; \sigma(W_{o}z+b_{o}),\quad h_f=\tanh(W_{f}z+b_f) && \text{// } [d_{model},\,\bcancel{2d_{model}+d_{in}}]\times[\bcancel{2d_{model}+d_{in}}]\to[d_{model}]\ \text{——三门并行计算，无串行门依赖}\\
\tilde{h}^{pri}_{t-1} &= (1-S_t)\odot h^{pri}_{t-1}+S_t\odot h_f && \text{// } [d_{model}]\ \text{逐通道凸组合——selection gate 决定保留原主状态/注入融合候选的比例}\\
h^{pri}_{t} &= \tanh\big(\tilde{h}^{pri}_{t-1}\big)\odot O_t && \text{// } [d_{model}]\ \text{——output gate 控制本格最终输出}
\end{aligned}$$

### Stage 3 · 反对角线波前并行

全部 $r+c=\text{const}$ 的格点互不依赖，构成一个 slice，一次批量算完；串行步数 $H=R\times C\to R+C-1$，取 $R,C=O(\sqrt H)$ 得 $O(\sqrt H)$（总计算量仍 $O(H)$，每个格点各算一次）。机制与实现细节见 [[recurrent-acceleration-network|RAN]]。

### Stage 4 · 末行—各列配对预测

$$\begin{aligned}
H^{hor}_{rep} &= \operatorname{Repeat}(h_{hor}) && \text{// 横向隐藏态最后一行 } [N, 1, d_{model}] \xrightarrow{\text{沿列维复制}} [N, C, d_{model}]\ \text{——全局锚：含全局+最新短期语义}\\
H_{h\text{-}v} &= \operatorname{Reshape}([\,H^{hor}_{rep},\,H_{ver}\,]) && \text{// } [N, C, 2d_{model}] \to [C,\, N{\cdot}2d_{model}]\ \text{——每列一条「全局锚+该列纵向语义」配对特征}\\
\hat{Y} &= \operatorname{FC1}(H_{h\text{-}v}) && \text{// } [C,\,\bcancel{N\cdot 2d_{model}}]\times[\bcancel{N\cdot 2d_{model}},\,R_{de}\cdot d_{model}]\to[C,\,R_{de}\cdot d_{model}]\ \text{——}\\
Y &= \operatorname{FC2}\big(\operatorname{Reshape}(\hat{Y})+TFE_{de}\big) && \text{// } \to[P]\text{，}R_{de}\times C=P\ \text{——每列负责 } P/C \text{ 个未来点；norm}=1\text{ 时加回 } x_H
\end{aligned}$$

> [!note] 原文维度不自洽处
> 论文式 (2) 写 $H_{hor}\in\mathbb{R}^{L\times R\times d_{model}}$、$H_{ver}\in\mathbb{R}^{L\times C\times d_{model}}$，但式 (5) 按"横向最后一行 + 纵向全部列、均按 $C$ 列对齐"使用两者；疑为 $R/C$ 标注互换 [INFERENCE]。本页按下游用法书写。

## 实验证据

数据集（Appendix E，Table 7）：ETTh1/ETTh2、ECL（321 变量）、Traffic（862 变量）、Weather（21 变量），全部聚合到 1h 频率，6:2:2 时序切分；**所有实验只在单通道上做**（[[channel-independence|channel independence]] 单变量口径，§4.1）。任务沿用 Pyraformer 设定，记法"输入-预测"：long-range $\{168,336,720\}\times\{168,336,720\}$ 共 5 个任务/数据集（Table 3），ultra-long-range $\{720,1440\}$ 输入、$\{1440,2880\}$ 预测共 3 个任务/数据集（Table 4）；10 个基线（PatchTST/FEDformer/Pyraformer/Autoformer/Informer/Vanilla Transformer/MICN/TimesNet/DLinear/FiLM）[^src-witran-jia-2023]。

**总体口径（§4.1，论文自述）**：long-range 平均 MSE 降 5.803%（ECL 10.246% / Traffic 3.879% / ETTh1 2.519% / ETTh2 4.431% / Weather 7.939%）；ultra-long-range 降 14.275%（ECL 39.916% / Traffic 3.122% / ETTh1 14.837% / ETTh2 2.441% / Weather 11.062%）。该总体值为五个数据集级百分比的等权平均；数据集级百分比无法由表格逐任务"相对最优基线"精确复现，引用时按 §4.1 原样标注。

**逐任务实况（据 Table 3/4 复核）**：40 个任务中 35 个 MSE 最优。非最优的 5 个：Traffic 720-720（0.1408 vs FiLM 0.1383）、ETTh1 168-168（0.1105 vs FiLM 0.1091）、ETTh1 168-336（0.1189 vs FiLM 0.1187）、ETTh2 336-720（0.2373 vs Pyraformer 0.2341）、Traffic 720-1440（0.1672 vs FiLM 0.1638）；另有 ETTh2 1440-2880 仅 MSE 最优、MAE 0.4585 差于 Informer 0.4482。代表性大幅领先：ECL 1440-1440（0.2408 vs FiLM 0.4849）、ETTh1 1440-2880（0.1850 vs Informer 0.3177）。论文同时报告相同预测长度下更长输入通常带来更大提升，且 ultra 档相对提升更大（§4.1）。误差棒见 Table 9（5 次重复）。

**效率（§4.2，Figure 5）**：ECL 上固定一侧变另一侧，论文自述 WITRAN 时间/内存均最低；TimesNet 即使 batch=1 也 OOM 未入图。**全文无加速倍数或显存绝对数值，结论仅有定性曲线支撑**（Appendix I.2）。WIT（不带 RAN 的框架）与 WITRAN 的曲线差即 RAN 的效率贡献。

**消融（Appendix G，Table 8）**：五个变体中，去掉配对预测（-FC）损失最大（ECL 1440-2880：0.7406 vs 0.3359）；方向间不融合的双 cell 变体（-2DLSTM/-2DGRU）与单 cell 双向共用状态变体（-LSTM/-GRU）居中且互有胜负——但这两组变体同时更换了 cell 与预测模块（不融合则只能用 FC 预测），论文自己改用两段式解读归因，对照并不完全干净。RAN 只影响效率，论文自述对精度"无显著影响"。**ETTh1 336-720 上全部 5 个变体（0.1310–0.1347）均优于 WITRAN（0.1494）**，与"拆件则性能下降"的整体叙述相反，引用消融时需并列此反例。

**鲁棒性（Table 5）**：随机 $\varepsilon\in\{1\%,5\%,10\%\}$ 比例输入点注入 $[-2x_i,2x_i]$ 扰动，误差仅轻微上升；但存在 ECL 720-720 $\varepsilon{=}1\%$（0.2345）低于 $\varepsilon{=}0\%$（0.2478）的反例。

## 局限与边界

论文自述与实验边界（§3.1、§5、App. F/I）[^src-witran-jia-2023]：

- **周期 $C$ 需人工设定**且要求 $H\bmod C=0$；实测 $C=24$ 最佳（Table 11，ECL/Traffic）。输入长度不能整除时任务直接空缺（ECL/Traffic 的 168/336 任务在 $C{=}48$ 下缺失）。
- **Python 实现非最优**：slice 间信息等待使递归结构在 Python 下效率受限，作者计划提供 C++ 接口（§5）。
- 单通道评估，未报告多变量联合建模；数据集统一聚合到 1h，周期只能落在 12/24/48 量级。
- 基线取舍：ARIMA、简单 RNN/CNN、LogTrans、Reformer 未亲自复跑（理由为既有文献已证其弱）；效率对比只选 FiLM 等强基线。
- norm 选择规则依赖训练/验证集统计（方差差约 2 倍取 1，否则 0，App. I.1）。
- 论文未给每数据集最终超参（$d_{model}$、层数、$C$、norm 的选中值）与参数量，复现有缺口。

超参搜索空间（App. F）：$d_{model}\in\{32,\dots,1024\}$，层数 $\{1,2,3\}$，Adam lr $10^{-3}$，batch 32，早停 5 epoch（上限 25），seed 2023、每组 5 次重复，RTX A4000 16GB。

## Connections

- [[timesnet]] — WITRAN 的 1D→2D 周期重排直接受 TimesNet 启发，但改 FFT 自适应多周期为人工设定单周期；TimesNet 在其效率实验中 OOM 未入对比。
- [[patchtst]] — WITRAN 引用 PatchTST（与 TimesNet 并列）作为"patch 化信息传递可同时保留全局局部语义"的证据，并把该思路迁移到 RNN。
- [[recurrent-acceleration-network]] — RAN 技术页：反对角线分片并行的机制、实现与 $O(\sqrt L)$ 口径辨析。
- [[periodicity-modeling-in-time-series]] — 周期建模谱系中"显式—变换"路线的 RNN 变体。
- [[cross-period-sparse-forecasting]] — 同样按周期 $w$ 重排 $w$ 个子序列，SparseTSF 目的是参数压缩，WITRAN 目的是双向信息传递。
- [[cyclenet]] / [[sparsetsf]] — 均需预设周期长度；对比见周期建模谱系页。

## 引用

[^src-witran-jia-2023]: [[source-witran-jia-2023]]
