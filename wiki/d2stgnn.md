---
title: "D²STGNN（Decoupled Dynamic Spatial-Temporal Graph Neural Network）"
type: entity
tags:
  - traffic-forecasting
  - spatial-temporal
  - graph-neural-network
  - diffusion-convolution
  - dynamic-graph
  - decoupling
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# D²STGNN（Decoupled Dynamic Spatial-Temporal Graph Neural Network）

D²STGNN 是 Shao Zezhi 等（中科院计算所 / 华中科大 / UNSW / Aalborg）在 PVLDB 15(11): 2733–2746（2022）提出的交通预测模型[^src-2206-09112-d2stgnn]。它是 [[diffusion-inherent-signal-decoupling|DSTF]] 框架的实例：每层把传感器信号拆成扩散分量与固有分量，前者交给 [[spatial-temporal-localized-convolution|时空局部化卷积]]，后者交给逐节点 GRU 加时间维自注意力，另用自注意力掩码把静态路网转移矩阵逐时间步调制。

## 动机

论文把既有 STGNN 的做法归纳为「把交通系统建模为扩散过程」：DCRNN 用扩散卷积替换 GRU 的全连接层，Graph WaveNet 逐层堆叠门控 TCN 与 GCN[^src-2206-09112-d2stgnn]。论文的反对意见有两条。

第一，每个传感器的读数被当作扩散结果的叠加，忽略了与其它传感器无关的固有分量。Figure 2 的例子是传感器 2 早 8 点的流量由本地出发车辆与从相邻工业区、农业区扩散来的车辆共同构成，论文称二者为 hidden diffusion / hidden inherent time series[^src-2206-09112-d2stgnn]。

第二，空间依赖随时间变化：传感器 3、4 在 8 a.m. 对传感器 2 影响显著，10 a.m. 影响很小，而预定义邻接矩阵静态[^src-2206-09112-d2stgnn]。

## 机制

1. **估计门（estimation gate）**：用日内时间槽嵌入 $\mathbf{T}^D$、星期嵌入 $\mathbf{T}^W$、source/target 节点嵌入 $\mathbf{E}^u,\mathbf{E}^d$ 经两层非线性映射得到标量门 $\Lambda_{t,i}\in(0,1)$，令 $\mathbf{X}^{dif}=\Lambda\odot\mathbf{X}_l$，即「该时刻该节点的扩散分量占比」。门的输入只有嵌入，不含当前信号内容[^src-2206-09112-d2stgnn]。

2. **残差分解（residual decomposition）**：扩散块跑完，backcast 分支重建它能解释的那部分信号 $\mathbf{X}_b^{dif}$，上残差给出固有信号 $\mathbf{X}^{inh}=\mathbf{X}_l-\mathbf{X}_b^{dif}$；固有块跑完，下残差给出下一层输入 $\mathbf{X}^{l+1}=\mathbf{X}^{inh}-\mathbf{X}_b^{inh}$，保留本层没分解掉的残余[^src-2206-09112-d2stgnn]。论文称 backcast 分支对解耦是必需的：它按信息重构方式学「本块能吃下多少」，残差链接再把吃下的部分扣掉。

3. **扩散块**：时空局部化卷积，假设只有 $k_s$ 阶邻居在过去 $k_t$ 步的信号能扩散进来，把时间窗拼进节点维后一次卷积，转移矩阵同时用 forward、backward 与自适应的 $\mathbf{P}_{apt}=\operatorname{Softmax}(\sigma(\mathbf{E}^d(\mathbf{E}^u)^\top))$[^src-2206-09112-d2stgnn]。细节见 [[spatial-temporal-localized-convolution]]。

4. **固有块**：逐节点 GRU 捕捉短期依赖 → 不可训练的正弦位置编码 → 沿时间维的多头自注意力捕捉全局依赖。多步预测用滑窗自回归而不是 encoder-decoder，理由是隐藏的固有序列没有监督真值，decoder 无法训练[^src-2206-09112-d2stgnn]。

5. **动态图**：对每个时间步构造动态特征 $\mathbf{DF}^u_t$（历史通道拼接、时间槽嵌入、节点嵌入），用自注意力打分做逐元素掩码调制静态转移矩阵。没有 Gumbel、top-$k$ 或采样；为省算力，论文假设 $\mathbf{P}^{dy}$ 在一个 $T_h$ 窗口内静态，算法里在层循环外算一次[^src-2206-09112-d2stgnn]。

6. **输出与训练**：各层扩散块、固有块的 forecast 隐状态逐时刻求和，两层全连接回归，MAE 损失，curriculum learning[^src-2206-09112-d2stgnn]。

## 数据流与维度

**维度符号**：$T_h=12$ 输入步、$T_f=12$ 预测步（5 分钟采样 → 3/6/12 步对应 15/30/60 分钟）；$N$ 传感器数；$C$ 每步特征通道（四个数据集均为 1）；$d=32$ 隐维；$k_s=2$ 空间核、$k_t=3$ 时间核；$L$ 层数（原文未给具体值）；$N_D$ 日内槽位数（5 分钟采样为 288）、$N_W=7$。形状记作 $[\text{行},\text{列}]$。

### Stage 1 · 投影进隐空间

$$\begin{aligned}
\mathbf{X} &\in \mathbb{R}^{T_h\times N\times C} \xrightarrow{\text{逐时刻线性投影}} [T_h, N, d] && \text{// } [T_h,N,\bcancel{C}]\times[\bcancel{C},d]\to[T_h,N,d]\ \text{——通道维换成隐维，后续记号默认 } \mathbf{X}_l=\mathbf{X}
\end{aligned}$$

### Stage 2 · 层外先备好三种转移矩阵与动态图

静态转移矩阵按行归一化取两个方向，自适应矩阵由可学节点嵌入内积后 softmax 得到；动态图再用注意力打分逐元素调制它们。

$$\begin{aligned}
\mathbf{P}_f &= \mathbf{A}/\operatorname{rowsum}(\mathbf{A}),\quad \mathbf{P}_b=\mathbf{A}^\top/\operatorname{rowsum}(\mathbf{A}^\top) && \text{// } [N,N]\ \text{——前向/后向随机游走，行和为 1}\\
\mathbf{P}_{apt} &= \operatorname{Softmax}\!\left(\sigma(\mathbf{E}^d(\mathbf{E}^u)^\top)\right) && \text{// } [N, \bcancel{d}]\times[\bcancel{d}, N]\xrightarrow{\text{沿目标节点维 softmax}}[N,N]\ \text{——路网里没有的隐式扩散边}\\
\mathbf{X}^c &= \mathbf{X}[:,:,c]^\top && \text{// } [T_h,N]\xrightarrow{\top}[N,T_h]\ \text{——每个通道的历史读成节点×时间}\\
\mathbf{DF}^u_t &= \operatorname{Concat}\!\left[\operatorname{FC}\big(\textstyle\big\|_{c=1}^{C}\mathbf{X}^c\big),\ \mathbf{T}^D_t,\ \mathbf{T}^W_t,\ \mathbf{E}^u\right] && \text{// } [N, \bcancel{CT_h}]\xrightarrow{\text{两层 FC}}[N,d]\ \text{再与 }[d],[d],[N,d]\ \text{拼接}\to[N,4d]\\
\mathbf{P}^{dy}_{f,t} &= \mathbf{P}_f\odot\operatorname{Softmax}\!\left(\frac{(\mathbf{DF}^u_t\mathbf{W}_Q)(\mathbf{DF}^u_t\mathbf{W}_K)^\top}{\sqrt{d}}\right) && \text{// } [N,4d]\times[4d,N]\to[N,N]\xrightarrow{\odot\mathbf{P}_f}[N,N]\ \text{——注意力只当掩码，不改拓扑}\\
\mathbf{P}^{dy}_{b,t} &= \mathbf{P}_b\odot\operatorname{Softmax}\!\left(\frac{(\mathbf{DF}^d_t\mathbf{W}_Q)(\mathbf{DF}^d_t\mathbf{W}_K)^\top}{\sqrt{d}}\right) && \text{// 同上，改用 target 侧嵌入}\to[N,N]
\end{aligned}$$

$\mathbf{W}_Q,\mathbf{W}_K$ 的输入维原文未印出，按 $\mathbf{DF}$ 的 $4d$ 取，输出维按 $\sqrt{d}$ 归一化推断为 $d$[INFERENCE]。$\mathbf{P}^{dy}$ 在 $t-T_h\!:\!t$ 内视为静态，故在层循环外算一次[^src-2206-09112-d2stgnn]。

### Stage 3 · 估计门切出扩散分量

$$\begin{aligned}
\Lambda_{t,i} &= \operatorname{Sigmoid}\!\Big(\sigma\big((\mathbf{T}^D_t\|\mathbf{T}^W_t\|\mathbf{E}^u_i\|\mathbf{E}^d_i)\mathbf{W}_1\big)\mathbf{W}_2\Big) && \text{// } [4d]\times[4d,d]\times[d,1]\to(0,1)\ \text{——每个（时刻，节点）一个标量占比}\\
\mathbf{X}^{dif} &= \Lambda\odot\mathbf{X}_l && \text{// } [T_h,N,1]\text{ 广播到 }[T_h,N,d]\ \text{——整条通道按同一比例缩放}
\end{aligned}$$

### Stage 4 · 扩散块：时空局部化卷积

先沿时间窗把 $k_t$ 步的线性投影拼到行维，再让局部化转移矩阵左乘，一次消掉「邻居 × 时间窗」两维。

$$\begin{aligned}
\mathbf{X}^{lc}_t &= \big[\sigma(\mathbf{X}^{dif}_{t-k_t+1}\mathbf{W}_{k_t-1})\,\|\,\cdots\,\|\,\sigma(\mathbf{X}^{dif}_t\mathbf{W}_0)\big]^{\!\top} && \text{// } k_t\ \text{个 }[N,d]\ \text{沿行堆叠}\to[k_tN,d]\ \text{——论文写作转置，维度按 }[k_tN,d]\ \text{取}[INFERENCE]\\
(\mathbf{P}^{local})^k &= \underbrace{(\mathbf{P}^k\odot(1-\mathbf{I}_N))\,\|\,\cdots\,\|\,(\mathbf{P}^k\odot(1-\mathbf{I}_N))}_{k_t} && \text{// } k_t\ \text{个 }[N,N]\ \text{按列拼}\to[N,k_tN]\ \text{——}k\ \text{阶游走，时间维是块下标}\\
\mathbf{H}^{dif}_t &= \sum_{k=1}^{k_s}\Big[(\mathbf{P}^{lc}_f)^k\mathbf{X}^{lc}_t\mathbf{W}_{k1}+(\mathbf{P}^{lc}_b)^k\mathbf{X}^{lc}_t\mathbf{W}_{k2}+(\mathbf{P}^{lc}_{apt})^k\mathbf{X}^{lc}_t\mathbf{W}_{k3}\Big] && \text{// } [N,\bcancel{k_tN}]\times[\bcancel{k_tN},d]\to[N,d]\ \text{——三种拓扑各一套参数，按阶求和}\\
\mathbf{H}^{dif} &= \Theta*_G(\mathbf{X}^{dif})=[\cdots,\mathbf{H}^{dif}_{T-2},\mathbf{H}^{dif}_{T-1},\mathbf{H}^{dif}_T] && \text{// 逐时刻同构计算后沿时间堆叠}\to[T_h,N,d]
\end{aligned}$$

对角块 $\mathbf{P}^{local}[i,\,i+k'N]$（$k'=0,\dots,k_t-1$）被置零，因为「节点自身的模式」留给固有块[^src-2206-09112-d2stgnn]。

### Stage 5 · 上残差给出固有信号

$$\begin{aligned}
\mathbf{X}_b^{dif} &= \sigma(\mathbf{H}^{dif}\mathbf{W}_b^{dif}) && \text{// } [T_h,N,\bcancel{d}]\times[\bcancel{d},d]\to[T_h,N,d]\ \text{——backcast：本块能解释多少}\\
\mathbf{X}^{inh} &= \mathbf{X}_l-\mathbf{X}_b^{dif} && \text{// } [T_h,N,d]-[T_h,N,d]\ \text{——扣掉扩散部分，剩下固有信号}
\end{aligned}$$

### Stage 6 · 固有块：GRU → 位置编码 → 时间维自注意力

**支路 A：逐节点 GRU（短期）**，每个节点只读自己的序列：

$$\begin{aligned}
\mathbf{z}_t,\mathbf{r}_t &= \sigma(\mathbf{W}_{z,r}\mathbf{X}^{inh}_t[i,:]+\mathbf{U}_{z,r}\mathbf{H}^{inh}_{t-1}[i,:]+\mathbf{b}_{z,r}) && \text{// } [d]+[d]\to[d]\ \text{——更新/重置门，节点间不共享隐状态}\\
\tilde{\mathbf{H}}^{inh}_t[i,:] &= (1-\mathbf{z}_t)\odot\hat{\mathbf{H}}^{inh}_{t-1}[i,:]+\mathbf{z}_t\odot\hat{\mathbf{H}}^{inh}_t[i,:] && \text{// } [d]\ \text{——按时间递推，输出}\ [T_h,N,d]
\end{aligned}$$

**支路 B：时间维多头自注意力（长期）**，在 GRU 输出之上按节点独立做：

$$\begin{aligned}
head_s &= \operatorname{softmax}\!\left(\frac{\mathbf{H}_{v_i}\mathbf{W}^Q_s(\mathbf{H}_{v_i}\mathbf{W}^K_s)^\top}{\sqrt{d}}\right)\mathbf{H}_{v_i}\mathbf{W}^V_s && \text{// } [T_h,\bcancel{d}]\times[\bcancel{d},T_h]\to[T_h,T_h]\xrightarrow{\times\mathbf{H}_{v_i}\mathbf{W}^V_s}[T_h,d]\ \text{——打分矩阵是同一节点不同时刻之间}\\
\mathbf{H}^{inh}[:,i,:] &= \operatorname{Concat}(head_1,\dots,head_S)\mathbf{W}^O && \text{// } [T_h,\bcancel{Sd}]\times[\bcancel{Sd},d]\to[T_h,d]\ \text{——所有节点并行，堆成}\ [T_h,N,d]
\end{aligned}$$

两支路串行（GRU 之后接位置编码与自注意力），不是并行加权；头数 $S$ 原文未给数值[^src-2206-09112-d2stgnn]。

### Stage 7 · 下残差传递到下一层

$$\begin{aligned}
\mathbf{X}_b^{inh} &= \sigma(\mathbf{H}^{inh}\mathbf{W}_b^{inh}),\quad \mathbf{X}^{l+1}=\mathbf{X}^{inh}-\mathbf{X}_b^{inh} && \text{// } [T_h,N,d]\ \text{——本层没分解掉的残余进入第 } l+1 \text{ 层}
\end{aligned}$$

### Stage 8 · 汇总各层 forecast 并回归

$$\begin{aligned}
\mathbf{H} &= \mathbf{H}^{dif}_f+\mathbf{H}^{inh}_f=\sum_{l=0}^{L-1}\mathbf{H}^{dif,l}_f+\sum_{l=0}^{L-1}\mathbf{H}^{inh,l}_f && \text{// } L\ \text{层 × 两支路，逐时刻相加}\to[T_f,N,d]\\
\hat{\mathbf{Y}} &= \operatorname{MLP}^{(2)}(\mathbf{H}) && \text{// } [T_f,N,\bcancel{d}]\xrightarrow{\text{两层全连接}}[T_f,N,C^{out}]\ \text{——输出预测窗口全部步}\\
\mathcal{L} &= \frac{1}{T_fNC^{out}}\sum_{i,j,k}\big|\hat{\mathbf{Y}}_{ijk}-\mathbf{Y}_{ijk}\big| && \text{// MAE，全窗口全节点全通道平均}
\end{aligned}$$

两支路的 forecast 分支都是自回归：拿最后一个隐状态推下一步，再喂回自己[^src-2206-09112-d2stgnn]。Algorithm 1 的顺序是：算 $\mathbf{A}_{apt}$ → 算 $\mathbf{P}^{dy}$ → 逐层（门 → 扩散块 → 上残差 → 固有块 → 下残差 → 收集 forecast）→ 求和 → MLP → 反向[^src-2206-09112-d2stgnn]。

## 实验

四个数据集（Table 2）：METR-LA 207 节点 / 1722 边、PEMS-BAY 325 / 2694、PEMS04 307 / 680、PEMS08 170 / 548；前两个是速度、后两个是流量，采样间隔 5 分钟[^src-2206-09112-d2stgnn]。METR-LA、PEMS-BAY 按 70/20/10 划分训练/测试/验证，PEMS04、PEMS08 按 60/20/20；滑窗宽 24，前 12 步输入、后 12 步作真值[^src-2206-09112-d2stgnn]。

主结果（Table 3，3/6/12 步 MAE）：

| 数据集 | D²STGNN | 最优基线（模型 / MAE） | 出处 |
|---|---|---|---|
| METR-LA | 2.56 / 2.90 / 3.35 | DGCRN 2.62 / 2.99 / 3.44（h12 与 GMAN 3.44 并列） | [^src-2206-09112-d2stgnn] |
| PEMS-BAY | 1.24 / 1.55 / 1.85 | DGCRN 1.28 / 1.59；h12 GMAN 1.86 | [^src-2206-09112-d2stgnn] |
| PEMS04 | 17.44 / 18.28 / 19.55 | DGCRN 18.27 / GMAN 18.75 / GMAN 19.95 | [^src-2206-09112-d2stgnn] |
| PEMS08 | 13.14 / 14.21 / 15.69 | GMAN 13.80 / 14.62 / 15.72 | [^src-2206-09112-d2stgnn] |

逐格看，Table 3 里有三格没打显著性标记：PEMS-BAY h12 MAPE 4.37%（与 GMAN 4.37% 持平）、PEMS04 h12 RMSE 31.99（GMAN 30.21）、PEMS08 h12 RMSE 26.41（GMAN 25.96）[^src-2206-09112-d2stgnn]。后两格数值劣于 GMAN，与 §6.2.2「在所有数据集所有 horizon 上都最好」的自述不一致。

解耦与否的对照（Table 4）用同一模型的两个变体：D²STGNN† 把动态图换成静态图、保留 DSTF；D²STGNN‡ 在 † 基础上去掉估计门与残差分解、直接连两支路[^src-2206-09112-d2stgnn]。

| 数据集 h12 MAE | GWNet | DGCRN† | D²STGNN‡ | D²STGNN† | 出处 |
|---|---|---|---|---|---|
| METR-LA | 3.53 | 3.64 | 3.51 | 3.38 | [^src-2206-09112-d2stgnn] |
| PEMS-BAY | 1.95 | 1.99 | 1.94 | 1.85 | [^src-2206-09112-d2stgnn] |
| PEMS04 | 20.69 | 22.95 | 22.57 | 19.59 | [^src-2206-09112-d2stgnn] |
| PEMS08 | 16.67 | 17.80 | 18.01 | 15.65 | [^src-2206-09112-d2stgnn] |

METR-LA 消融（Table 5，h12 MAE）：完整模型 3.35；`switch`（固有块放前面）3.35；`w/o gate` 3.44；`w/o res` 3.37；`w/o decouple` 3.51；`w/o dg` 3.38；`w/o apt` 3.40；`w/o gru` 3.38；`w/o msa` 3.37；`w/o ar`（去掉自回归，一次回归多步）3.39；`w/o cl` 3.38[^src-2206-09112-d2stgnn]。按论文的解读，门与残差分解各自都重要，去掉两者同时（`w/o decouple`）退化最大；`switch` 与完整模型持平，说明两支路顺序可交换[^src-2206-09112-d2stgnn]。

效率（Figure 6，METR-LA，每 epoch 平均训练时间，batch 统一 32）：GWNet 61.3 s、MTGNN 47.9 s、GMAN 548.3 s、DGCRN 116.4 s、D²STGNN 98.1 s、D²STGNN† 89.3 s[^src-2206-09112-d2stgnn]。论文的解读是解耦没有显著增加开销，且比 GMAN、DGCRN 又快又准；GWNet 与 MTGNN 最快但精度更低[^src-2206-09112-d2stgnn]。

参数敏感性（§6.6）只给图：$k_s$、$k_t$ 各扫 1–5，$d$ 扫 $\{16,32,64,96,128\}$，纵轴是 METR-LA h12 的 MAE，刻度 3.35–3.44；具体每点数值原文未印出。论文未明说最优取值，实际使用 $k_s=2$、$k_t=3$、$d=32$[^src-2206-09112-d2stgnn]。

## 边界与原文瑕疵

- 论文没有独立的局限段落，也没有 future work；最接近的两处自述是 Figure 8 的可视化分析里承认「局部细节受随机噪声影响预测不准」，以及 §6.4 承认 GWNet 与 MTGNN 更高效[^src-2206-09112-d2stgnn]。
- 两类隐藏信号无真值，解耦的正确性只能由下游预测精度间接支持；论文没有报告 $\Lambda$ 的分布或分解结果的可视化[^src-2206-09112-d2stgnn]。
- 层数 $L$ 全文未给具体值，只写作符号；消融与主表都没有参数表或推理延迟[^src-2206-09112-d2stgnn]。
- 原文自相矛盾处：PEMS08 时间步正文写 17,833、Table 2 写 17856；Table 1 用 $T_p$ 表预测步、正文与 §5.4 用 $T_f$；Table 4 中 PEMS-BAY h6 的 GWNet MAE 印作 `1,63`；§5.3 式 13 的通道下标一处写 $C$ 一处写 $d$，$\mathbf{X}^c$ 的时间维写作 $T$ 而非 $T_h$；式 14 后的两个矩阵都写作 $\mathbf{P}^{dy}_{f,t}$（应为 $f,t$ 与 $b,t$）[^src-2206-09112-d2stgnn]。
- STGCN 出现在 Table 3 每一栏，但 §6.1 的基线清单里没有它，论文也没有把它归入任何类别[^src-2206-09112-d2stgnn]。

## 相关页面

- [[source-2206-09112-d2stgnn|原文摘要]]
- [[diffusion-inherent-signal-decoupling|DSTF：扩散/固有信号解耦]]
- [[spatial-temporal-localized-convolution|时空局部化卷积]]
- [[dcrnn]]、[[gwnet]] — 被论文批评的扩散过程建模范式
- [[traffic-forecasting]]

[^src-2206-09112-d2stgnn]: [[source-2206-09112-d2stgnn]]
