---
title: "Spatial-Temporal Localized Convolution"
type: technique
tags:
  - traffic-forecasting
  - graph-neural-network
  - diffusion-convolution
  - spatial-temporal
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Spatial-Temporal Localized Convolution

时空局部化卷积是 [[d2stgnn|D²STGNN]] 里处理扩散分量的卷积层[^src-2206-09112-d2stgnn]。核心假设是时空局部性：只有 $k_s$ 阶邻居在过去 $k_t$ 步的信号能扩散到目标节点；按车速与采样频率，$k_s$、$k_t$ 的典型取值是 2 或 3，Figure 4 的例子取两者都为 2[^src-2206-09112-d2stgnn]。与先算空间再算时间的常规做法不同，它把时间窗直接拼进节点维，一次矩阵乘法同时消掉「邻居」与「时间步」两维。

## 三个构件

### 局部化转移矩阵

给定路网邻接 $\mathbf{A}$，先取两个方向的随机游走矩阵 $\mathbf{P}_f=\mathbf{A}/\operatorname{rowsum}(\mathbf{A})$、$\mathbf{P}_b=\mathbf{A}^\top/\operatorname{rowsum}(\mathbf{A}^\top)$[^src-2206-09112-d2stgnn]。再把 $k$ 阶转移按 $k_t$ 块并排成 $[N,k_tN]$ 的矩阵，每个块对应时间窗里的一个滞后：

$$\begin{aligned}
(\mathbf{P}^{local})^k &= \underbrace{(\mathbf{P}^k\odot(1-\mathbf{I}_N))\,\|\,\cdots\,\|\,(\mathbf{P}^k\odot(1-\mathbf{I}_N))}_{k_t} && \text{// } k_t\ \text{个 }[N,N]\ \text{按列拼}\to[N,k_tN]\ \text{——列块下标 = 滞后步数，块内 = }k\text{ 阶游走}\\
(\mathbf{P}^{lc})^k[i,\,i+k'N] &\equiv 0 && \text{// 对角块置零——节点自身的历史模式留给固有块}
\end{aligned}$$

对角块置零是这个设计里最具体的一处取舍：$\mathbf{P}^{local}[i,i+k'N]$（$k'=0,\dots,k_t-1$）被屏蔽，因为「它们描述的是目标节点自身的固有模式，会由固有模型学习」[^src-2206-09112-d2stgnn]。也就是说，卷积层被刻意禁止从自己的历史读数里取信息，那部分必须由另一条支路（GRU + 自注意力）承担。

### 特征矩阵

$$\begin{aligned}
\mathbf{X}^{lc}_t &= \big[\sigma(\mathbf{X}^{dif}_{t-k_t+1}\mathbf{W}_{k_t-1})\,\|\,\cdots\,\|\,\sigma(\mathbf{X}^{dif}_t\mathbf{W}_0)\big]^{\!\top} && \text{// } k_t\ \text{个 }[N,d]\ \text{沿行堆叠}\to[k_tN,d]\ \text{——每个滞后先过自己的线性层 + ReLU}
\end{aligned}$$

原文写作 $[\cdots]^T\in\mathbb{R}^{k_tN\times d}$，块按行堆叠、转置记号按论文的书写习惯理解[INFERENCE]。行序与转移矩阵的列块序必须对齐，否则滞后错位。

### 卷积算子

$$\begin{aligned}
\mathbf{H}^{dif}_t &= \sum_{k=1}^{k_s}(\mathbf{P}^{lc})^k\,\mathbf{X}^{lc}_t\,\mathbf{W}_k && \text{// } [N,\bcancel{k_tN}]\times[\bcancel{k_tN},d]\to[N,d]\ \text{——一次消掉邻居×滞后两维}\\
\mathbf{H}^{dif}_t &= \sum_{k=1}^{k_s}\Big[(\mathbf{P}^{lc}_f)^k\mathbf{X}^{lc}_t\mathbf{W}_{k1}+(\mathbf{P}^{lc}_b)^k\mathbf{X}^{lc}_t\mathbf{W}_{k2}+(\mathbf{P}^{lc}_{apt})^k\mathbf{X}^{lc}_t\mathbf{W}_{k3}\Big] && \text{// 三种拓扑各一套参数再按阶求和}\\
\mathbf{H}^{dif} &= \Theta*_G(\mathbf{X}^{dif})=[\cdots,\mathbf{H}^{dif}_{T-2},\mathbf{H}^{dif}_{T-1},\mathbf{H}^{dif}_T] && \text{// 逐时刻同构计算后沿时间堆叠}\to[T_h,N,d]
\end{aligned}$$

式 6 是单拓扑版本，式 8 是实际使用的三拓扑版本：forward、backward 之外再加自适应矩阵 $\mathbf{P}_{apt}=\operatorname{Softmax}(\sigma(\mathbf{E}^d(\mathbf{E}^u)^\top))\in\mathbb{R}^{N\times N}$，论文说它补充路网转移矩阵里漏掉的隐式扩散过程[^src-2206-09112-d2stgnn]。多步预测由自回归完成：拿最后一个隐状态 $\mathbf{H}^{dif}_T$ 推下一步，再喂回自己[^src-2206-09112-d2stgnn]。

## 与相邻工作的关系

- 相比 [[dcrnn|DCRNN]]：DCRNN 的扩散卷积在单步上做 $K$ 阶游走求和，时间维交给 DCGRU；这里把 $k_t$ 个滞后并进节点维，时空一次卷积，且用屏蔽对角块的方式把自身模式让给另一条支路[^src-2206-09112-d2stgnn]。
- 相比 [[gwnet|GWNet]]：自适应的 $\mathbf{P}_{apt}$ 与 GWNet 的自适应邻接同形，但这里与 forward/backward 三者在同一个卷积算子里按阶数加权[^src-2206-09112-d2stgnn]。

## 证据

- Figure 4 用一个 $k_s=k_t=2$ 的示意说明局部性：目标节点只读两跳内、两步内的信号[^src-2206-09112-d2stgnn]。
- §6.6 在 METR-LA 上把 $k_s$、$k_t$ 各从 1 扫到 5，纵轴为 h12 MAE，刻度区间 3.35–3.44；论文只说结果验证了扩散过程的时空局部性，没有给出各点数值，也没有指认最优取值[^src-2206-09112-d2stgnn]。实际配置是 $k_s=2$、$k_t=3$[^src-2206-09112-d2stgnn]。
- Table 5 去掉自适应矩阵（`w/o apt`）后 METR-LA h12 MAE 从 3.35 升到 3.40，退化幅度小于去掉估计门（3.44）或去掉整个解耦（3.51）[^src-2206-09112-d2stgnn]。
- 去掉自回归、改为在隐状态上直接一次回归多步（`w/o ar`），h12 MAE 3.39，说明单步递推在本文设置下更合适[^src-2206-09112-d2stgnn]。

## 边界

- $k_s$、$k_t$ 是数据集级超参，所有节点共用同一组值；论文没有讨论异构路网（高速与街道并存）是否需要逐节点的核大小[^src-2206-09112-d2stgnn]。
- 动态图虽然在层循环外只算一次，但它调制的是同一组 $\mathbf{P}_f,\mathbf{P}_b,\mathbf{P}_{apt}$；$\mathbf{P}^{dy}$ 在 $T_h$ 窗口内被假设为静态，属于为省算力的近似[^src-2206-09112-d2stgnn]。
- 多步预测靠单步自回归串联，误差会沿 horizon 累积；论文只给了「自回归优于一次回归」的单点对照，没有分解误差随步长的增长[^src-2206-09112-d2stgnn]。

## 相关页面

- [[d2stgnn]] — 该卷积所在的模型
- [[diffusion-inherent-signal-decoupling]] — 提供输入的 DSTF 分解
- [[dcrnn]]、[[gwnet]] — 扩散卷积与自适应邻接的来源
- [[source-2206-09112-d2stgnn]]

[^src-2206-09112-d2stgnn]: [[source-2206-09112-d2stgnn]]
