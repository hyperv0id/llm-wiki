---
title: "LCR（Laplacian Convolutional Representation）"
type: technique
tags:
  - low-rank
  - matrix-completion
  - spatio-temporal-imputation
  - laplacian-regularization
  - fft
  - convex-optimization
  - traffic
created: 2026-08-29
last_updated: 2026-08-29
source_count: 4
confidence: medium
status: active
---

# LCR（Laplacian Convolutional Representation）

**LCR**（Laplacian Convolutional Representation，Laplacian 卷积表示）是 Chen、Cheng、Cai、Saunier、Sun 提出的交通时间序列插补模型（arXiv:2212.01529v3，2024-06-24 水印；TKDE 2024 著录来自用户，未在 PDF 内核实）[^src-lcr]。与 wiki 中大量深度插补方法（[[csdi|CSDI]]、[[pristi|PriSTI]]、[[grin|GRIN]] 等）不同，LCR 是一个非深度的凸优化方法：不训练神经网络，而是求解一个带 Laplacian 核时域正则的低秩补全问题，并用 FFT 在频域求解[^src-lcr]。

## 问题与动机

交通数据因传感器故障等运营问题天然稀疏，插补需要同时利用时空相关性[^src-lcr]。论文将交通时间序列的模式分为两类：全局趋势（日/周周期的循环模式，可由低秩模型刻画）与局部趋势（短时平滑性）[^src-lcr]。论文认为现有低秩补全路线各有缺口：(i) 经典 LRMC 的重构对矩阵行列置换不变，无法刻画序列动态；(ii) Hankel/卷积矩阵等代数结构的低秩模型规模大、局限于中小规模问题；(iii) CircNNM 可经 FFT 高效求解，但论文指出 circulant 结构无法刻画局部趋势，且核范数最小化的默认结构不保证局部平滑[^src-lcr]。LCR 的做法是把刻画局部趋势的时域正则加到 circulant 低秩模型上，同时保持 FFT 求解效率[^src-lcr]。

## 模型：全局低秩 + 局部正则

单变量情形下，论文提出的目标函数为（式 12、13，第 4.2.2 节）：

$$\min_x \ \|C(x)\|_* + \gamma \cdot R_\tau(x) \quad \text{s.t.} \quad \|P_\Omega(x-y)\|_2 \le \epsilon$$

- **全局项**：$\|C(x)\|_*$ 是 circulant matrix nuclear norm。Lemma 1（第 4.2.3 节）给出 $\|C(x)\|_* = \|\mathcal{F}(x)\|_1$——circulant 矩阵可被酉矩阵（DFT 矩阵）对角化，其奇异值即 $\mathcal{F}(x)$ 的分量模长，因此核范数可由 FFT 一次算出[^src-lcr]。
- **局部项**：$R_\tau(x)$ 是 Laplacian kernelized temporal regularization，写成 circular convolution $\|\ell \star x\|_2^2/2$ 的形式，在频域等价于 $\|\mathcal{F}(\ell)\circ\mathcal{F}(x)\|_2^2/(2T)$（式 5、8）。该机制详见 [[laplacian-kernel-temporal-regularization]][^src-lcr]。
- **观测约束**：$\epsilon \ge 0$ 是容差。论文说明交通数据通常含噪，用松弛约束 $\|P_\Omega(x-y)\|_2 \le \epsilon$ 替代严格观测约束，使重构在拟合观测的同时滤除噪声（第 4.2.2、5 节）[^src-lcr]。

## 求解：频域 ADMM

论文用两块 ADMM 求解该凸优化问题（式 14–16，第 4.2 节）：引入辅助变量 $z$ 承载观测项，交替更新 $x, z, w$[^src-lcr]。

- **x-更新**：借助 Lemma 1 与 Parseval 定理，x-子问题化为频域复空间上的 ℓ1 范数最小化，逐分量有 shrinkage 闭式解（Lemma 2，式 24、28）：$\hat{x}_t := \frac{\hat{h}_t}{|\hat{h}_t|}\max\{0, |\hat{h}_t| - 1/\delta_t\}$，其中 $\delta_t = (\gamma|\hat{\ell}_t|^2 + \lambda)/T$[^src-lcr]。
- **z-更新**：由观测项的梯度为零给出闭式解（式 31）；权重 $\eta$ 取 $c\cdot\lambda$（$c\in\{10^2, 10^3\}$）以保留观测信息，$\eta\to+\infty$ 退回强观测约束（第 4.2.4 节）[^src-lcr]。
- **复杂度**：每次迭代以 FFT 为主，复杂度 $O(T\log T)$；论文对比 ConvNNM 需对 $T\times\tilde\tau$ 卷积矩阵做奇异值阈值化（时间复杂度 $O(\tilde\tau^2 T)$），且卷积矩阵越大开销越高（第 4.2.5 节）。Fig. 3 在长度 $2^{10}$ 至 $2^{20}$ 的生成数据上（默认 50 次迭代）报告了 LCR 相对 ConvNNM 的运行时间优势[^src-lcr]。

## 多元扩展：LCR-2D

多元情形下，论文提出 LCR-2D（第 4.3 节）：核范数换成 circulant tensor nuclear norm（Definition 2，按 Tucker 格式/高阶 SVD 定义），空间与时间依赖用可分离核 $K \triangleq \ell_s \ell^\top$（空间核 $\ell_s = e_1$，即单位阵第一列）的二维 circular convolution（Definition 3）表达，整体经二维 FFT 求解（Algorithm 2）[^src-lcr]。论文说明 $\ell_s = e_1$ 本身不提供空间依赖，空间建模由 circulant 算子的核范数隐式完成；如需显式空间相关性也可换用 Laplacian 核（第 4.3.2 节）[^src-lcr]。此外论文还考虑两个变体：LCR_N（对 N 条序列独立跑单变量 LCR）与 LCR（对 N×T 矩阵向量化后跑单变量 LCR）（第 6.2 节）[^src-lcr]。

## 实验（作者报告）

1. **单变量交通速度/体积（Portland 双环检测器，15 分钟分辨率，3 天、长度 288）**：速度序列在 80%、90% 缺失下 LCR 的重构 MAPE 分别为 1.42%、1.69%（Fig. 4）；95% 缺失（仅 14 个观测）时 LCR（$\tau=2, \gamma=5\lambda$）MAPE 2.13%，对照 CircNNM 2.47%、ConvNNM 2.33%、ConvNNM+（ConvNNM 加同款时域正则）2.30%（Fig. 5）[^src-lcr]。强日周期的体积序列 95% 缺失下，CircNNM 36.31%、ConvNNM 33.18%、LCR 19.59%（Fig. 6）[^src-lcr]。
2. **速度场重建（HighD video #46：142×595×3；CitySim 高速路段：126×442×3，随机掩 30%/50%/70% 轨迹）**：LCR-2D 在两个数据集、全部缺失率下 MAPE/RMSE 均为 Table 1 中最优（如 HighD 30%：3.57/1.41；CitySim 30%：8.88/2.71），对照 LCR_N、CTNNM（去掉正则的特例）、QVC、LKC（去掉核范数的特例）、LRMC、HTF、HaLRTC、LRTC-TNN；论文据此归因于 circulant tensor nuclear norm 与 Laplacian 核时空正则两者缺一不可（第 6.1.4 节）[^src-lcr]。超参：$\eta=10^2\lambda$；HighD $\lambda=10^{-3}NT$、$\gamma=\lambda$、$\tau=1$（30%）/$2$（50%/70%）；CitySim $\lambda=10^{-4}NT$、$\gamma=\lambda$、$\tau=3$（第 6.1.3 节）[^src-lcr]。
3. **大规模插补（PeMS-4W：11,160 个传感器、2018 年前 4 周、5 分钟分辨率，矩阵 11160×8064，约 9000 万观测；随机掩 30%/50%/70%/90%）**：LCR-2D、LCR_N、LCR 三变体精度接近且优于 CircNNM、LRMC、HaLRTC、LRTC-TNN、NoTMF；如 90% 缺失下 LCR-2D MAPE/RMSE 为 3.19/3.05，CircNNM 为 5.34/3.96，LRTC-TNN 为 3.40/3.10（Table 2）[^src-lcr]。论文将复杂度对比写作 $O(\min\{N^2T, NT^2\})$（SVD）对 $O(NT\log(NT))$（FFT）（第 6.2 节）[^src-lcr]。

## 定位与后续引用

论文将 LCR 的建模思路定位为承袭 CircNNM 与 ConvNNM（Liu & Zhang、Liu 的工作）：在 CircNNM 之上加局部时域正则；相对 ConvNNM 的差异是保留 FFT 快速实现，并新增翻转操作处理序列首尾相关（第 1 节）[^src-lcr]。论文自述："据我们所知，我们是首个提出与 circular convolution 结合的 Laplacian kernelized temporal regularization、从而可用 FFT 的方案"（第 2.2 节，作者自述，中译）[^src-lcr]。

在 wiki 已 ingest 的后续论文中，LCR 被 [[fence|FENCE]]（AAAI 2026）与 [[loft|LOFT]]（KDD 2026）列为 PEMS 设置下的对比基线[^src-fence][^src-loft]。注意两文的对比表格中 LCR 数字是各自复现口径，非 LCR 原文实验设置（原文使用 Portland/HighD/CitySim/PeMS 速度数据与 MAPE/RMSE）[^src-lcr]。

## 自述边界

- circulant 结构隐含序列首尾相连的假设，论文在 Remark 1 自认这是"真实数据分析中的缺点"，处理办法是把序列翻转拼接（$\mathbf{x}_{new} = [\mathbf{x}; J_T\mathbf{x}]$，式 4）或多元情形用 2N×2T 块矩阵翻转（Fig. 7）；对日周期强的数据（Portland 速度/体积、PeMS）论文说明可不做翻转（第 5、6.2 节）；速度场实验（HighD/CitySim）则使用翻转后的矩阵作为输入（第 6.1.3 节）[^src-lcr]。
- 论文未设独立的局限性章节；$\gamma, \lambda, \eta, \tau$ 等超参按数据集调节（第 6.1.3 节）[^src-lcr]。

## 相关页面

- [[laplacian-kernel-temporal-regularization]] — 局部趋势正则机制子页（Definition 1 与频域形式）
- [[source-lcr]] — 源文件摘要
- [[loft]] — LOFT（KDD 2026），以 LCR 为判别式基线；其 [[low-rank-prior-estimation|低秩先验]] 与 LCR 的低秩刻画同谱系但走神经参数化路线[^src-loft]
- [[fence]] — FENCE（AAAI 2026），扩散引导插补，同样以 LCR 为基线[^src-fence]
- [[imputeformer]] — ImputeFormer（KDD 2024），把低秩归纳偏置做进 Transformer 的深度路线[^src-2312-01728]；与 LCR 的非深度低秩补全形成对照
- [[csdi]]、[[pristi]]、[[grin]] — 深度扩散/GNN 插补代表，在 FENCE/LOFT 实验中与 LCR 同台对比[^src-fence][^src-loft]
- [[mts-imputation-taxonomy]] — MTSI 深度学习综述的分类框架；LCR 为非深度方法，未被该综述收录
- [[directed-graph-laplacian-regularizer]] — wiki 中另一处图 Laplacian 正则（有向图、谱滤波语境）

[^src-lcr]: [[source-lcr]]
[^src-fence]: [[source-fence]]
[^src-loft]: [[source-loft]]
[^src-2312-01728]: [[source-2312-01728]]
