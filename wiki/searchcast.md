---
title: "SearchCast：预处理调优的 Ridge 预测器"
type: entity
tags:
  - time-series-forecasting
  - linear-model
  - hyperparameter-search
  - ridge-regression
created: 2026-10-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# SearchCast

SearchCast 是论文 *How Good Can Linear Models Be for Time-Series Forecasting?* 提出的可复现线性预测管线，也是主表中的方法名称：模型本身固定为闭式解的 Ridge 回归，全部建模自由度都转移到预处理超参的分组搜索上。其核心立场在于——长程预测里线性基线显得弱，是预处理未调优，而非模型容量不足[^src-2606-27282-ltsf-linear-models]。

## 模型设定与数据流

**维度符号**：$L$ 回看步数；$H$ 预测步数；$N$ 训练滑窗数；$C$ 序列（通道）数。逐 variate 独立建模（channel-independent），$W \in \mathbb{R}^{H \times L}$、$b \in \mathbb{R}^{H}$[^src-2606-27282-ltsf-linear-models]。

### Stage 1 · 闭式解 Ridge 权重（训练）

由 $N$ 个滑窗拼成训练矩阵 $X \in \mathbb{R}^{L \times N}$、$Y \in \mathbb{R}^{H \times N}$：

$$\begin{aligned}
W^* &= \arg\min_{W} \ \|Y - WX\|_F^2 + \alpha \|W\|_F^2 \\
W^* &= YX^{\top}\!\left(XX^{\top} + \alpha I\right)^{-1} && \text{// } [H,\bcancel{N}]\times[\bcancel{N},L]\times\left([L,\bcancel{L}]\right)^{-1}\to[H,L]\ \text{——}N\text{ 个滑窗一次消去，无迭代优化}
\end{aligned}$$

$\alpha$ 不占外层搜索预算：每个 trial 在 21 个对数等距值（$[10^{-6}, 10^3]$）中按验证损失选出[^src-2606-27282-ltsf-linear-models]。

### Stage 2 · 逐窗口前向

$$\begin{aligned}
\hat{y} &= W^* x && \text{// } [H,\bcancel{L}]\times[\bcancel{L},1]\to[H,1]\ \text{——单序列一步线性读出}
\end{aligned}$$

局部归一化下窗口均值已为零、截距失去信息，改为把 per-window scale $\sigma$ 作为附加特征拼入 $x$，让回归学习随波动率变化的偏移（volatility-dependent shift）；$\sigma$ 系数为 0 即退化为纯 instance normalization[^src-2606-27282-ltsf-linear-models]。

### Stage 3 · 预处理四轴（建模自由度所在）

| 轴 | 搜索空间（原文） |
|---|---|
| Normalization | global / local × standardization / robust；local 在窗口最近 $r \cdot L$ 步上统计，$r \in [0.001, 1]$（log） |
| Context length | $L \in [32, 2048]$（log），上界受数据量与 horizon 限制 |
| Augmentation | 时域高斯噪声 / 频域 Fourier 系数扰动 / 不加噪，$\sigma \in [0.001, 0.5]$（log） |
| Regularization | $\alpha$ 内层 21 点网格，不占外层预算 |

搜索按 $(g_h, g_s)$ 分组执行（见 [[grouped-hyperparameter-search]]），每单元 20 个 Optuna-TPE（树结构 Parzen 估计器贝叶斯优化）trials，$k=3$ expanding-window CV，test 不参与搜索[^src-2606-27282-ltsf-linear-models]。

## 证据

1. **主结论**：搜索后的 Ridge 在线性类内八项中七项最优，对 PatchTST 等非线性基线 6/8 最优；容量未变，增益来自预处理[^src-2606-27282-ltsf-linear-models]。
2. **归因**：贡献最大的是 context length——只扫 $L$ 时 ETT/Weather 误差随 $L$ 增到 2000 降三到五成，ETTm2 中等长度即饱和；归一化与增广量级更小[^src-2606-27282-ltsf-linear-models]。
3. **缺口**：输掉的是通道最多的 Electricity 与 Traffic；非线性基线取原文配置、未联合重调，非同口径；非线性残差检验（Brock–Dechert–Scheinkman, BDS 检验，用于判断残差是否残留非线性结构）显示 PatchTST 确有多余非线性结构，却不改变排名[^src-2606-27282-ltsf-linear-models]。

## 对 foundation model 的立场

论文认为 Chronos/TimesFM/Moirai/DAM 这类模型追求广覆盖而非单数据集精度：通用性需要大预训练语料与远超逐数据集拟合的算力，而在标准监督 benchmark 上的精度常与调优后的逐数据集基线相当。这是立场陈述，不是本文的实验结论[^src-2606-27282-ltsf-linear-models]。

## 局限

论文自述：结论限于标准数值长程 benchmark 与点估计 MSE；非线性基线取原论文配置而非联合重调；逐序列搜索成本随通道数增长，故取较粗分组；把同一预处理搜索用于更深模型属后续工作[^src-2606-27282-ltsf-linear-models]。

## 关联页面

- [[source-2606-27282-ltsf-linear-models]] — 本源摘要
- [[grouped-hyperparameter-search]] — 跨槽位正交：本文采用的 $(g_h, g_s)$ 分组搜索组织方式
- [[trailing-fraction-local-normalization]] — 跨槽位正交：归一化轴的搜索机制与结论
- [[lookback-horizon-scaling]] — 跨槽位正交：context length 轴的幂律规律
- [[instance-normalization]] — 退化特例：$r=1$ 且 $\sigma$ 系数为 0 即退化为全窗口 instance normalization
- [[channel-independence]] — 同构/继承：沿用逐变量独立建模范式
- [[ltsf-linear]] — 同槽位互斥：以闭式 Ridge + 预处理搜索统一并对照 DLinear/NLinear 家族

出版信息：arXiv:2606.27282v2（2026-06-29，Preprint）。

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
