---
title: "SearchCast：预处理调优的 Ridge 预测器"
type: entity
tags:
  - time-series-forecasting
  - linear-model
  - hyperparameter-search
  - ridge-regression
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# SearchCast

**SearchCast** 是 arXiv:2606.27282 发布的可复现预测管线及其主表方法名：模型固定为闭式解 Ridge 回归，建模自由度全部转至预处理超参分组搜索（§1、§4.1、Table 1）[^src-2606-27282-ltsf-linear-models]。论文立场：长程预测中线性基线的"弱"反映预处理欠调而非模型容量不足（§1）[^src-2606-27282-ltsf-linear-models]。

## 模型设定与数据流

**维度符号**：$L$ 回看步数；$H$ 预测步数；$N$ 训练滑窗数；$C$ 序列（通道）数。逐 variate 独立建模（channel-independent），$W \in \mathbb{R}^{H \times L}$、$b \in \mathbb{R}^{H}$（§3.1）[^src-2606-27282-ltsf-linear-models]。

### Stage 1 · 闭式解 Ridge 权重（训练）

由 $N$ 个滑窗拼成训练矩阵 $X \in \mathbb{R}^{L \times N}$、$Y \in \mathbb{R}^{H \times N}$，求解式 (1)：

$$\begin{aligned}
W^* &= \arg\min_{W} \ \|Y - WX\|_F^2 + \alpha \|W\|_F^2 \\
W^* &= YX^{\top}\!\left(XX^{\top} + \alpha I\right)^{-1} && \text{// } [H,\bcancel{N}]\times[\bcancel{N},L]\times\left([L,\bcancel{L}]\right)^{-1}\to[H,L]\ \text{——}N\text{ 个滑窗一次消去，无迭代优化，单 GPU 上训练近乎瞬时}
\end{aligned}$$

$\alpha$ 不参与外层搜索：每个 trial 在 21 个对数等距值（$[10^{-6}, 10^3]$）中按验证损失选出，作为"不增加搜索成本的内层优化回路"（§3.2）[^src-2606-27282-ltsf-linear-models]。

### Stage 2 · 逐窗口前向

$$\hat{y} = W^* x && \text{// } [H,\bcancel{L}]\times[\bcancel{L},1]\to[H,1]\ \text{——单序列一步线性读出}$$

局部归一化下截距失去信息（窗口局部均值已为零），改为把 per-window scale $\sigma$ 作为附加特征拼入 $x$，让回归学习 volatility-dependent shift；$\sigma$ 系数为 0 即退化为纯 instance normalization（§3.2）[^src-2606-27282-ltsf-linear-models]。

### Stage 3 · 预处理四轴（建模自由度所在）

| 轴 | 搜索空间（原文） |
|---|---|
| Normalization | global / local × standardization / robust；local 在窗口最近 $r \cdot L$ 步上统计，$r \in [0.001, 1]$（log） |
| Context length | $L \in [32, 2048]$（log），上界受数据量与 horizon 限制 |
| Augmentation | 时域高斯噪声 / 频域 Fourier 系数扰动 / 不加噪，$\sigma \in [0.001, 0.5]$（log） |
| Regularization | $\alpha$ 内层 21 点网格，不占外层预算 |

搜索按 $(g_h, g_s)$ 分组执行（[[grouped-hyperparameter-search]]），每单元 20 个 Optuna-TPE trials，$k=3$ expanding-window CV，test 不参与搜索（§3.3、§4.1）[^src-2606-27282-ltsf-linear-models]。

## 主结果（Table 1，MSE，8 benchmark）

- 线性类内（对照 OLS/FITS/DLinear，取 Toner & Darlow 带 instance normalization 的数字，$L=720$）：7/8 数据集平均最优；ETTh2 与 FITS 打平（0.334 vs 0.333）；对 OLS 降幅 ETTm2 4.8%、ETTh1 4.3%、Exchange 14.8%（§4.2）[^src-2606-27282-ltsf-linear-models]。
- 对非线性基线（PatchTST/iTransformer/TimeMixer/TimesNet/Autoformer，取原论文发表的 best lookback）：6/8 平均最优；两处失利均输给 PatchTST——Electricity（0.159 平）、Traffic（0.404 vs 0.391），恰为通道数最多的两个数据集（321、862）；其余六项领先 4%（Weather）至 16%（ETTm2）（§4.2）[^src-2606-27282-ltsf-linear-models]。
- 计算成本仅定性口径：单 trial "a few milliseconds on a GPU"，整体 "orders of magnitude cheaper to train"；无墙钟/FLOPs/参数量数字（§1、§3.1、Appendix E）[^src-2606-27282-ltsf-linear-models]。

## BDS 残差诊断（Table 5，Appendix C）

BDS 检验（embedding dimension 2，$B=5000$ bootstrap，按 (dataset, L, H, seed, channel) 配对）显示：PatchTST 在六个标准 benchmark 上残差更接近白噪声（mean $\Delta = -7.83$，18/24 单元显著），表明其确实捕获了线性模型无法捕获的非线性模式，但未改变精度排名；而在其获胜的两个大数据集上仅 1/8 单元显著，即优势并非来自非线性结构捕获[^src-2606-27282-ltsf-linear-models]。DLinear 作为线性对照 $\Delta \approx 0$，验证该检验不在无差异处报差异。论文总论：这些 benchmark 上决定精度的主要是长程线性信号（Appendix A）经合适 context length 的利用（Appendix B）[^src-2606-27282-ltsf-linear-models]。

persistence 基线（$\hat{y}_{t+h} = y_t$）在 $H=720$ 上比 Ridge 差 1.4–5.9 倍（ETTm1 5.87、ETTh1 2.56、Weather 1.81、Exchange 1.75、ETTh2 1.56、ETTm2 1.44），说明模型在做实质预测而非复读近期值（Table 4）[^src-2606-27282-ltsf-linear-models]。

## 对 foundation model 的立场（related-work 口径）

论文认为 Chronos/TimesFM/Moirai/DAM 追求广覆盖而非单数据集精度，通用性需要大预训练语料与远超逐数据集拟合的算力，在标准监督 benchmark 上的精度却常与调优后的逐数据集基线相当——"对追加成本的小边际回报"（§2）[^src-2606-27282-ltsf-linear-models]。该论断为立场陈述，非本文实验结论。

## 局限与发布状态（Appendix D、§1）

论文自述：结论限于标准数值长程 benchmark 与点估计 MSE（小差距应视作平手）；非线性基线取原论文配置而非联合重调；per-series 搜索成本随通道数增长；将同一预处理搜索用于更深模型属 future work[^src-2606-27282-ltsf-linear-models]。SearchCast 正文链接为占位符 "(URL)"，所称交互式在线 demo 与代码库均未给出实际地址[^src-2606-27282-ltsf-linear-models]。

## 关联页面

- [[source-2606-27282-ltsf-linear-models]] — 源摘要
- [[ltsf-linear]] — 所挑战的线性基线家族（DLinear/NLinear）
- [[grouped-hyperparameter-search]] — $(g_h, g_s)$ 搜索方案
- [[trailing-fraction-local-normalization]] — 其归一化轴的搜索结论
- [[lookback-horizon-scaling]] — 其 context length 轴的搜索结论
- [[instance-normalization]] — $r=1$ 退化所得的 full-window instance normalization
- [[channel-independence]] — 其逐 variate 建模范式

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
