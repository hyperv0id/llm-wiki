---
title: "MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters"
type: source-summary
tags:
  - time-series-forecasting
  - long-term-forecasting
  - linear-model
  - low-rank
  - frequency-domain
  - lightweight
  - iclr-2026
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters

- **作者/机构**：Aitian Ma、Dongsheng Luo、Mo Sha（Florida International University）[^src-mixlinear]
- **出处**：ICLR 2026[^src-mixlinear]
- **分级**：Tier 2（轻量化入库，未为 `mixlinear` 新建 `source_count: 1` 独立页）

当长时预测模型把参数压缩到百参数级别时，单层共享线性层往往难以兼顾局部时域形态与全局周期趋势。MixLinear 的核心思路是先对输入序列下采样做隐式低通滤波，再拆成「局部时域形态」与「全局频域趋势」两条极简通路直接相加：时域通路把下采样序列折成二维分段矩阵，沿段内和段间两个方向做双线性因子分解，把平方级参数压成线性级；频域通路则对傅里叶低通成分施加极低秩的复数矩阵投影。两条极简通路合起来仅需 45 至 176 个参数，在 0.1K 极低资源槽位上逼近或持平既有轻量模型[^src-mixlinear]。

## 去黑话算子配方与路线差异

模型未引入改变底层范式的新型算子，而是将下采样、二维分段双线性分解与低秩复频域投影并联相加[^src-mixlinear]：

- **下采样与二维分段时域双线性分支**：输入 $X\in\mathbb{R}^L$ 按步长 $\pi$ 下采样（隐式低通）得到长度 $n=L/\pi$ 的序列，再切分为 $M$ 段（每段长 $r=L/(\pi M)$），分别沿段内维 $r$ 与跨段维 $M$ 做双线性线性映射后上采样至预测长度 $H$，把参数量从 $O(n^2)$ 线性层压至 $O(n)$（即 $dr+dM+d+M$），得到时域输出 $X_T$[^src-mixlinear]。
- **频域秩-2 复投影与双域相加**：对同一长度为 $n$ 的下采样序列做 `FFT`（保留低通频谱成分），施加秩 $n_z=2$ 的复数分解矩阵投影 $\Phi(F)=U(VF)$（其中 $U\in\mathbb{C}^{n\times 2}$、$V\in\mathbb{C}^{2\times n}$），经 `iFFT` 取实部并上采样后与时域输出直接相加：$Y=X_T+X_F$[^src-mixlinear]。相比单纯时域下采样线性模型，该分支引入了 `FFT`/`iFFT` 的 $O(n\log n)$ 计算与略高于 SparseTSF 的 MACs[^src-mixlinear]。
- **有向关系对照**：相对 [[sparsetsf|SparseTSF]] 属**瓶颈接力 $\Delta$ 与复现对齐**（保留下采样 + 共享线性骨架，拓展出段内/跨段二维双线性因子分解与频域秩-2 复投影分支，并在 Exchange 上相对 SparseTSF 显著降错且推理提速约 3.2 倍）；相对 [[source-timebase|TimeBase]] 属**同槽位互斥**（同处百参数 0.1K 级极轻量槽位：TimeBase 按主周期切段提取 $R=6$ 正交约束时间基，MixLinear 用时频双低秩分支相加，但两者缺乏同台实验对比）；相对 [[cyclenet|CycleNet]] 属**同槽位互斥**（可学习周期表残差剥离 vs 时域双线性 + 频域秩-2 基）；相对 [[ltsf-linear|LTSF-Linear]] 属**演进对照**（将 $O(LH)$ 稠密线性压缩至 0.1K 因子化表示）；相对 [[source-frets|FreTS]] 属**跨槽位正交**（全谱复数 MLP 与通道学习器 vs 下采样时域双线性 + 低秩频谱）；相对 FITS 则去掉全尺寸复数线性改为秩-2 共享基并补回时域分支[^src-mixlinear]。

## 证据

1. **定位与主结论**：45–176 参数、0.25–2.05 ms 延迟逼近极轻量基线；8 集 MSE 与 SparseTSF、FITS 同量级，Exchange 相对 SparseTSF 降错且提速约 3.2 倍，ECL/Traffic 略逊于部分高容量基线[^src-mixlinear]。
2. **真实涨点归因**：消融按维度分化——低维 ETTh1/h2 去时域支掉点最重（0.425→0.474），ETTh1 去频域支几无差（0.425→0.423）；高维 ECL/Traffic 去频域支掉点最重（Traffic 0.452→0.528），「双域互补」主要由高维集支撑[^src-mixlinear]。
3. **证据缺口**：主表**只报 MSE 不报 MAE**，RPD 仅对标 SparseTSF，缺 iTransformer、TimeMixer 与 TimeBase 基线；$n_z$、$\pi$ 敏感性在附录；参数削减存在 81% 与 11%–81% 口径出入[^src-mixlinear]。

## 相关页面

- [[sparsetsf]] · [[source-sparsetsf]] — 瓶颈接力 $\Delta$ 与复现对齐（下采样共享骨架加宽为 2D 双线性 + 频域秩-2 复投影）
- [[source-timebase]] — 同槽位互斥（同在 0.1K 参数极轻量槽位：时频双低秩分支 vs 正交约束时间基）
- [[cyclenet]] — 同槽位互斥（可学习周期表残差 vs 时域双线性 + 频域秩-2 基）
- [[ltsf-linear]] — 演进对照（将 $O(LH)$ 稠密线性压缩至 0.1K 因子化表示）
- [[source-frets]] — 跨槽位正交（全谱频域 MLP vs 下采样时域双线性 + 低秩频谱）

[^src-mixlinear]: [[source-mixlinear]]
