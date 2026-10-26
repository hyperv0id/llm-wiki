---
title: "Root Mean Square Layer Normalization (RMSNorm)"
type: source-summary
tags:
  - normalization
  - transformer
  - training-efficiency
  - machine-translation
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Root Mean Square Layer Normalization (RMSNorm)

Biao Zhang、Rico Sennrich，NeurIPS 2019（arXiv:1910.07467v1，2019-10-16）。问题：LayerNorm 每步计算均值与方差的开销抵消了 RNN「收敛更快」的净收益——loss 到 7.0 时，LayerNorm 在同训练步数下已达 5.4，同训练时间下只到 5.9（Figure 1）[^src-zhang-2019-rmsnorm]。

**核心论点**：LayerNorm 起效的关键在于 re-scaling 不变性，可舍弃 re-centering。做法上仅按均方根归一化，$\bar{a}_i=a_ig_i/\mathrm{RMS}(a)$、$\mathrm{RMS}(a)=\sqrt{\frac{1}{n}\sum_i a_i^{2}}$，均值为零时与 LayerNorm 逐元素相同。该方法保留权重矩阵与数据的整体缩放不变性，失去全部 re-centering 不变性（Table 1）。梯度侧，$\partial L/\partial W$ 对输入缩放不变、与权重缩放负相关，实现隐式学习率调节（式 8–10）。pRMSNorm 仅用前 $p\%$ 分量估计 RMS，利用 RMS 的一次齐次性维持该不变性，$p=6.25\%$ 可用[^src-zhang-2019-rmsnorm]。

**证据**：WMT14 En-De（GRU RNNSearch、base Transformer）、CNN/Daily Mail、MS COCO 图文检索、CIFAR-10 四类任务上与 LayerNorm 质量相当，运行时间降 7%~64%（Table 2/3/4/6/8/10）；Transformer 不加归一化训不出，RMSNorm 与 pRMSNorm 分别快 6.9% 与 9.3%。CIFAR-10 上 LayerNorm 掉点：10.49% 对 Baseline 8.96%，RMSNorm 8.83% 基本持平；两者都不如 BatchNorm 8.25% 与 WeightNorm 8.28%。§6.4 正文称 RMSNorm 超 Baseline 0.013%，与 8.83%/8.96% 算出的 0.13 个百分点不一致。Table 5 的隐藏状态统计是 re-centering 非必需的直接支持：Baseline 全位置平均均值 −1.60、标准差 3.04，RMSNorm 为 −0.73 / 1.50，未归一化的均值反而比 Baseline 稳定[^src-zhang-2019-rmsnorm]。

**局限**：不同框架、硬件与结构下的加速比差异大（6.9%~40.8%），未覆盖低精度训练与 kernel 融合，未讨论归一化、残差结构与精度的耦合；pRMSNorm 实测未稳定提速；$k$ 较小时梯度爆炸，未给理论界。模型页见 [[rmsnorm]]。

[^src-zhang-2019-rmsnorm]: [[source-zhang-2019-rmsnorm]]
