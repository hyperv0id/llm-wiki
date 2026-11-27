---
title: "尾部比例局部归一化（Trailing-Fraction Local Normalization）"
type: technique
tags:
  - time-series-forecasting
  - normalization
  - distribution-shift
  - preprocessing
created: 2026-10-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# 尾部比例局部归一化（Trailing-Fraction Local Normalization）

标准做法是用训练集全局统计量、或至少整个输入窗口的统计量来做归一化。问题在于：当序列在回看尺度上本身就不平稳时，一个覆盖全窗口的均值方差会把若干个不同的分布状态混成一团，模型拿到的是一个谁都不代表的尺度。尾部比例局部归一化只取窗口最近 $r \cdot L$ 步来估计尺度，$r \in (0, 1]$ 为 local ratio，按 log 在 $[0.001, 1]$ 上搜索，让统计量贴合最近的分布状态[^src-2606-27282-ltsf-linear-models]。

配套设计是截距的处理。局部归一化后窗口均值为零，截距不再携带信息，于是把每窗口的 scale $\sigma$ 作为附加特征拼进输入，替代截距的位置，让回归自己学一个随波动率变化的偏移；$\sigma$ 的系数为零时即退化为纯 instance normalization[^src-2606-27282-ltsf-linear-models]。

## 搜索结论

- **local 普适占优**：在 ETTh1、ETTm2、Weather、Exchange 上，local 归一化在 62–100% 的 dataset–horizon 单元被搜索选中[^src-2606-27282-ltsf-linear-models]。
- **最优尾部很短**：最优 $\log_{10} r$ 几乎总为负，集中在 $[-2.5, -0.5]$，即尾部分数 0.3%–30%；full-window 归一化（$r = 1$）几乎从不被选中[^src-2606-27282-ltsf-linear-models]。
- **最强的数据集**：ETTm2 与 Exchange 多数单元 $\log_{10} r \approx -2$，仅回看的最后几个百分点决定尺度。论文的归因是这些序列在回看尺度上非平稳，全窗口统计量会模糊本应保持区分的 regime[^src-2606-27282-ltsf-linear-models]。
- **robust 变体劣于标准归一化**：median/IQR 版本在所有测试设置下都劣于 mean/std，说明在这些基准上尾部波动与大偏差本身携带有用的尺度信息；主表因此只用 local standardization，只搜 $r$[^src-2606-27282-ltsf-linear-models]。
- **per-series 分歧**：Weather 上 local ratio 与正则强度 $\alpha$ 均呈现明显的逐通道发散，表明该数据集的 per-series 收益主要来自 $r$ 与 $\alpha$，而不是 lookback[^src-2606-27282-ltsf-linear-models]。

## 结论口径

上述搜索结论来自 Ridge 加四轴联合搜索在四个数据集（ETTh1、ETTm2、Weather、Exchange）上的超参选择频率，尚非跨模型类的对照实验[^src-2606-27282-ltsf-linear-models]。

## 关联页面

- [[instance-normalization]] — 同槽位互斥 / 退化特例：$r=1$ 即退化为全窗口实例归一化（如 RevIN），两者在同一归一化槽位选择不同统计范围，但在 Ridge 测试床上短至 0.3% 的尾部分数在强非平稳序列上更优。
- [[searchcast]] — 跨槽位正交：采用该归一化轴的线性预测管线。
- [[grouped-hyperparameter-search]] — 跨槽位正交：前者决定归一化尾部分数 $r$，后者决定跨步长与跨序列的搜索分组方式。

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
