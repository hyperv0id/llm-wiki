---
title: "尾部比例局部归一化（Trailing-Fraction Local Normalization）"
type: technique
tags:
  - time-series-forecasting
  - normalization
  - distribution-shift
  - preprocessing
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# 尾部比例局部归一化（Trailing-Fraction Local Normalization）

arXiv:2606.27282 在预处理搜索中采用的归一化方案：local normalization 仅在输入窗口**最近 $r \cdot L$ 步**计算统计量（mean/std 或 median/IQR），$r \in (0, 1]$ 为 local ratio，在 $[0.001, 1.0]$ 上按 log 搜索（§3.2）[^src-2606-27282-ltsf-linear-models]。$r < 1$ 把统计量限制在尾部区间，使模型适配最近的分布状态（distributional regime）（§3.2）[^src-2606-27282-ltsf-linear-models]。

$r = 1$ 退化为前作 full-window instance normalization（RevIN 式）；配套设计：局部归一化下窗口均值已为零、截距失去信息，改为把 per-window scale $\sigma$ 作为附加特征替代截距，让回归学习 volatility-dependent shift，$\sigma$ 系数为 0 即恢复纯 instance normalization（§3.2）[^src-2606-27282-ltsf-linear-models]。

## 搜索结论（§5.4、Fig. 7）

- **local 普适占优**：ETTh1、ETTm2、Weather、Exchange 上，local 在 62–100% 的 dataset–horizon 单元被搜索选中[^src-2606-27282-ltsf-linear-models]。
- **最优尾部很短**：最优 $\log_{10} r$ 几乎总为负、集中在 $[-2.5, -0.5]$，即尾部分数 0.3%–30%；**full-window 归一化（$r=1$）几乎从不被选中**[^src-2606-27282-ltsf-linear-models]。
- **最强的数据集**：ETTm2 与 Exchange 多数单元 $\log_{10} r \approx -2$——仅 lookback 最后几个百分点决定 scale。论文归因：序列在 $L$ 尺度上非平稳，全窗口统计量会模糊本应保持区分的 regime[^src-2606-27282-ltsf-linear-models]。
- **robust 变体全败**：median/IQR 版本在所有测试设置下均劣于 mean/std，因此主表仅用 local standardization、仅搜 $r$[^src-2606-27282-ltsf-linear-models]。
- **per-series 分歧**（§5.5、Fig. 7）：Weather 上 local ratio 逐通道发散（OT $\log_{10} r \approx -2$ vs LULL 接近 0），正则强度也出现通道间大缺口（OT $\log_{10}\alpha \approx 1\text{–}2$ vs 其他通道 $\approx 4$）——Weather 的 per-series 收益来自 $r$ 与 $\alpha$ 而非 lookback[^src-2606-27282-ltsf-linear-models]。

## 与 RevIN 的关系

[[instance-normalization|RevIN]] 对整个输入窗口归一化均值与方差以对冲分布漂移；本技术的搜索证据表明，在 Ridge 测试床上全窗口统计几乎从不是最优——短至 0.3% 的尾部分数统计量在强非平稳序列上更优[^src-2606-27282-ltsf-linear-models]。该结论口径为：Ridge + 四轴联合搜索、四个数据集（ETTh1/ETTm2/Weather/Exchange）上的超参选择频率，非跨模型类的对照实验（§5.4、Appendix D）[^src-2606-27282-ltsf-linear-models]。

## 关联页面

- [[instance-normalization]] — $r=1$ 的特例与被挑战的默认做法
- [[searchcast]] — 使用该技术的管线
- [[grouped-hyperparameter-search]] — $r$ 的搜索载体
- [[source-2606-27282-ltsf-linear-models]] — 源摘要

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
