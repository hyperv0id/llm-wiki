---
title: "LCR — Laplacian Convolutional Representation for Traffic Time Series Imputation"
type: source-summary
tags:
  - low-rank
  - matrix-completion
  - spatio-temporal-imputation
  - laplacian-regularization
  - fft
  - traffic
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# LCR — Laplacian Convolutional Representation for Traffic Time Series Imputation

**作者:** Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier（通讯作者）, Lijun Sun（Polytechnique Montreal / McGill University / University of Central Florida）
**版本:** arXiv:2212.01529v3 [cs.LG]，水印日期 2024-06-24。PDF 为 IEEE 期刊双栏排版（Index Terms、作者简介、"Senior Member, IEEE" 标注齐全），但 PDF 内未出现 "IEEE Transactions on Knowledge and Data Engineering" 或接收信息字样；TKDE 2024 著录来自用户，未在 PDF 内核实
**raw:** `raw/chen-laplacian-convolutional-representation-arxiv-2022.pdf`

## 核心论点

论文面向交通时空数据插补，主张同时刻画时间序列的全局趋势（日/周周期等循环模式）与局部趋势[^src-lcr]。论文提出 LCR 模型：以 circulant matrix nuclear norm 刻画全局低秩（承袭 CircNNM），并引入 Laplacian kernelized temporal regularization 刻画局部趋势——将度 2τ 的无向 circulant 图之 Laplacian 矩阵第一列定义为 Laplacian kernel（Definition 1），正则写成 circular convolution（式 5），经卷积定理与 Parseval 定理转到频域（式 8）[^src-lcr]。论文证明该凸模型可用两块 ADMM 求解：x-子问题化为频域复空间 ℓ1 最小化、有 shrinkage 闭式解（Lemma 2），每次迭代 FFT 复杂度 O(T log T)；多元扩展 LCR-2D 用 circulant tensor nuclear norm 与二维卷积核、二维 FFT（Algorithm 2）[^src-lcr]。论文自述是首个把 Laplacian 核时域正则与 circular convolution 结合、从而可用 FFT 的方案（第 2.2 节，作者自述）[^src-lcr]。

## 实验结果（作者报告）

单变量 Portland 速度序列 95% 缺失下 LCR MAPE 2.13%，优于 CircNNM 2.47% / ConvNNM 2.33%（Fig. 5）；体积序列 95% 缺失 LCR 19.59% vs ConvNNM 33.18%（Fig. 6）；HighD/CitySim 速度场重建中 LCR-2D 在全部缺失率下优于 LCR_N、CTNNM、QVC、LKC、LRMC、HTF、HaLRTC、LRTC-TNN（Table 1）；PeMS-4W 大规模插补（11,160 传感器×4 周，约 9000 万观测，Table 2）中 LCR-2D/LCR_N/LCR 优于 CircNNM、LRMC、HaLRTC、LRTC-TNN、NoTMF[^src-lcr]。

## 范围与局限

- circulant 结构隐含序列首尾相连假设，论文自认这是缺点，以翻转操作（Remark 1、Fig. 7）缓解；日周期强的数据可省略翻转（第 5、6.2 节）[^src-lcr]。
- 论文未设独立局限性章节；γ、λ、η、τ 等超参按数据集调节（第 6.1.3 节）[^src-lcr]。

## 相关页面

[[lcr]] · [[laplacian-kernel-temporal-regularization]] · [[loft]] · [[fence]] · [[mts-imputation-taxonomy]]

[^src-lcr]: [[source-lcr]]
