---
title: "Lookback–Horizon Scaling：最优回看长度与预测长度的幂律关系"
type: concept
tags:
  - time-series-forecasting
  - lookback-window
  - context-length
  - ltsf
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# Lookback–Horizon Scaling

**Lookback–horizon scaling** 指 benchmark 上"最优回看长度 $L^*$ 如何随预测长度 $H$ 变化"的经验规律。arXiv:2606.27282 逐 (series, horizon) 搜索测得 $L^*$ 并拟合幂律 $L^* = a \cdot H^{b}$（§5.1，Fig. 3）：指数 $b$ 跨正负且跨越一个数量级，直接挑战"更长 horizon 需要更长历史"的惯例（Abstract、§5.1）[^src-2606-27282-ltsf-linear-models]。

## 逐数据集指数（Fig. 3b）

| 数据集 | $b$ | 数据集 | $b$ |
|---|---|---|---|
| ETTm2 | +0.46 | Weather | 0.05 |
| ETTm1 | +0.23 | Electricity | 0.02 |
| ETTh2 | +0.11 | Exchange | −0.19 |
| ETTh1 | 0.09 | Traffic | −0.19 |

ETTm2 是唯一强符合"长 horizon 要长历史"直觉的数据集（但次线性）；ETTm1/ETTh2 处于弱正区（$b \approx 0.1\text{–}0.2$）；Weather/Electricity 近乎平坦（$|b| < 0.1$，稳定在约 $10^3$ 步）；Exchange/Traffic 为负——长 horizon 反而偏好短 context，与"远端历史主动误导模型"的非平稳性一致（§5.1）[^src-2606-27282-ltsf-linear-models]。

> [!note] 正文与图的口径差异
> 正文把 ETTm1 归入 $b \approx 0.1\text{–}0.2$ 的弱正区，但 Fig. 3(b) 标注 ETTm1 $= 0.23$、ETTh1 $= 0.09$。两处并列报告，本文页面以图上数字为准。

**推论（§5.1）**：社区默认 $L = 96$ "在两个相反方向上同时犯错"——对 Weather/Electricity 少给约一个数量级的历史，对 Exchange/Traffic 的长 horizon 则过量（最优值跌破 96）[^src-2606-27282-ltsf-linear-models]。

## 只变 $L$ 的收益分层（Table 3，universal default，$H=720$，中位 MSE）

| Regime | 数据集 | $L: 96 \to 2000$ |
|---|---|---|
| strong | ETTh1 / ETTh2 / Weather | 总降幅 36%–53%（ETTh1 −39%、ETTh2 −53%、Weather −36%） |
| plateau | ETTm2 | −16%，大部分降幅在 $L \approx 192\text{–}336$ 达成 |
| flat | ETTm1 | 几乎无改善 |

两条配套结论（Appendix B）：固定 $L=96$ 时 strong regime "在任何其他预处理被调优之前就已放弃大部分可达精度"；将 $L$ 固定为短值后重跑完整预处理搜索无法找回长 context 收益——收益源于 context length 本身[^src-2606-27282-ltsf-linear-models]。属于哪个 regime 无法从数据集元信息预判，必须实测[^src-2606-27282-ltsf-linear-models]。

## 长程信号真实性（Table 2，Appendix A）

对 lag $k = 720$、经 harmonic 去季节化的序列，六个标准 benchmark 仍保留 0.39–0.73 的跨通道平均 Pearson 相关（ETTm2 0.725、Exchange 0.532、ETTh2 0.526、ETTm1 0.435、Weather 0.428、ETTh1 0.387），95% bootstrap CI 远离 0[^src-2606-27282-ltsf-linear-models]。表明长 context 捕获了真实的远端线性信息，而非仅重复表征周期——否则 trend-seasonal 分解或频域算子即可低成本复现（Appendix A）[^src-2606-27282-ltsf-linear-models]。ETTm1/ETTm2 去季节化后的相关反而高于原始相关：强日周期掩盖了更慢的结构[^src-2606-27282-ltsf-linear-models]。

权重可视化（Fig. 6）展现微观机制：Exchange 的权重几乎全在近期 lag；Traffic 在 168、336… 处有周周期带；Weather 与 ETTm 的最大权重常在数百步前——长 $L^*$ 的价值主要是"取到少数相位对齐的观测"，而非利用全部历史（§5.3）[^src-2606-27282-ltsf-linear-models]。

## per-series 分歧（Fig. 3c）

Weather 是唯一 per-series 指数紧密聚在中位数附近的数据集；ETTh2、Electricity、Traffic 的 per-series 指数跨越 0——同一数据集内部分通道偏好随 horizon 增长 context、部分偏好缩短（§5.1）[^src-2606-27282-ltsf-linear-models]。

## 与既有论断的关系

- [[lstf]] 既有论断："Transformer 随回看窗口增长性能不提升、LTSF-Linear 显著提升"（Zeng et al. 2022）；PatchTST 是唯一随 lookback 增大持续降 MSE 的 Transformer。本页补充：即便在线性模型内部，最优 $L$ 也不是常数，而是 series-specific 且对 horizon 非单调[^src-2606-27282-ltsf-linear-models]。
- 该结论口径在 Ridge + 搜索式预处理下测得（论文自述局限，Appendix D），推广到其他模型类属 future work[^src-2606-27282-ltsf-linear-models]。

## 关联页面

- [[searchcast]] — 测得该规律的管线
- [[source-2606-27282-ltsf-linear-models]] — 源摘要
- [[ltsf-linear]] / [[patchtst]] — lookback 敏感性的既有对照
- [[lstf]] — LSTF 问题设定

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
