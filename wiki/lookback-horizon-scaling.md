---
title: "Lookback–Horizon Scaling：最优回看长度与预测长度的幂律关系"
type: concept
tags:
  - time-series-forecasting
  - lookback-window
  - context-length
  - ltsf
created: 2026-10-24
last_updated: 2026-11-27
source_count: 3
confidence: medium
status: active
---

# Lookback–Horizon Scaling

长程预测里有一个几乎没人验证的默认做法：回看窗口取固定的几十步，预测多长都用同一个。SearchCast 把回看长度当作可搜索的超参，逐序列、逐预测长度地搜出最优值后拟合幂律，发现最优回看对预测长度的依赖没有统一方向——幂律指数从明显为正跨到明显为负，等于说"更长的预测需要更长的历史"这条惯例在相当一部分数据上根本不成立。这条规律本身也成了诊断工具：指数符号反映序列有多平稳[^src-2606-27282-ltsf-linear-models]。

## 逐数据集指数

| 数据集 | $b$ | 数据集 | $b$ |
|---|---|---|---|
| ETTm2 | +0.46 | Weather | 0.05 |
| ETTm1 | +0.23 | Electricity | 0.02 |
| ETTh2 | +0.11 | Exchange | −0.19 |
| ETTh1 | 0.09 | Traffic | −0.19 |

只有 ETTm2 强符合"长 horizon 要长历史"的直觉（且次线性）；ETTm1/ETTh2 在弱正区；Weather/Electricity 近乎平坦，稳定在约 $10^3$ 步；Exchange/Traffic 为负——长 horizon 反而偏好短 context，与远端历史主动误导模型的非平稳性一致[^src-2606-27282-ltsf-linear-models]。

**推论**：社区默认的 $L = 96$ 在两个相反方向上同时犯错——对 Weather/Electricity 少给约一个数量级的历史，对 Exchange/Traffic 的长 horizon 则过量，最优值跌破 96[^src-2606-27282-ltsf-linear-models]。

## 只变回看长度的收益分层

作者用一套固定预处理（universal default）只扫 $L$、在最长 horizon 上测误差，把数据集分成三档：

| Regime | 数据集 | $L: 96 \to 2000$ |
|---|---|---|
| strong | ETTh1 / ETTh2 / Weather | 总降幅 36%–53% |
| plateau | ETTm2 | 约 16%，大部分在中等长度达成 |
| flat | ETTm1 | 几乎无改善 |

两条配套结论：固定短 $L$ 在 strong regime 上"在任何其他预处理被调之前就已放弃大部分可达精度"；把 $L$ 钉死后再跑完整预处理搜索也找不回这部分收益，说明增益源自回看长度本身。数据集属于哪一档无法从元信息预判，只能实测[^src-2606-27282-ltsf-linear-models]。

## 长程线性信号的可检出性

对 lag 720、经 harmonic 去季节化的序列，六个标准 benchmark 仍保留 0.39–0.73 的跨通道平均 Pearson 相关，bootstrap 置信区间远离 0[^src-2606-27282-ltsf-linear-models]。长 context 因此捕获的是真实的远端线性信息，而不只是重复表征一个周期——后者用 trend-seasonal 分解或频域算子即可廉价复现。ETTm1/ETTm2 去季节化后的相关反而高于原始相关，说明强日周期掩盖了更慢的结构[^src-2606-27282-ltsf-linear-models]。

权重可视化给出微观对应：Exchange 的权重几乎全压在近期 lag，Traffic 在 168、336 等周周期位置成带，Weather 与 ETTm 的最大权重常在数百步之前。长 $L^*$ 的价值主要是"取到少数相位对齐的观测"，而非把全部历史都用上[^src-2606-27282-ltsf-linear-models]。

## per-series 分歧

Weather 是唯一 per-series 指数紧密聚在中位数附近的数据集；ETTh2、Electricity、Traffic 的 per-series 指数跨越 0——同一数据集里，部分通道偏好随 horizon 增长 context，部分偏好缩短[^src-2606-27282-ltsf-linear-models]。

## 与其他路线的关系

- **实证细化（瓶颈接力 $\Delta$）**：[[lstf]] 既有论断认为「Transformer 随回看窗口增长性能不提升」，PatchTST 是例外。本页在线性模型内部进一步细化：最优 $L$ 也不是常数，而是因序列而异且对预测步长非单调[^src-2606-27282-ltsf-linear-models]。
- **同槽位互斥（长程上下文获取）**：[[source-cometnet|CometNet]] 保持固定短回看窗，改为训练前离线抽取跨尺度母题原型、在线用内容门控检索补足长程上下文；它换了「记忆」载体，本页调节回看长度本身[^src-cometnet]。
- **同槽位互斥（长程上下文获取）**：[[source-pih|PIH]] 将「继续拉长回看不再改善 MSE 的临界长度」定义为最大有效窗口（MEW），用变分信息瓶颈门控与 Mamba 分块滤波统一外推该上限；PIH 按数据集求单一 MEW，而本页逐序列、逐步长搜索 $L^*$ 且允许负斜率[^src-pih]。

## 关联页面

- [[searchcast]] — 本源派生：产出该规律的线性预测管线
- [[source-2606-27282-ltsf-linear-models]] — 本源摘要
- [[trailing-fraction-local-normalization]] — 跨槽位正交：同一预处理搜索中的归一化轴
- [[grouped-hyperparameter-search]] — 跨槽位正交：逐序列搜索的分组组织方式

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
[^src-cometnet]: [[source-cometnet]]
[^src-pih]: [[source-pih]]
