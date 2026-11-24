---
title: "SparseTSF"
type: entity
tags:
  - time-series
  - forecasting
  - model
  - lightweight
  - tpami-2026
  - icml-2024
created: 2026-04-28
last_updated: 2026-11-24
source_count: 5
confidence: medium
status: active
---

# SparseTSF

SparseTSF 是 IEEE TPAMI 2026 接收的极轻量级时序预测模型，由华南理工大学林伟伟团队提出（ICML 2024 Oral）。其核心创新是 **Cross-Period Sparse Forecasting** 技术，通过跨周期稀疏预测将模型参数量降至 **1,000 以下**[^src-sparsetsf]。

## 核心创新

### Cross-Period Sparse Forecasting

传统 LTSF 模型需要 O(L×H) 的参数量来建模长时间依赖。SparseTSF 将这一任务分解为：

1. **下采样**：将长度为 L 的序列按周期 w 重排为 w × (L/w) 的矩阵
2. **共享预测**：用单一 Linear/MLP backbone 预测所有子序列
3. **上采样**：将预测结果重排回长度为 H 的序列

参数量从 O(L×H) 降至 O(L/w × H/w)，实现 1~4 个数量级的压缩。

### 隐式正则化

论文从理论上证明：Sparse 技术等价于在权重矩阵上施加结构化约束——惩罚非周期位置的连接（稀疏），强制周期内参数一致（共享）。这起到类似 L1 正则化的作用，增强模型对噪声的鲁棒性。

## 架构变体

| 变体 | 参数量 | 适用场景 |
|------|--------|----------|
| SparseTSF/Linear | < 1,000 | 单变量预测、计算资源极度受限 |
| SparseTSF/MLP | < 8,000 | 高维多变量预测 |

## 性能

- 在 ETTh1, ETTh2, Electricity, Traffic, Solar-Energy, Weather 上取得 SOTA 或接近 SOTA
- 相比 FITS（10k 参数），SparseTSF 参数量再降 10 倍
- 长 look-back 窗口（720）下优势更明显

## 与其他轻量模型的关系

| 模型 | 参数量 | 技术路线 |
|------|--------|----------|
| DLinear | ~1k | 单层线性 |
| FITS | ~10k | 频域 + 低通滤波 |
| **SparseTSF** | **<1k** | **跨周期稀疏预测** |
| CycleNet | ~10k | 可学习周期参数 |
| [[source-timebase\|TimeBase]] | 0.39k（ECL 720→720） | 段级低秩正交基 |

- **同槽位互斥（[[source-timebase|TimeBase]], ICML 2025）**：两者均按主周期切分历史以避开 $O(LH)$ 点级映射，但 SparseTSF 将序列下采样为周期内子序列并共享单一线性/MLP 头，TimeBase 则改为通过正交正则线性层提取 $R=6$ 个可学习时间基再线性重构未来分段（需周期 $P$ 先验，正文缺组件级独立消融表）[^src-timebase]。
- **瓶颈接力 $\Delta$ 与复现对齐（[[source-dsr-tsf|DSR-TSF]], NeurIPS 2026）**：DSR-TSF 保留 SparseTSF「周期分段 + 共享头」骨架，针对同一变量在不同窗口主导周期漂移导致固定周期错配的问题，前置 `rFFT` 谱条件软路由与 $R$ 条时域响应基残差修正，并对共享头施加低秩约束（ETTh1/Linear 参数 $0.45\text{K}\to 0.90\text{K}$）；消融表明低秩头约束的退化贡献（ETTh1 $0.393\to 0.404$）略重于改静态路由（$0.393\to 0.402$），且存在复现口径矛盾——DSR-TSF 正文称 Traffic 仍有提升，而 Table 1 实测其 MLP 版（$0.403$）劣于 SparseTSF/MLP（$0.396$）、Linear 版（$0.412$）仅持平且 MAE 落后（$0.274$ vs $0.264$）[^src-dsr-tsf]。
- **瓶颈接力 $\Delta$ 与复现对齐（[[source-mixlinear|MixLinear]], ICLR 2026）**：MixLinear 沿用按步长下采样骨架，将单共享线性头拓展为「时域 2D 分段双线性分解 + 频域秩-2 复投影」双支相加（全模型仅 45–176 参数、0.25–2.05 ms 延迟），在 Exchange 上相对 SparseTSF 显著降错且提速约 3.2 倍；但消融显示其低维集主要靠时域双线性、高维集才靠频域秩-2 投影，且主表只报 MSE 不报 MAE[^src-mixlinear]。

## 在 PIR 后处理中的表现

[[pir|PIR]]（Post-forecasting Identification and Revision，Liu et al., NeurIPS 2025）将 SparseTSF 作为 channel-independent 骨干测试后处理修订：论文报告在 48 个实验设置上平均 MSE 降低 25.87%，为四个骨干（PatchTST 8.99%、iTransformer 3.47%、TimeMixer 2.34%）中最大 [^src-pir]。定性分析中，论文报告 SparseTSF 在 ETTh1 上的最大单实例预测误差从 2.85 降至 0.81（加 PIR 后），作为其存在实例级失效的直接证据 [^src-pir]。

## 引用

[^src-sparsetsf]: [[source-sparsetsf]]
[^src-pir]: [[source-pir]]
[^src-timebase]: [[source-timebase]]
[^src-dsr-tsf]: [[source-dsr-tsf]]
[^src-mixlinear]: [[source-mixlinear]]