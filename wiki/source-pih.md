---
title: "PIH: Enhancing the Maximum Effective Window for Long-Term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - long-term-forecasting
  - context-length
  - information-bottleneck
  - mamba
  - transformer
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# PIH: Enhancing the Maximum Effective Window for Long-Term Time Series Forecasting

- **作者/机构**：Jiahui Zhang、Zhengyang Zhou、Wenjie Du、Yang Wang（中国科学技术大学 / 中科大苏州高等研究院；代码 `forever-ly/PIH`）[^src-pih]
- **出处**：核心切片未见明确会议标识（仅 checklist 为 AAAI 模板；本仓库 OpenReview CCF-A 调研记录其录用于 NeurIPS 2025 Poster，未在切片正文核实）[^src-pih]
- **分级**：Tier 2（轻量化入库，未为 `pih`、`maximum-effective-window`、`IBF`、`HTM` 新建 `source_count: 1` 独立页）

直觉上回看窗口越长，模型掌握的历史信息越充分，但现有时序 Transformer 拉长回看窗口后预测误差往往不降反升，因为多出来的历史夹杂冗余与噪声，不仅摊薄注意力权重，也诱发过拟合。论文将「继续加长回看窗口不再改善 MSE 的临界长度」定义为最大有效窗口（MEW），把变分信息瓶颈门控加噪滤波（IBF）与「Mamba 长程滤波加分块短序列 Transformer」（HTM）串联挂载到 [[patchtst|PatchTST]] 构成 PIH，试图在压降显存与算力的同时把有效回看窗口向外推长[^src-pih]。

## 去黑话算子配方与图谱关系

PIH 将与骨干解耦的变分滤波模块（IBF）与混合序列模块（HTM）串入 patch 化流程[^src-pih]：

- **变分信息瓶颈滤波（IBF）**：在 Patch Embedding 之后，先经 `MLP + Sigmoid` 评估各 patch 的重要度，再由 `Gumbel-Sigmoid` 采样伯努利掩码 $\lambda_i$，按掩码注入同均值方差的高斯噪声 $z^{\text{noise}}=\lambda z + (1-\lambda)\epsilon$ 抑制冗余历史，训练目标在 MSE 上叠加变分压缩上界惩罚项联合优化，从而避免对长历史子序列做组合枚举[^src-pih]。
- **混合架构切分（HTM）**：前向流程为 `Patch Embedding → Mamba → Dropout → IBF → 切块 → Transformer`；切块支持按步长模 $K$ 间隔抽稀或连续分段，将自注意力输入长度从 $L/P$ 压至 $L/(PK)$，把长程依赖交由 [[mamba|Mamba]] 线性滤波、分段局部交互交由 Transformer 承担，在 $L=1024$ 下相对 PatchTST 将显存与耗时降为 $1/2\text{–}1/3$[^src-pih]。
- **图谱关系**：对 [[patchtst|PatchTST]] 属**瓶颈接力 $\Delta$**（$L=1024$ 下降低算力并小幅推长 MEW，但短窗 $L=96$ 反而落后）；对 [[information-bottleneck-principle|信息瓶颈原理]] 属**同构实现**（以序列级 `Gumbel-Sigmoid` 噪声掩码落地上界压缩）；对 [[mamba|Mamba]] 属**跨槽位正交混合**（Mamba 担纲长程滤波、分块 Transformer 处理短上下文）；与 [[lookback-horizon-scaling|Lookback–Horizon Scaling]] 及 [[source-2606-27282-ltsf-linear-models|SearchCast]] 形成**视角对照与张力**（SearchCast 按单序列、单步长搜索最优 $L^*$ 且幂律斜率可负，PIH 则依赖离散粗网格以架构滤波统一推长 MEW）[^src-pih]。

## 证据

1. **定位与主结论**：$L=1024$ 下整体优于 PatchTST、Bi-Mamba 与 DLinear 且显存耗时降为 PatchTST 的 $1/2\text{–}1/3$，但短窗 $L=96$ 落后 PatchTST[^src-pih]。
2. **消融真相与 MEW 边界**：正文唯一消融仅给柱/折线图无数值表，无法量化拆出单个组件的精确掉点量级；对 PatchTST 的 MEW 扩张较温和（ETTh1/ECL $512\to 1024$、ETTm1 $1024\to 1440$，Traffic/ETTh2 持平或触顶、Weather 拉长后 MSE 微升）[^src-pih]。
3. **跨源张力与缺口**：与 SearchCast「最优 $L^*$ 可呈负斜率」张力对照；缺 Exchange、统计显著性与 LLM/TSFM 基线，MEW 依赖离散粗网格[^src-pih]。

## 相关页面

- [[patchtst]] — 宿主对照与瓶颈接力 $\Delta$（$L=1024$ 降耗并温和扩窗，短窗 $L=96$ 落后）
- [[lookback-horizon-scaling]] — 视角张力对照（统一架构推长 MEW vs 逐序列步长异质幂律 $L^*$）
- [[source-2606-27282-ltsf-linear-models]] — SearchCast 口径对照（离散粗网格滤波 vs 闭式 Ridge 搜索）
- [[information-bottleneck-principle]] — 同构实现（序列版 `Gumbel-Sigmoid` 噪声掩码变分上界）
- [[mamba]] — 跨槽位正交混合（Mamba 长程滤波 + 分块短序列 Transformer）

[^src-pih]: [[source-pih]]
