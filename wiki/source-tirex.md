---
title: "TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning"
type: source-summary
tags:
  - time-series
  - foundation-model
  - zero-shot
  - xlstm
  - slstm
  - probabilistic-forecasting
  - neurips-2025
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning

**来源**：Andreas Auer、Patrick Podest、Daniel Klotz、Sebastian Böck、Günter Klambauer、Sepp Hochreiter（NXAI / JKU Linz / IT:U Austria），NeurIPS 2025；提出 35M 参数零样本概率时序预测模型 **TiRex** 与 **Contiguous Patch Masking（CPM）** 机制[^src-tirex]。

现有零样本时序预测模型常在上下文模式学习与长程状态追踪之间顾此失彼，而逐窗自回归回喂点估计又会切断不确定性向远期的传递。为此，TiRex 选用具备显式标量状态追踪的纯 `sLSTM` 担任因果解码器，并在预训练阶段随机遮蔽连续未来分块为缺失输入；推理时只需将超长预测窗置为缺失占位，即可在不回喂点预测的条件下连续推进隐藏状态并同步输出多步分位数分布[^src-tirex]。

## 算子配方与基线差分

剥离论文术语后，TiRex 由标准算子构成：单序列先经 Z-score 归一化并按长度 $m_{\text{in}}=32$ 非重叠分块，与二值缺失掩码拼接后通过共享两层残差 MLP 映射至隐藏维；主干堆叠带有 RMSNorm、残差跳跃连接与前馈网络的纯 `sLSTM` 解码块（上下文上限 $2048$，共 35M 参数）；输出端经两层残差 MLP 预测每个 $m_{\text{out}}=32$ 步块内的 $9$ 个分位数 $\{0.1,\dots,0.9\}$，仅在非缺失位置优化分位数回归损失[^src-tirex]。

关键训练与增强部件是**连续块掩码（CPM）**：预训练时按均匀采样的连续块数与掩码比例遮蔽输入分块，配合幅值调制（概率 $0.5$）、截断（$0.5$）与周期尖峰（$0.05$）增强上下文鲁棒性[^src-tirex]。与 [[timesfm|TimesFM]]、[[chronos|Chronos-Bolt]] 等自回归 Transformer 相比，TiRex 以纯 `sLSTM` 加缺失占位替代注意力与点估计回喂，参数量从 200M–500M 降至 35M；与混合 xLSTM 相比则彻底删去 `mLSTM`[^src-tirex]。

## 证据

1. **定位与主结论**：35M TiRex 在 GiftEval-ZS（81 设置）与 Chronos-ZS（27 集）的概率指标（CRPS/WQL）与排名均居首，长程首次零样本超越监督 [[patchtst|PatchTST]] 与 [[tft|TFT]]，推理速度较 Chronos-Bolt-Base 快 4 倍以上、较 TimesFM-2.0 快 11 倍以上[^src-tirex]。
2. **真实涨点归因**：去掉多块占位退化为自回归、改用末尾朴素掩码、或将主干换为纯 `mLSTM` / Transformer 均导致长程或短程明显退化，同数据同增强重训的 Chronos-Bolt 长程亦落后；去幅值调制会拉低 Chronos-ZS，但在 GiftEval-ZS 上略微反升[^src-tirex]。
3. **证据缺口与负结果**：仅支持单变量输入；GiftEval-ZS 主评测剔除了 16/97 个重叠设置；超参未经精细调优，且解码深度与隐藏宽度仅见于论文附录[^src-tirex]。

## 相关页面

- [[tirex]] — TiRex 机制拆解、分阶段数据流与跨模型对照页
- [[timesfm]] / [[chronos]] — Transformer 自回归零样本基础模型对照
- [[ar-vs-nar-decoding]] — 自回归、非自回归与连续占位解码对照
- [[in-context-learning]] — 时序上下文条件化与提示式少样本的边界

[^src-tirex]: [[source-tirex]]
