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
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning

零样本时序预测模型长期在两种能力之间顾此失彼：Transformer 类架构擅长从历史窗口提取上下文模式，却缺少显式状态追踪，长程预测容易退化；循环模型天然适合追踪序列状态，上下文学习能力又偏弱。主流模型做超长预测时还要把上一步的点估计回喂给模型再滚动一步，每回喂一次就把预测分布压缩成单点，不确定性无法传递到远处。TiRex 的核心巧思是一次化解这两重矛盾：选用具备显式标量状态追踪的纯 sLSTM 担任因果解码器，并在预训练时随机把连续的未来分块遮蔽成缺失输入（Contiguous Patch Masking）；推理时只需把超长预测窗置为同样的缺失占位，即可在不回喂点估计的前提下连续推进隐藏状态并同步读出多步分位数分布[^src-tirex]。论文报告 TiRex 在 GiftEval-ZS 与 Chronos-ZS 两个零样本基准的概率指标与综合排名均居首[^src-tirex]。

## 核心做法

- **分块归一化与缺失掩码输入**：每条序列先做实例级 Z-score 归一化消除量纲差异，再切成互不重叠的定长分块，与一个二值缺失掩码拼接后由共享的两层残差 MLP 映射到隐藏维；非重叠分块同时把主干需处理的序列长度按块长倍数压缩，掩码通道则让模型区分真实观测与占位缺失，是占位式解码的基础[^src-tirex]。
- **纯 sLSTM 因果主干与分位数读出**：主干堆叠纯 sLSTM 解码块（含 RMSNorm、前馈网络与残差连接），选 sLSTM 而非 xLSTM 家族中的 mLSTM，是因为只有 sLSTM 的显式循环状态支持状态追踪；输出端用对称的两层残差 MLP 把每个分块的隐藏表示映射为块内每步的多个分位数预测，训练仅在非缺失位置优化分位数回归损失[^src-tirex]。
- **连续块掩码（Contiguous Patch Masking）**：预训练时按随机采样的连续块数与掩码比例，把输入中随机一段连续分块整体遮蔽为缺失，被遮蔽段在输入中的形态与推理期的缺失占位完全一致；损失对每个输出分块都计算，模型由此练习跨过任意长度的缺失段继续预测。推理时把整个预测窗当缺失占位拼接在历史之后，一次前向读出整段多分位数预测，免去自回归回喂。预训练另配幅值调制、截断与周期尖峰注入三项数据增强[^src-tirex]。

## 证据

1. **定位与主结论**：论文报告 TiRex 在两个基准的概率指标（CRPS/WQL）与平均排名均居首，长程上首次有零样本模型超过监督训练的 [[patchtst|PatchTST]] 与 [[tft|TFT]]，推理较 Chronos-Bolt-Base 与 TimesFM-2.0 分别快逾 4 倍与 11 倍[^src-tirex]。
2. **真实涨点归因**：消融显示涨点来自占位式连续解码与纯 sLSTM 主干：占位退化为自回归损害长程，末尾朴素掩码拖累短程；主干换纯 mLSTM 或 Transformer 长程落后，同数据同增强重训的 Chronos-Bolt 亦落后[^src-tirex]。
3. **证据缺口与负结果**：仅支持单变量输入；主评测剔除了与训练数据重叠的部分设置；超参数未经精细调优[^src-tirex]。

## 相关页面

- [[tirex]] — 机制拆解页：将本页算子配方展开为分阶段维度变换与跨模型对照矩阵
- [[timesfm]] / [[chronos]] — 同槽位互斥：Transformer 自回归零样本基础模型，以注意力与逐块回喂点估计替换 sLSTM 与缺失占位
- [[ar-vs-nar-decoding]] — 特例关系：CPM 占位延续是自回归/非自回归解码谱系的第三条路线实例
- [[in-context-learning]] — 跨槽位正交：单序列上下文条件化概念与本页解码机制分属不同设计轴

---

**来源**：Andreas Auer、Patrick Podest、Daniel Klotz、Sebastian Böck、Günter Klambauer、Sepp Hochreiter（NXAI / JKU Linz / IT:U Austria），NeurIPS 2025。

[^src-tirex]: [[source-tirex]]
