---
title: "BigST"
type: entity
tags:
  - traffic-forecasting
  - spatio-temporal
  - graph-neural-network
  - linear-attention
  - scalability
  - large-scale
created: 2026-06-09
last_updated: 2026-07-19
source_count: 2
confidence: medium
status: active
---

# BigST

**BigST** 是一个**线性复杂度**时空图神经网络（STGNN），用于**大规模路网**上的交通预测，可扩展到约 **10 万节点**（北京滴滴路网 99,716 段），比常用交通数据集大两个数量级[^src-bigst]。出自 Han et al.，[[source-bigst|PVLDB 2024]]，开源于 usail-hkust/BigST[^src-bigst]。

## 动机
现有 STGNN 有两大瓶颈：长程时间依赖建模代价随序列长度爆炸；[[gwnet|GWNET]] 式自适应邻接 A=σ(E1E2ᵀ) 的图结构学习需 O(N²)，无法扩展到大路网[^src-bigst]。BigST 的核心思想是**解耦**长序列建模与预测，把前者作为可缓存的预处理步骤[^src-bigst]。

## 架构（两阶段）

### 阶段一：长序列特征提取器 [[long-sequence-feature-extractor|LSFE]]（预处理）
- **上下文感知线性化 Transformer**：先用膨胀时间卷积补点级语义，再用 Performer 正随机特征 (PRF) 近似 softmax 核注意力，把时间自注意力从 O(T_l²) 降到 **O(T_l)**；以生成式预训练（由长历史预测未来）学习长程动态[^src-bigst]。
- **周期特征采样**：免训练，取过去 D 天、W 周同期流量特征，显式建模日/周周期[^src-bigst]。
- 输出 H_long、H_per 可**整库预计算缓存**，把重计算移出预测阶段[^src-bigst]。

### 阶段二：线性化全局空间卷积网络 (LGSCN)（预测）
- **Patch 级动态图学习 (PDGL)**：仅取最近 T（如 12）步聚成 patch；每节点维护静态嵌入（固有属性）+动态嵌入（短期演化，拼时刻/星期嵌入），点积得分经温度 τ 的 softmax 构造**时变邻接**[^src-bigst]。
- **[[linearized-spatial-convolution|线性化空间卷积 (LSC)]]**：用 PRF 核分解 A≈D⁻¹φ(E1)φ(E2)ᵀ + 乘法结合律重排，**免显式构造 O(N²) 稠密邻接**，把 mix-hop 消息传递降到 **O(N)**；并加距离先验空间正则 L_r[^src-bigst]。
- 末端：concat(H ‖ H_long ‖ H_per) → MLP **非自回归**一次出 T_f 步[^src-bigst]。

整体时间/空间复杂度对序列长度 T_l 与节点数 N 均为**线性**[^src-bigst]。

## 结果与效率
- 精度：在 California（9,638）、Beijing（99,716）上全面超越 [[dcrnn|DCRNN]]、ASTGCN、[[gwnet|GWNET]]、AGCRN、STGODE、DSTAGNN；California 平均提升 6.3/7.6/8.4%（MAE/RMSE/MAPE）[^src-bigst]。
- 效率：较 GWNET 训练加速 2.3–20.6×、推理 1.7–26.5×、省显存达 76.1%；推理延迟随节点数增加保持稳定（线性复杂度 + GPU 并行）[^src-bigst]。
- 消融：去长程表征/周期特征/空间卷积近似/动静态嵌入均掉点；**静态嵌入比动态嵌入更关键**（短期信息噪声大）[^src-bigst]。

## 关联
- [[gwnet]] — 直接前身：BigST 线性化了 GWNET 的自适应邻接 A=σ(E1E2ᵀ)
- [[ragc]] — 同属大规模高效交通预测（正则化自适应图卷积）
- [[patchstg]] — 同期大规模交通 Transformer（不规则空间分块）
- [[large-scale-spatial-temporal-graph]] — BigST 是扩到 ~10 万节点的代表性工作
- [[linearized-spatial-convolution]]、[[long-sequence-feature-extractor]] — 两大核心机制
- [[traffic-forecasting]] — 任务
- [[mage|MAGE]] (NeurIPS 2025) — 同为线性复杂度自适应图学习，但用 kernel 近似 + MoE 替代 PRF，实验全面超越 BigST[^src-mage]

[^src-bigst]: [[source-bigst]]
[^src-mage]: [[source-mage]]
