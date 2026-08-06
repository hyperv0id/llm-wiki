---
title: "PatchTST"
type: entity
tags:
  - time-series
  - forecasting
  - transformer
  - patch
  - channel-independence
  - self-supervised
  - ICLR-2023
created: 2026-05-30
last_updated: 2026-08-06
source_count: 7
confidence: medium
status: active
---

# PatchTST

**PatchTST** (Patch Time Series Transformer) 由 Nie, Nguyen, Sinthong & Kalagnanam (Princeton/IBM) 提出，发表于 ICLR 2023。它是首个将 **patching** 和 **channel independence** 同时引入时间序列 Transformer 的模型，在长期预测和自监督表示学习上均取得 SOTA，证明 Transformer 在正确设计下可以超越简单线性模型 [^src-patchtst]。

## 核心设计

### Patching

将时间序列分割为固定长度的子序列级 patch 作为 Transformer 输入 token，而非逐点输入 [^src-patchtst]：

- Patch 长度 $P=16$，步长 $S=8$（可重叠）
- 输入 token 数从 $L$ 降至 $L/S$，注意力复杂度按 $S^2$ 下降
- Patch 内信息自动聚合，保留局部语义
- 两个变体：**PatchTST/42**（L=336, 42 patches）和 **PatchTST/64**（L=512, 64 patches）

### Channel Independence

多元时间序列各通道独立处理，共享 Transformer 权重 [^src-patchtst]：

- M 个通道 → M 个独立样本，显著增加训练数据量
- 各通道独立学习分布，注意力聚焦时间维度
- 消融实验证明 CI 是性能提升的关键因素之一

### Instance Normalization

Patching 前对每个单变量序列做零均值单位方差归一化，预测后恢复统计量，缓解分布漂移 [^src-patchtst]。

## 架构

```
Input: x^(i) ∈ ℝ^(1×L), i=1...M
  → Instance Norm → Patching (P=16, S=8) → N patches
  → Linear Projection Wp ∈ ℝ^(D×P) + Positional Encoding Wpos ∈ ℝ^(D×N)
  → Vanilla Transformer Encoder (BatchNorm + FFN + Residual)
  → Flatten + Linear Head
Output: x̂^(i) ∈ ℝ^(1×T)
```

## 性能

| 对比基线 | MSE 降幅 | MAE 降幅 |
|----------|---------|---------|
| vs 最佳 Transformer 基线 (PatchTST/64) | 21.0% | 16.7% |
| vs 最佳 Transformer 基线 (PatchTST/42) | 20.2% | 16.4% |
| vs DLinear | 大和中等数据集优势明显 | — |

训练速度 [^src-patchtst]：
- Traffic：22× 加速（464s vs 10040s）
- Electricity：19× 加速
- Weather：4× 加速

## 自监督表示学习

Patch-level masked autoencoder（借鉴 CV [[mae|MAE]] 的范式）：非重叠 patch，40% 掩码率，训练重建被掩码 patch [^src-patchtst]。

- Fine-tuning 在大数据集上超越 supervised training
- Transfer learning（Electricity 预训练→迁移）仍优于 DLinear 和其他 Transformer
- vs TS2Vec/BTSF/TNC/TS-TCC：linear probing 即有 34.5%–48.8% MSE 提升

## 消融实验关键发现

1. **Patching + CI 缺一不可**：移除任一组件性能显著下降 [^src-patchtst]
2. **PatchTST 是唯一随 L 增大持续降低 MSE 的模型**：FEDformer/Autoformer/Informer 在 L 增大时性能不变或变差 [^src-patchtst]
3. **BatchNorm 优于 LayerNorm**：在时序 Transformer 中已验证 [^src-patchtst]

## 历史地位与影响

PatchTST 是 LSTF 领域的关键转折点 [^src-patchtst]：

- **回击"Transformer 无用论"**：在 DLinear (Zeng et al., 2022) 质疑 Transformer 有效性后，PatchTST 证明正确设计下的 Transformer 可超越线性模型
- **Patching 成为标配**：后续模型 [[simdiff|SimDiff]], [[cvpe|CVPE]], [[sparsetsf|SparseTSF]], [[srsnet|SRSNet]] 等均采用 patch tokenization；[[selective-representation-space|SRS]] 进一步将固定 adjacent patching 升级为自适应选择与重排 [^src-srsnet]
- **CI 成为默认策略**：大多数后续 Transformer 采独立处理各通道，跨变量交互仅作可选增强
- **自监督 + Transfer 潜力**：为时序基础模型（[[timesfm|TimesFM]], [[chronos|Chronos]]）提供预训练范式参考

## 局限性

- Channel Independence 完全忽略跨变量依赖 [^src-patchtst]
- 小数据集（ETT 系列）优势不明显
- CI 的跨变量建模能力有限——[[cvpe|CVPE]] 和 [[crossformer|Crossformer]] 尝试补充此缺陷
- 实例级失效现象：PIR 论文（Post-forecasting Identification and Revision，Liu et al., NeurIPS 2025）报告在 ETTh1 上 PatchTST 的逐实例 MSE 呈长尾分布——多数实例误差低，但误差曲线存在尖峰，即平均性能良好时仍有预测失效的个别实例（Fig 1）[^src-pir]
- [[probts|ProbTS]] 显示：作为长程 **NAR 点预测** 代表，PatchTST 在长程趋势/季节场景强势，但短程高 [[non-gaussianity|非高斯性]] 上优势收缩；强季节 Traffic 上可被 AR 概率模型（TimeGrad）超过[^src-probts]
- [[zeus|Zeus]]（ICML 2026）批评 patch tokenization 纠缠细粒度变化、损害逐点级任务（论文自述），并给出实证：MOMENT 从 patch-missing 换到 point-missing 插补 MSE 平均恶化 −22.4%（ETTm1 −21.7%、ETTh2 −24.8%、Weather −16.9%）[^src-2607-01918]

## Connections

- **前驱/基线**：[[informer|Informer]], [[autoformer|Autoformer]], [[fedformer|FEDformer]], [[ltsf-linear|DLinear / LTSF-Linear]]
- **同期/对照**：[[tide|TiDE]] — residual MLP + 协变量高速路，线性复杂度；在 Traffic 等大数据集上可超过 PatchTST，且训练/推理显著更快[^src-tide]
- **后续/继承**：[[simdiff|SimDiff]]（patch + CI + diffusion）, [[cvpe|CVPE]]（CI + CD 折中）, [[sparsetsf|SparseTSF]]（patch + 极致压缩）, [[srsnet|SRSNet]] / [[selective-representation-space|SRS]]（自适应 patch 表示空间插件）
- **多模态改造**：[[tess|TESS]]（arXiv:2603.12664v2）直接以 PatchTST 为 backbone，输入前拼接 4 个离散时序原语（mean shift/volatility/shape/lag）类别 embedding 并附加门控 BCE 监督，四数据集全面超过原版 PatchTST（Bitcoin MSE 2.2726 vs 3.2456）；[[timi|TiMi]] 则用 MMoE 替换 FFN——两者同为 PatchTST 的多模态改造样板[^src-tess]
- **后处理插件**：PIR（Liu et al., NeurIPS 2025）以 PatchTST 为 channel-independent 骨干测试后处理修订，论文报告 48 个实验设置平均 MSE 降低 8.99%（Table 1）[^src-pir]
- **与 [[zeus|Zeus]] 的对照**：点预测上 Zeus（ICML 2026）零样本在多数数据集上超过 full-shot PatchTST（Zeus 表 1）；异常检测（UCR Anomaly Archive，adjusted F1）PatchTST 0.877 vs Zeus 0.900（实证结果，Zeus 表 9）[^src-2607-01918]
- **核心概念**：[[patch-based-tokenization]], [[channel-independence]], [[instance-normalization]], [[lstf]]
- **自监督/预训练**：[[mae|MAE]], [[videomae]], [[timesfm|TimesFM]], [[chronos]]

[^src-patchtst]: [[source-patchtst]]
[^src-srsnet]: [[source-srsnet]]
[^src-probts]: [[source-probts]]
[^src-tide]: [[source-tide]]
[^src-tess]: [[source-tess]]
[^src-pir]: [[source-pir]]
[^src-2607-01918]: [[source-2607-01918]]
