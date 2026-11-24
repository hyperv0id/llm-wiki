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
last_updated: 2026-11-24
source_count: 14
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
- 实例级失效现象：[[source-pir|PIR 论文]]（Post-forecasting Identification and Revision，Liu et al., NeurIPS 2025）报告在 ETTh1 上 PatchTST 的逐实例 MSE 呈长尾分布——多数实例误差低，但误差曲线存在尖峰，即平均性能良好时仍有预测失效的个别实例（Fig 1）[^src-pir]
- [[probts|ProbTS]] 显示：作为长程 **NAR 点预测** 代表，PatchTST 在长程趋势/季节场景强势，但短程高 [[non-gaussianity|非高斯性]] 上优势收缩；强季节 Traffic 上可被 AR 概率模型（TimeGrad）超过[^src-probts]
- [[zeus|Zeus]]（ICML 2026）批评 patch tokenization 纠缠细粒度变化、损害逐点级任务（论文自述），并给出实证：MOMENT 从 patch-missing 换到 point-missing 插补 MSE 平均恶化 −22.4%（ETTm1 −21.7%、ETTh2 −24.8%、Weather −16.9%）[^src-2607-01918]

## Connections

- **前驱/基线**：[[informer|Informer]], [[autoformer|Autoformer]], [[fedformer|FEDformer]], [[ltsf-linear|DLinear / LTSF-Linear]]
- **同期/对照**：[[tide|TiDE]] — residual MLP + 协变量高速路，线性复杂度；在 Traffic 等大数据集上可超过 PatchTST，且训练/推理显著更快[^src-tide]
- **后续/继承**：[[simdiff|SimDiff]]（patch + CI + diffusion）, [[cvpe|CVPE]]（CI + CD 折中）, [[sparsetsf|SparseTSF]]（patch + 极致压缩）, [[srsnet|SRSNet]] / [[selective-representation-space|SRS]]（自适应 patch 表示空间插件）
- **RNN 迁移**：[[witran|WITRAN]]（NeurIPS 2023）引用 PatchTST（与 TimesNet 并列）作为"把信息传递切成 patch 分开处理可同时保留全局与局部语义"的证据，据此把 RNN 输入按周期拆成子序列分块递推[^src-witran-jia-2023]
- **多模态改造**：[[tess|TESS]]（arXiv:2603.12664v2）直接以 PatchTST 为 backbone，输入前拼接 4 个离散时序原语（mean shift/volatility/shape/lag）类别 embedding 并附加门控 BCE 监督，四数据集全面超过原版 PatchTST（Bitcoin MSE 2.2726 vs 3.2456）；[[timi|TiMi]] 则用 MMoE 替换 FFN——两者同为 PatchTST 的多模态改造样板[^src-tess]
- **后处理插件**：[[pir|PIR]]（Liu et al., NeurIPS 2025）以 PatchTST 为 channel-independent 骨干测试后处理修订，论文报告 48 个实验设置平均 MSE 降低 8.99%（Table 1）[^src-pir]
- **与 [[zeus|Zeus]] 的对照**：点预测上 Zeus（ICML 2026）零样本在多数数据集上超过 full-shot PatchTST（Zeus 表 1）；异常检测（UCR Anomaly Archive，adjusted F1）PatchTST 0.877 vs Zeus 0.900（实证结果，Zeus 表 9）[^src-2607-01918]
- **与 [[tirex|TiRex]] 的对照**：在 GiftEval-ZS（81 项设置）长程概率预测上，35M 纯 sLSTM 模型 TiRex（NeurIPS 2025）首次零样本超越监督训练的 PatchTST（实证对照）[^src-tirex]
- **多变量因果扩展**：[[timer-xl|Timer-XL]]（ICLR 2025）——**瓶颈接力 $\Delta$**，针对 PatchTST 的 Encoder-only 固定输出头与通道独立局限，改为因果 Decoder-only 并按 $M = C \otimes T$ 展平跨变量与协变量时间块，支持变长滚动并在零样本与跨站预测上超越 PatchTST[^src-timer-xl]
- **跨槽位正交（视觉-数值分解框架中的数值主干消融）**：[[source-dmmv|$\mathrm{D_{MMV}}$]]（NeurIPS 2025）在其「MAE 周期分支 + 线性趋势残差门控」消融中，将线性数值分支替换为 PatchTST，整体精度几乎不变（仅 ETTm1 略微改善），表明该双视图框架的增益源自回测残差分解与归一化层微调分工，而非数值主干本身的复杂度[^src-dmmv]
- **窗外母题外推（瓶颈接力 $\Delta$）**：[[source-cometnet|CometNet]]（AAAI 2026）针对 PatchTST 在固定短回看窗（$L=96$）内仅做局部 patch 编码、无法捕捉跨数千步重现规律的视野瓶颈，在预测前从全训练集离线以 `MovingAvg + FFT + DTW` 聚类抽取 $K$ 个跨尺度母题原型，在线以 Dense MoE 内容门控匹配母题并注入生命周期相对相位信号；在 8 个标准 LTSF 基准（$L=96$）中 7 项平均 MSE/MAE 优于包括 PatchTST 在内的基线（但 862 维 Traffic 排第二，且引入离线建库开销与数据集绑定约束）[^src-cometnet]
- **前置 token 压缩插件（跨槽位正交）**：[[source-timebase|TimeBase]]（ICML 2025）在 PatchTST 完成序列切分后、送入 Transformer 编码器前接入正交约束低秩基提取投影（将输入分段压缩为默认 $R=6$ 个时间基），可削减 PatchTST $77.7\%\text{–}93.0\%$ 的 MACs 且整体精度大体持平，但在 ETTh1/ETTh2（$H=720$）与 Weather（$H=336$）等设置上 MSE 轻微退回，且引入主周期 $P$ 先验[^src-timebase]
- **被对比骨干（实证复现对齐）**：[[source-tailfix|TailFix]]（NeurIPS 2026）在 WPMixer 骨干外叠加零初始化门控残差以修正末端退化，但 Traffic（$L=1200$）实测 MSE $0.410$ 逊于 PatchTST（$0.391$）与原版 WPMixer（$0.403$）[^src-tailfix]
- **核心概念**：[[patch-based-tokenization]], [[channel-independence]], [[instance-normalization]], [[lstf]]
- **自监督/预训练**：[[mae|MAE]], [[videomae]], [[timesfm|TimesFM]], [[chronos]]

[^src-patchtst]: [[source-patchtst]]
[^src-srsnet]: [[source-srsnet]]
[^src-probts]: [[source-probts]]
[^src-tide]: [[source-tide]]
[^src-tess]: [[source-tess]]
[^src-pir]: [[source-pir]]
[^src-2607-01918]: [[source-2607-01918]]
[^src-witran-jia-2023]: [[source-witran-jia-2023]]
[^src-tirex]: [[source-tirex]]
[^src-timer-xl]: [[source-timer-xl]]
[^src-dmmv]: [[source-dmmv]]
[^src-cometnet]: [[source-cometnet]]
[^src-timebase]: [[source-timebase]]
[^src-tailfix]: [[source-tailfix]]
