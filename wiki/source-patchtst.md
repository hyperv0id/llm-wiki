---
title: "PatchTST: A Time Series is Worth 64 Words"
type: source-summary
tags:
  - time-series
  - forecasting
  - transformer
  - patch
  - channel-independence
  - self-supervised
  - ICLR-2023
created: 2026-05-30
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# PatchTST: A Time Series is Worth 64 Words

**Authors**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam (Princeton University & IBM Research)

**Venue**: ICLR 2023 | **arXiv**: 2211.14730v2

## 核心贡献

PatchTST 提出两种关键设计显著提升 Transformer 在多元时间序列长期预测中的表现：(1) **Patching**——将时间序列分段为子序列级 patch 作为 Transformer 输入 token；(2) **Channel Independence**——每个通道（变量）独立处理，共享 embedding 和 Transformer 权重 [^src-patchtst]。

## Patching 机制

每个单变量时间序列 $x^{(i)} \in \mathbb{R}^{1 \times L}$ 被分割为 patch 长度 $P$、步长 $S$ 的重叠或非重叠 patch，生成 $N = \lfloor(L-P)/S\rfloor + 2$ 个 patch（通过末尾填充最后值的重复实现）。Patching 带来三重收益 [^src-patchtst]：

1. **局部语义保留**：单个时间步无语义（不像 NLP 中的词），patch 聚合保留局部信息
2. **二次方复杂度降低**：输入 token 数从 $L$ 降至 $L/S$，注意力复杂度按 $S^2$ 因子下降
3. **更长历史窗口**：相同计算约束下可看到更长历史（L=336→MSE 0.367 vs L=96→MSE 0.518）

## Channel Independence

多元时间序列被视为多通道信号，每个通道独立送入共享权重的 Transformer。此前 CI 在 CNN 和线性模型中已证明有效，但 PatchTST 是首个将其应用于 Transformer 的模型 [^src-patchtst]。与 [[channel-independence|channel-mixing]] 对比，CI 增加训练样本量（M 个通道→M 个独立样本）、改善各通道分布学习、使注意力聚焦时间维度而非被通道间相关性分散。

## 架构细节

- **Transformer Encoder**：使用 vanilla Transformer encoder，patch 通过线性投影 $W_p \in \mathbb{R}^{D \times P}$ 映射到 $D$ 维空间，加可学习位置编码 $W_{pos} \in \mathbb{R}^{D \times N}$
- **Instance Normalization**：patching 前对每个单变量序列做零均值单位方差归一化，预测后将均值和标准差加回 [^src-patchtst]
- **Loss**：各通道 MSE 损失取平均：$\mathcal{L} = \frac{1}{M}\sum_{i=1}^{M} \|\hat{x}^{(i)}_{L+1:L+T} - x^{(i)}_{L+1:L+T}\|_2^2$
- 使用 **BatchNorm** 而非 LayerNorm（已证明在时序 Transformer 中更优 [^src-patchtst]）

## 自监督表示学习

PatchTST 的 masked autoencoder 版本（借鉴 CV [[mae|MAE]] 范式）：将非重叠 patch 随机掩码（40%），训练重建被掩码的 patch。相比此前 point-wise 掩码（TST/Zerveas et al. 2021），patch-level 掩码避免了"通过插值即可恢复"的问题 [^src-patchtst]。

关键结果 [^src-patchtst]：
- **Fine-tuning** 在大数据集上超越 supervised training（Traffic MSE 0.349 vs 0.367）
- **Transfer Learning**：在 Electricity 预训练→迁移至 Traffic/Weather，仍优于 DLinear 和其他 Transformer
- 与 TS2Vec/BTSF/TNC/TS-TCC 对比，linear probing 即可取得 34.5%–48.8% 的 MSE 提升

## 预测性能

在 8 个数据集上，PatchTST/64 相比最佳 Transformer 基线取得 **21.0% MSE 降幅**和 **16.7% MAE 降幅**；PatchTST/42 取得 **20.2% MSE 降幅**和 **16.4% MAE 降幅** [^src-patchtst]。相比 DLinear，PatchTST 在大和中等数据集（Weather, Traffic, Electricity, ILI）上优势明显。

训练速度提升：Patching 将 Traffic 训练时间从 10040s 降至 464s（**22× 加速**），Electricity 19× 加速 [^src-patchtst]。

## 消融实验

- **Patching + CI** 缺一不可：移除 patching（仅 CI）MSE 上升；移除 CI（仅 patching）MSE 更高；两者结合取得最优 [^src-patchtst]
- **更长 look-back window**：PatchTST 是唯一随 L 增大持续降低 MSE 的模型，其他 Transformer（FEDformer/Autoformer/Informer）反而 L 增大时性能不变或变差 [^src-patchtst]

## 局限性

- Channel Independence 完全忽略跨变量依赖，论文也承认这是未来工作方向 [^src-patchtst]
- 在小数据集（ETT 系列）上优势不明显，部分设置 DLinear 仍更优

## 历史意义

PatchTST 是 [[lstf|LSTF]] 领域的里程碑，首次证明 Transformer 在正确设计下（patch + CI）可以超越简单线性模型。其 patch 化设计被后续模型广泛采用（[[simdiff|SimDiff]], [[cvpe|CVPE]], [[sparsetsf|SparseTSF]], [[cyclenet|CycleNet]]），Channel Independence 成为时序 Transformer 的默认处理策略。[[tslib|TSLib]] 评价 Patch-wise Transformers (PatchTST) 在长期预测中表现卓越。

[^src-patchtst]: [[source-patchtst]]
