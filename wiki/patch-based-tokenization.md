---
title: "Patch-based Tokenization"
type: technique
tags:
  - time-series
  - transformer
  - tokenization
  - patch
created: 2026-04-28
last_updated: 2026-08-06
source_count: 8
confidence: high
status: active
---

# Patch-based Tokenization

Patch-based tokenization 是现代时间序列 Transformer 使用的关键预处理技术，将连续的时间序列分割为固定长度的子序列（patch）作为输入 token，而非逐点输入。**PatchTST** (ICLR 2023) 首次系统地将 patch tokenization 引入时序 Transformer 并证明其三重收益：局部语义保留、二次方复杂度降低、更长历史窗口 [^src-patchtst]。此后 [[simdiff|SimDiff]], [[cvpe|CVPE]], [[sparsetsf|SparseTSF]] 等均采用该设计 [^src-simdiff]。

## 方法

1. **Patch 划分**：将时间序列划分为重叠或非重叠的固定长度窗口（patches），PatchTST 使用 $P=16, S=8$（可重叠）[ ^src-patchtst]
2. **Token 映射**：每个 patch 通过线性投影 $W_p \in \mathbb{R}^{D \times P}$ 转换为 token embedding [^src-patchtst]
3. **位置编码**：可学习位置编码 $W_{pos} \in \mathbb{R}^{D \times N}$ 监控 patch 的时序 [^src-patchtst]

## 优势

- **局部语义保留**：单个时间步无语义（不像 NLP 中的词），patch 聚合保留局部信息 [^src-patchtst]
- **计算效率**：输入 token 数从 $L$ 降至 $L/S$，注意力复杂度按 $S^2$ 下降 [^src-patchtst]
- **更长历史窗口**：相同计算约束下可看到更长历史（L=336→MSE 0.367 vs L=96→MSE 0.518）[^src-patchtst]
- **信息聚合**：patch 内信息自动聚合，减少噪声 [^src-simdiff]

## 论文自述的批评（Zeus, ICML 2026）

[[zeus|Zeus]]（ICML 2026）在论文附录 C.1 中对 patch tokenization 提出三点自述批评 [^src-2607-01918]：

- **纠缠细粒度变化**：patch 将细粒度变化纠缠在一起，削弱逐点推理能力
- **预训练目标过拟合 patch-wise 缺失**：patch 级重建预训练过拟合 patch 粒度的缺失模式，难以泛化到逐点缺失
- **周期退化为 FFN**：当序列周期恰好等于 patch 长度时，所有 patch 相同，Transformer 退化为 FFN

论文报告了对应实证（表 6）：将 MOMENT 的预训练掩码从 patch-missing 换成 point-missing 后，插补平均 MSE 恶化 −22.4%（ETTm1 −21.7%、ETTh2 −24.8%、Weather −16.9%），论文归因于 patch 预训练目标与逐点缺失分布不匹配（OOD）[^src-2607-01918]。

## 在 PatchTST 中的应用

**PatchTST** 是首个系统地将 patch tokenization 引入时序 Transformer 的模型 [^src-patchtst]。每个单变量时间序列被分割为 $P=16, S=8$ 的 patch，通过线性投影 $W_p \in \mathbb{R}^{D \times P}$ 和可学习位置编码 $W_{pos} \in \mathbb{R}^{D \times N}$ 映射到 Transformer 输入空间。两个变体：PatchTST/42（L=336, 42 patches）和 PatchTST/64（L=512, 64 patches）。在 Traffic 上训练时间从 10040s 降至 464s（22× 加速）[^src-patchtst]。

## 在 SimDiff 中的应用

SimDiff 使用 patch-based tokenization 将时间序列转换为重叠的 tokens，然后通过 Transformer backbone 进行去噪处理 [^src-simdiff]。这种设计使模型能够平衡简洁性和深度，确保鲁棒且高效的时间序列预测 [^src-simdiff]。

## 跨变量增强

### Crossformer DSW Embedding

[[dsw-embedding|DSW Embedding]] 与 patch tokenization 类似（都是单变量分段），但保留 2D 结构（时间 × 维度）以支持跨维度注意力，而 patch tokenization 通常展平为 1D [^src-crossformer-2023]。DSW embedding 是 Crossformer 的核心组件，使跨维度依赖可被显式建模。

### CVPE (Cross-Variate Patch Embedding) 在 vanilla patch embedding 后添加可学习位置编码 $W_P \in \mathbb{R}^{P \times d_m}$ 和 Router-Attention 机制，使 patch token 在被送入后续 CI 层之前就已携带跨变量信息 [^src-cvpe-2025]。这意味着即使后续层将序列拆分为 N 个独立通道处理，跨变量上下文仍能通过 patch embedding 传播 [^src-cvpe-2025]。实验证明：在 Weather 和 Traffic 等强相关数据集上显著提升（↓4.6%-6.7% MSE），但在弱相关数据集上可能过拟合 [^src-cvpe-2025]。

## iTransformer Variate Token：Patch 的极端情况

[[itransformer|iTransformer]] 的 [[variate-token-embedding|variate token]] 可视为 patch token 的极端情况——将**整条时间序列**作为一个 token，最大化感受野[^src-itransformer]。与 patch tokenization 的区别：

| 特性 | Patch Token | Variate Token |
|------|-----------|--------------|
| 序列覆盖 | 局部（固定长度窗口） | 全局（整条序列） |
| token 数量 | 多（T/P 个 per variate） | 少（1 个 per variate） |
| 计算复杂度 | attention O((N·T/P)²) | attention O(N²) |
| 局部精细度 | 高 | 低（由 FFN 弥补） |
| 跨变量交互 | 跨变量 patch 间（Crossformer） | variate token 间（iTransformer） |

iTransformer 论文指出 PatchTST 在 PEMS 波动序列上表现不佳——patch 机制可能丢失局部关注[^src-itransformer]。而 variate token 整合整条序列表示，对波动更鲁棒。

## 相关技术

- **起源**：[[patchtst|PatchTST]] — 首次系统引入 patch tokenization 的时序 Transformer (ICLR 2023)
- 对比：[[channel-independence]] — 通道独立处理
- 对比：[[instance-normalization]] — RevIN 策略
- 相关：[[normalization-independence]] — SimDiff 的归一化技术
- 相关：[[cvpe]] — 跨变量增强的 patch embedding
- 相关：[[router-attention-for-cvpe]] — CVPE 的聚合-分发注意力机制
- 相关：[[learnable-patch-position-encoding]] — CVPE 的可学习位置编码
- 相关：[[dsw-embedding]] — Crossformer 的 2D 分段嵌入
- 相关：[[sundial]] — Sundial 使用 patch tokenization (P=16) 实现 patch 级预测，减少自回归步数 (ICML 2025)[^src-sundial]
- **固定 → 自适应**：[[selective-representation-space|SRS]] / [[selective-patching|Selective Patching]] 批评固定 stride 的 representation space，以可微选 patch + 重排 + 融合增强 patch 骨干（含 [[srsnet|SRSNet]]）[^src-srsnet]
- 批评者：[[zeus]] — Zeus 自述批评 patch 纠缠细粒度变化、patch 级重建对逐点缺失 OOD、周期 = patch 长度时退化为 FFN (ICML 2026)[^src-2607-01918]
- 相关：[[multi-objective-temporal-masking]] — Zeus 的 MOTM（Multi-Objective Temporal Masking）多目标掩码预训练

[^src-simdiff]: [[source-simdiff]]
[^src-patchtst]: [[source-patchtst]]
[^src-cvpe-2025]: [[source-cvpe-2025]]
[^src-crossformer-2023]: [[source-crossformer-2023]]
[^src-itransformer]: [[source-itransformer]]
[^src-sundial]: [[source-sundial]]
[^src-srsnet]: [[source-srsnet]]
[^src-2607-01918]: [[source-2607-01918]]