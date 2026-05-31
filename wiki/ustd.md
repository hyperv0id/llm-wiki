---
title: "USTD"
type: entity
tags:
  - diffusion-models
  - spatio-temporal-graph
  - probabilistic-forecasting
  - kriging
  - traffic-forecasting
  - sigspatial-2024
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# USTD (Unified Spatio-Temporal Diffusion)

**USTD** 是首个将去噪扩散概率模型（[[ddpm|DDPM]]）统一应用于时空图预测（forecasting）和插值（kriging）两个任务的概率框架，由 Hu, Liu, Fan, Liang & Zimmermann 于 SIGSPATIAL 2024 提出（arXiv:2310.17360）[^src-ustd]。核心贡献是通过预训练编码器与任务特定 denoiser 的解耦训练，首次让 diffusion 模型在时空预测上全面超越确定性 baseline。

## 核心思想

USTD 的出发点是：forecasting 和 kriging 本质上都在建模同一个条件分布 $P_{\phi,\theta}(Y|X)$——条件 $X$ 中的时空依赖是共享的，区别仅在于生成目标 $Y$ 的维度侧重（预测关注时间轴、插值关注空间轴）[^src-ustd]。基于此观察，USTD 设计了：

1. **共享预训练编码器**：从条件数据 $X$ 中提取确定性时空表示 $\mathbf{H}$，所有任务共用
2. **任务特定 denoiser**：TGA（预测）和 SGA（插值）在各自最关键的单维度上做 attention

## 架构

### Stage 1: 预训练编码器

编码器采用 GWNet 风格的 STGNN 层（gated 1D conv + GCN + skip connection），通过非对称 autoencoding 策略预训练 [^src-ustd]：

- **Graph Sampling**（80% 节点随机采样）：阻止编码器记忆完整图结构，使编码器学习"节点间的相对关系"而非"特定图结构的绝对位置"——对插值任务中图结构改变的场景至关重要
- **Masking**（75% rate, MAE 风格）：打破 latent space 维度大于输入的平凡解风险。mask 掉 3/4 的节点-时间格点，迫使编码器从稀疏信号中提取核心模式
- **TCN 无 padding**：时间维度被压缩至 $\tau \ll T$，条件被映射到低维 latent space（$d_h=64$），大幅降低后续 denoising 计算量

轻量解码器（3 层 STGNN + MLP）从 $\mathbf{H} \in \mathbb{R}^{N\times\tau\times d_h}$ 重建原始信号，损失为 MAE。预训练完成后解码器丢弃。

### Stage 2: 条件扩散 & 任务特定 Denoiser

给定预训练编码器提取的表示 $\mathbf{H}$，DDPM 的条件反向过程为 [^src-ustd]：

$$p_\theta(Y_{0:K} | \mathbf{H}, \mathcal{G}) = p(Y_K) \prod_{k=1}^K p_\theta(Y_{k-1} | Y_k, \mathbf{H}, \mathcal{G})$$

训练目标为简化噪声预测损失 $\mathcal{L}(\theta) = \mathbb{E}[\|\epsilon - \epsilon_\theta(\sqrt{\alpha_k}Y_0 + \sqrt{1-\alpha_k}\epsilon, \mathbf{H}, \mathcal{G}, k)\|^2]$。编码器以 denoiser 十分之一的学习率微调。

#### TGA: Temporal Gated Attention（预测）

流程 [^src-ustd]：
1. Flatten + Linear 将 denoised target $Y \in \mathbb{R}^{N\times T'\times d_y}$ 与编码器表示压为 $\mathbf{R} \in \mathbb{R}^{N\times d_h}$
2. **Cross-Attention（时间轴，每节点独立）**：$\mathbf{R}_i^{ca} = \text{softmax}(Q_i K_i^T / \sqrt{d_h}) V_i$，$Q_i$ 来自 target embedding，$K_i, V_i$ 来自编码器表示 $\mathbf{H}_i$
3. **Self-Attention（节点间）**：捕获目标节点间的空间交互
4. **Gated Fusion**：$\mathbf{R} = \sigma(\mathbf{R}^{ca}W_{g1} + \mathbf{R}^{sa}W_{g2} + b_g) \odot \mathbf{R}^{ca} + (1-\text{Gate}) \odot \mathbf{R}^{sa}$

可堆叠 2 层（最优配置）。注入时间嵌入（Informer 风格）和 diffusion embedding 作为额外上下文。

#### SGA: Spatial Gated Attention（插值）

与 TGA 结构相同，三处调整 [^src-ustd]：
1. Embedding 层吸收编码器表示的时间维度 → $\mathbf{H} \in \mathbb{R}^{N\times d_h}$
2. Cross-attention 在**空间轴**：$Q$ 来自未观测（目标）节点，$K, V$ 来自已观测节点表示
3. Self-attention 在目标节点间

复杂度从 $O(((M+N)\tau)^2)$ 降至 $O((M+N)M)$。

### 为什么解耦训练是关键

CSDI、PriSTI、DiffSTG 把编码器和 denoiser 耦合训练——同一个 loss 要同时逼编码器提取高质量特征 + denoiser 学会预测。两个子目标互相牵制，导致条件依赖捕获不充分，forecasting 上连 deterministic baseline 都打不过 [^src-ustd]。USTD 的解耦策略（先预训练学习条件依赖、再微调学习预测分布）打破了这一困境。

## 实验结果

4 个数据集（PEMS-03/PEMS-BAY 交通，AIR-BJ/AIR-GZ 空气质量）× 16 baselines [^src-ustd]：

### Forecasting（12→12）

| 数据集 | USTD MAE | 最佳 Deterministic | USTD CRPS | 最佳 Probabilistic | CRPS 提升 |
|--------|----------|-------------------|-----------|-------------------|-----------|
| PEMS-03 | 15.32 | 15.78 (GMSDR) | 0.087 | 0.092 (PriSTI) | ↓5.4% |
| PEMS-BAY | 1.63 | 1.67 (GWN) | 0.022 | 0.025 (PriSTI) | ↓12.0% |
| AIR-BJ | 9.70 | 9.98 (DCRNN) | 0.084 | 0.093 (DCRNN) | ↓9.7% |
| AIR-GZ | 30.09 | 30.60 (GWN) | 0.348 | 0.363 (TimeGrad) | ↓4.1% |

唯一被 deterministic 反超的指标：PEMS-BAY RMSE（STGODE 3.33 vs USTD 3.55），因为 STGODE 的 ODE 框架对长程依赖有特殊优势。

### Kriging（$N:M=2:1$）

USTD 在所有数据集上超越所有 baseline [^src-ustd]：
- PEMS-03: MAE 14.73 vs INCREASE 15.34 (↓4.0%), CRPS 0.071 vs PriSTI 0.076 (↓6.6%)
- PEMS-BAY: MAE 1.96 vs PriSTI 2.06 (↓4.9%)
- AIR-BJ: MAE 13.30 vs IGNNK 13.86 (↓4.0%)
- AIR-GZ: MAE 8.61 vs IGNNK 9.62 (↓10.5%), CRPS 0.213 vs PriSTI 0.231 (↓7.8%)

### 推理效率

USTD 推理 0.49–0.50s，比 CSDI (~0.88s) 快约 47%，比 PriSTI (~1.05s) 快约 2×。加速来源：编码器 TCN 压缩后的低维表示使 denoiser 的 attention 计算量大幅下降 [^src-ustd]。

### 消融关键发现

- **w/o EN**（去掉编码器）：性能崩塌——shared encoder 是系统基石
- **w/o PT**（不分阶段预训练）：性能显著下降——验证了耦合训练的代价
- **w/o MK**（去掉 masking）：下降——masking 防止 latent 平凡解
- **w/o GS**（去掉 graph sampling）：预测几乎不受影响，插值明显下降——GS 的核心价值在图泛化性
- **TGA r/p TCN**（用 TCN 替换 cross-attention）：预测大幅下降——TCN 假设均匀时间演进，与 denoised target 和条件的任意依赖语义不同

## 与相关方法的对比

| 方法 | 扩散域 | Encoder-Denoiser 耦合 | 时空建模 | 超越 Deterministic？ |
|------|--------|----------------------|---------|---------------------|
| [[diffstg|DiffSTG]] | 原始域 | 耦合（UGnet 内） | GCN + U-Net | 否 |
| [[csdi|CSDI]] | 原始域 | 耦合（Transformer） | 无（逐传感器） | 否 |
| [[pristi|PriSTI]] | 原始域 | 耦合（STGNN concat） | GNN + 先验引导 | 否 |
| [[specstg|SpecSTG]] | 图谱域 | 耦合（SG-GRU） | 谱域 GCN | 否（效率优势） |
| **USTD** | 原始域 | **解耦（预训练+微调）** | GWNet encoder + Gated Attn | **是**（除 PEMS-BAY RMSE） |

USTD 与 [[uniflow|UniFlow]]、[[urbandit|UrbanDiT]] 等时空基础模型的路线不同：USTD 走的是任务统一（一个编码器覆盖 forecasting+kriging）+ diffusion 概率建模路线，而非基础模型的零样本泛化路线。两种路线的互补性在于——USTD 的 masking+graph sampling 预训练策略可用于基础模型的编码器训练。

## 局限性

1. 目前仅覆盖 forecasting 和 kriging——imputation、分类、异常检测未统一 [^src-ustd]
2. TGA/SGA 分开训练，同一模型不能同时做预测和插值
3. 静态邻接矩阵，无动态图学习
4. 仅单模态数值输入
5. 未在超大规模图上验证（最大 358 节点 vs FaST 的 8,600+）

## 关联页面

- [[source-ustd]] — 论文摘要
- [[traffic-forecasting]] — 时空图交通预测总览
- [[diffusion-model]] — 扩散模型理论基础
- [[ddpm]] — DDPM 去噪扩散概率模型
- [[diffstg]] — DiffSTG，首个 STG 扩散框架（AAAI 2023）
- [[specstg]] — SpecSTG，谱域扩散 STG 预测（arXiv 2024）
- [[csdi]] — CSDI，条件扩散时序插补（NeurIPS 2021）
- [[pristi]] — PriSTI，先验引导扩散 ST 插补（ICDE 2023）
- [[spatio-temporal-foundation-model]] — 时空基础模型概念
- [[generative-time-series-forecasting]] — 生成式时间序列预测范式

[^src-ustd]: [[source-ustd]]
