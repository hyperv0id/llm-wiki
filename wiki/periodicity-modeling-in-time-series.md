---
title: "时序周期性建模文献梳理"
type: analysis
tags:
  - time-series
  - forecasting
  - periodicity
  - frequency-domain
  - decomposition
  - survey
created: 2026-04-28
last_updated: 2026-05-04
source_count: 17
confidence: medium
status: active
---

# 时序周期性建模文献梳理

本文梳理 2017–2026 年间深度时间序列预测中**周期性建模**的主要方法路线。周期性是真实世界时间序列（交通、能源、气象、金融）最突出的结构特征之一，但不同模型对"如何利用周期性"给出了截然不同的回答。本专题按方法学路线组织，重点关注本 wiki 已收录的 12 篇核心文献。

## 路线〇：高效 Transformer 基础

这类方法解决 LSTF（长序列时间序列预测）的计算瓶颈，为后续 Transformer-based 周期建模提供了效率基础。

### Informer（AAAI 2021 Best Paper）

[[informer|Informer]] 是首个系统解决 Transformer 应用于 LSTF 三大瓶颈的工作：$O(L^2)$ 计算、$O(J \cdot L^2)$ 内存、自回归解码速度慢[^src-zhou-informer-2021]。核心创新：
- **ProbSparse 自注意力**：利用注意力分布的长尾特性，仅对 Top-$c \cdot \ln L$ 个 query 做全量注意力计算，复杂度降至 $O(L \log L)$
- **自注意力蒸馏**：编码器中通过 1D 卷积 + MaxPool 逐层压缩时序维度，形成金字塔特征
- **生成式解码**：一次前向预测全部输出，避免累计误差

虽然 Informer 本身不直接建模周期性，但其建立了 Transformer-based LSTF 的研究范式，[[autoformer|Autoformer]] 在其基础上将注意力完全替换为 Auto-Correlation（利用周期自相似性），开启了后续一系列基于周期/频域的工作。

## 路线一：频域操作

这类方法将时域信号转换到频域（Fourier / Wavelet），在频域直接建模周期模式。

### FEDformer（ICML 2022）

[[fedformer|FEDformer]] 用 Fourier/Wavelet 增强的注意力块替代标准 Transformer self-attention，且随机选取固定数量的 Fourier 模式（而非仅取低频），理论和实验均证明这比仅保留低频有效[^src-fedformer]。配合 MOEDecomp 的季节—趋势分解，达到 O(L) 线性复杂度。

- **核心贡献**：首次系统论证 Transformer + 频域注意力在长时预测上的效力
- **局限**：模式选择不依赖输入自适应（后续 Dualformer 改进）

### FreTS（NeurIPS 2023）

[[source-frets|FreTS]] 将 MLP 从时域移到频域，利用 Fourier 变换的两个固有性质：(1) 能量聚集（Parseval）—大部分信号能量集中在少数频率分量；(2) 全局视野—频域 MLP 等价于时域全局长卷积[^src-frets]。

- **核心贡献**：证明简单 MLP 在频域可匹敌复杂 Transformer，参数和训练时间降低 3–10×
- **局限**：统一处理所有频率分量，无自适应频率选择机制

### Dualformer（2026）

[[dualsformer|Dualformer]] 针对 Transformer 固有的低通滤波效应（深层网络中高频信息被逐步衰减），提出**层级频率采样（HFS）**：浅层接收高频分量捕捉局部细节，深层接收低频分量建模长期趋势[^src-dualformer]。双分支架构（时域分支 + 频域分支）通过周期感知加权动态融合，权重由输入信号的谐波能量比决定。

- **核心贡献**：结构性反低通滤波，从层级视角重新思考频率分配
- **局限**：高度周期性数据（Traffic）上不如专门提取主导周期的方法

### AFE-TFNet

[[source-afe-tfnet|AFE-TFNet]] 在滚动窗口框架内并行使用 Wavelet Transform（捕捉局部时频特征）和 FFT（捕捉全局周期特征），用 Frequency Inception Block 进行多尺度卷积，DHSEW（主导谐波序列能量加权）机制动态融合时域和频域特征[^src-afe-tfnet]。

- **核心贡献**：WT+FFT 联合提取，滚动窗口杜绝数据泄漏
- **局限**：实验局限于海浪高度预测，泛化性未验证

## 路线二：周期分解架构

这类方法将周期性建模嵌入模型结构本身——通过分解块、多周期张量变换、或周期残差学习，使网络**天然感知周期结构**。

### Autoformer（NeurIPS 2021）

[[autoformer|Autoformer]] 将序列分解作为**模型内部模块**而非预处理——编码器和解码器中交替插入 Series Decomposition Block，逐步从中间隐变量提取趋势，使得深层也可利用浅层分离出的季节成分[^src-autoformer]。Auto-Correlation 机制通过 FFT 计算自相关发现周期，以子序列（而非点）为单位做时间延迟聚合，打破稀疏自注意力的信息利用瓶颈。

- **核心贡献**：渐进式分解 + 子序列级聚合 O(L log L)，38% 相对 MSE 提升
- **局限**：假定可加性趋势 + 季节结构；非平稳序列下 FFT 周期估计可能不可靠

### TimesNet（ICLR 2023）

[[timesnet|TimesNet]] 另辟蹊径：将 1D 时间序列按发现的多个周期重排为 2D 张量——列 = 周期内时序步，行 = 周期间相位——于是**周期内变化**和**周期间变化**分别成为 2D 卷积核可直接捕捉的列方向和行方向模式[^src-timesnet]。TimesBlock 内嵌参数高效的 Inception block，可替换为任意视觉骨干网络。

- **核心贡献**：1D→2D 变换，架接时间序列与计算机视觉；跨五大任务一致 SOTA
- **局限**：FFT 周期发现假设数据有明显周期性；弱周期数据收益不明确

### PRNet

[[source-prnet|PRNet]] 将人流预测重塑为**周期残差学习**问题：不直接预测高度动态的人流量，而是预测当前时间点与同一时刻一周期之前之间的残差——该残差远比原始流量平稳[^src-prnet]。差异函数（DIFF）先移除季节性，融合函数（FUSE）结合残差预测和周期预测重建输出。

- **核心贡献**：周期残差为可插拔模块，降低参数 1–147 倍同时提升精度
- **局限**：依赖已知固定周期长度（日、周）；网格化 CNN 对不规则传感器布局不适用

### HyperD（2025）

[[hyperd|HyperD]] 提出**混合周期解耦**（Hybrid Periodicity Decoupling），将交通信号显式分解为**短周期**（intra-day, 1–6 小时）、**长周期**（daily/weekly）和**残差**三部分[^src-hyperd]。FR 模块通过 RFFT 将频率分量聚类为高/中/低三频段，分别对应残差、短周期嵌入、长周期嵌入。双路径编码器（STFE + FR-STFE）各自处理原始信号和残差信号，由对应的频率嵌入引导。fDMLP 解码器进一步在预测阶段做趋势分离。

- **核心贡献**：首次在编码器层面实现短/长周期路径分离；频率阈值通过学习获得，自动对齐日周期（288 步/天）和周周期（2016 步/周）
- **局限**：针对交通预测设计，泛化到一般时间序列未验证；PeMS 之外数据集覆盖有限

### SparseTSF（TPAMI 2026 & ICML 2024 Oral）

[[sparsetsf|SparseTSF]] 提出 **Cross-Period Sparse Forecasting** 技术，将原始序列按周期 w 下采样为 w 个子序列，用共享参数的 Linear/MLP backbone 预测，再上采样回完整序列[^src-sparsetsf]。这种跨周期稀疏预测实现了两大效果：
- **极端压缩**：参数量从 O(L×H) 降至 O(L/w × H/w)，首次将 LTSF 模型降至 1k 级别
- **隐式正则化**：理论证明稀疏结构等价于 L1 正则化，增强鲁棒性

- **核心贡献**：首次从理论证明稀疏技术等价于隐式 L1 正则化；参数量 < 1k，压缩 1~4 个数量级
- **局限**：需要预先知道周期 w；弱周期数据可能因下采样损失信息

## 路线三：注意力机制中的周期先验

这类方法将周期结构编码为注意力偏置，使 Transformer 的注意力分布天然倾向周期对齐。

### PENGUIN（AISTATS 2026）

[[source-penguin|PENGUIN]] 为每个注意力头组注入周期偏置：对相对位置做模运算（模 = 周期长度），周期相同的位点获得更高注意力得分[^src-penguin]。各组分配不同周期长度（如 24 = 日周期，168 = 周周期），另设一组非周期偏置处理无周期性数据。周期信息通过 ACF 先验获取。

- **核心贡献**：证明仅在注意力中注入周期偏置即可让 Transformer 超越 MLP 模型
- **局限**：需要预先知道周期长度；非平稳数据集上弱于 CATS

### TQNet（ICML 2025）

[[tqn|TQNet]] 提出 **Temporal Query (TQ) 技术**，用周期性偏移的可学习向量作为注意力中的 Query，K/V 来自原始输入序列[^src-tqn]。这种设计使注意力同时融合：
- **全局先验**：TQ 向量捕捉稳定的变量间相关性
- **局部样本信息**：输入数据中的样本特异性变化

与 PENGUIN 的周期偏置不同，TQ 直接用可学习向量替换 Query，实现更灵活的全局相关性建模。

- **核心贡献**：首次将可学习 Query 引入时序预测注意力；极简架构（单层注意力 + 浅层 MLP）在 12 个数据集上取得 SOTA
- **局限**：需要预先确定周期长度 W；非平稳数据集上性能下降

## 路线四：时间分量分解

### ST-ResNet 与 ASTGCN

较早的空间—时间预测模型通过将输入组织为多个时间分量来隐式建模周期性：

- **[[source-st-resnet|ST-ResNet]]**（AAAI 2017）分 closeness/daily period/weekly trend 三个分量，各自经 ResNet 处理后通过参数矩阵加权融合[^src-st-resnet]。
- **[[source-astgcn|ASTGCN]]**（AAAI 2019）沿用了三段时间分量设计并引入时空注意力，使图卷积过程中的节点相关性和时间依赖性均动态调整[^src-astgcn]。

这种三段时间分量模式后来被 HyperD 等深度解耦方法所继承，但后者的关键改进在于**将分量内部进一步拆分**——不是三个独立的并行分支，而是将同一条信号显式分解为不同频段，各自用专门化结构处理。

## 路线五：轻量级稀疏周期建模

这类方法利用周期结构实现极端的模型压缩，通过稀疏/下采样策略减少参数量。

### SparseTSF（TPAMI 2026 & ICML 2024 Oral）

[[sparsetsf|SparseTSF]] 提出 **Cross-Period Sparse Forecasting** 技术，将原始序列按周期 w 下采样为 w 个子序列，用共享参数的 Linear/MLP backbone 预测，再上采样回完整序列[^src-sparsetsf]。这种跨周期稀疏预测实现了两大效果：
- **极端压缩**：参数量从 O(L×H) 降至 O(L/w × H/w)，首次将 LTSF 模型降至 1k 级别
- **隐式正则化**：理论证明稀疏结构等价于 L1 正则化，增强鲁棒性

- **核心贡献**：首次从理论证明稀疏技术等价于隐式 L1 正则化；参数量 < 1k，压缩 1~4 个数量级
- **局限**：需要预先知道周期 w；弱周期数据可能因下采样损失信息

### CycleNet（NeurIPS 2024）

[[cyclenet|CycleNet]] 提出 **Residual Cycle Forecasting (RCF)** 技术，使用可学习的循环周期 $Q \in \mathbb{R}^{W \times D}$ 显式建模数据内在的���期性模式[^src-cyclenet]。核心思想：
- 生成可学习循环周期 $Q$，通过循环复制得到周期分量 $C$[^src-cyclenet]
- 从原始输入中移除周期分量，对残差进行预测，最后加回周期分量[^src-cyclenet]

- **核心贡献**：首次系统论证显式建模全局周期模式的有效性；RCF 可作为即插即用模块提升 PatchTST、iTransformer 等模型性能；参数减少 90%+[^src-cyclenet]
- **局限**：在具有显著时空特性和极端值的 Traffic 数据集上性能略逊于 iTransformer；当前版本仅考虑单通道关系建模[^src-cyclenet]

## 交叉主题

### 频率分离策略的演变

| 模型 | 年份 | 分离粒度 | 自适应 |
|------|------|----------|--------|
| FEDformer | 2022 | 随机选取固定数量模式 | 否 |
| HyperD (FR) | 2025 | 三频段（高/中/低）聚类 | 学习阈值 |
| Dualformer (HFS) | 2026 | 按层级逐层分配频段 | 区间动态调整 |
| AFE-TFNet (DHSEW) | 2025 | 谐波能量比加权 | 动态加权 |

> 注：HyperD 的 FR 模块将频率分量聚类为高/中/低三段，对应残差、短周期、长周期——是首个在编码器层面实现**短/长周期路径分离**的模型。

路径清晰：从**固定选择 → 聚类 → 层级分配 → 动态加权**，频率分离的自适应性和结构化程度不断提升。

### 周期性认知的层级

1. **隐式**（ST-ResNet, ASTGCN）：通过时间分量组织输入让网络自己学习周期模式
2. **显式—分解**（Autoformer, HyperD）：将周期成分从信号中显式分离
3. **显式—变换**（TimesNet, FreTS）：通过域变换（Fourier, 2D reshape) 暴露周期结构
4. **显式—先验**（PENGUIN, PRNet, TQNet）：将周期知识编码为模型先验（注意力偏置、可学习 Query 或残差目标）
5. **显式—稀疏**（SparseTSF）：通过跨周期下采样实现周期结构利用 + 极端压缩

层级越高，模型对周期结构的"理解"越深入，但泛化到无周期或弱周期数据时风险越大。

### 分解到什么程度？

一个持续存在的设计张力：

- **趋势 vs 季节**：Autoformer、FEDformer 分趋势和季节两类（粗粒度）
- **短周期 vs 长周期**：HyperD 在季节内部再分短/长两类（中粒度）
- **多周期并行**：TimesNet 保留 top-k 个周期各自独立处理（细粒度）
- **周期残差**：PRNet 仅建模相对于已知周期的偏差（极简方案）

TPLib 调查的结论——"没有单一架构在所有任务上通用"——对周期建模同样适用[^src-survey]。高度周期性数据（交通、电力）受益于多周期并行处理，弱周期性数据（汇率、ILI）可能因过度周期化而受损。

## 时间线与中稿情况

| 模型 | 年份 | 发表 venue | 备注 |
|------|------|------------|------|
| [[source-st-resnet|ST-ResNet]] | 2017 | **AAAI** | 早期时间分量分解范式 |
| [[source-astgcn|ASTGCN]] | 2019 | **AAAI** | 注意力 + 图卷积 + 时间分量 |
| [[source-zhou-informer-2021|Informer]] | 2021 | **AAAI 2021** ★ | ProbSparse 注意力 + 生成式解码（Best Paper）|
| [[source-autoformer|Autoformer]] | 2021 | **NeurIPS** | 渐进分解 + Auto-Correlation（奠基之作）|
| [[source-fedformer|FEDformer]] | 2022 | **ICML** | 频域增强注意力 |
| [[source-prnet|PRNet]] | 2022 | DeepSpatial (KDD Workshop) | 周期残差学习 |
| [[source-timesnet|TimesNet]] | 2023 | **ICLR** | 1D→2D 多周期变换 |
| [[source-frets|FreTS]] | 2023 | **NeurIPS** | 频域 MLP 高效学习 |
| [[source-hyperd-hybrid-periodicity-decoupling|HyperD]] | 2025 | **AAAI 2026** | 短/长周期解耦 + 双路径编码器 ★ |
| [[source-afe-tfnet|AFE-TFNet]] | 2025 | Ocean Engineering (期刊) | WT+FFT 联合自适应提取 |
| [[source-deep-time-series-survey|TSLib Survey]] | 2025 | arXiv (期刊在投) | 系统性归类与基准评测 |
| [[source-dualformer|Dualformer]] | 2026 | arXiv (投稿中) | 层级频率采样反低通滤波 |
| [[source-penguin|PENGUIN]] | 2026 | **AISTATS 2026** | 周期嵌套分组注意力 |
| [[source-tqn|TQNet]] | 2025 | **ICML 2025** | Temporal Query 可学习向量 + 极简架构 |
| [[source-sparsetsf|SparseTSF]] | 2025 | **TPAMI 2026** & **ICML 2024 Oral** | 跨周期稀疏预测 <1k 参数 |
| [[source-cyclenet|CycleNet]] | 2024 | **NeurIPS 2024** | 可学习循环周期 RCF + 即插即用模块 |

> ★ 表示本 wiki 收录的 HyperD 原始论文（arXiv 2025-11, AAAI 2026 中稿）

## 开放问题

- **弱周期数据处理**：如何在不引入错误周期偏置的前提下让模型自适应退化到非周期建模
- **自适应周期发现**：当前多依赖 FFT/ACF 先验，但非平稳序列的周期可能随时间漂移
- **多周期融合效率**：TimesNet 的 top-k × 2D 卷积随周期数线性增长；PENGUIN 的分组注意力也有类似开销
- **理论基础**：除 Dualformer 的谐波能量下界和 FreTS 的全局卷积等价性外，多数模型缺乏关于周期建模为何有效的理论支撑

[^src-zhou-informer-2021]: [[source-zhou-informer-2021]]
[^src-fedformer]: [[source-fedformer]]
[^src-frets]: [[source-frets]]
[^src-dualformer]: [[source-dualformer]]
[^src-afe-tfnet]: [[source-afe-tfnet]]
[^src-autoformer]: [[source-autoformer]]
[^src-timesnet]: [[source-timesnet]]
[^src-prnet]: [[source-prnet]]
[^src-penguin]: [[source-penguin]]
[^src-st-resnet]: [[source-st-resnet]]
[^src-astgcn]: [[source-astgcn]]
[^src-survey]: [[source-deep-time-series-survey]]
[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-tqn]: [[source-tqn]]
[^src-sparsetsf]: [[source-sparsetsf]]
[^src-cyclenet]: [[source-cyclenet]]
