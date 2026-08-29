---
title: "PriSTI"
type: technique
tags:
  - diffusion-models
  - spatiotemporal-imputation
  - conditional-diffusion
  - attention-models
  - graph-neural-networks
  - air-quality
  - traffic
last_updated: 2026-08-29
source_count: 6
confidence: medium
status: active
---

# PriSTI

**PriSTI**（**P**rio**r**-guided Conditional Diffusion Framework for **S**pa**t**iotemporal **I**mputation）是 Liu, Huang 等人于 2023 年（ICDE 2023）提出的条件扩散时空插补框架[^src-pristi]。它是 [[csdi|CSDI]] 之后扩散插补方法在时空场景下的关键升级——核心贡献在于将"条件信息的使用方式"从 [[csdi|CSDI]] 的"混合输入、硬拼硬学"改造为"先提取条件先验、后引导噪声去噪"的分离式设计[^src-pristi]。

## 问题与动机

2021 年，CSDI 证明了扩散模型可以做多变量时间序列缺失值插补，但它有两个关键缺陷[^src-pristi]：

1. **忽略空间/地理信息**：[[csdi|CSDI]] 将所有传感器视为独立"特征"，通过全连接 self-attention 从数据中发现依赖关系——两个监测站相距 10 公里还是 1 公里，在 [[csdi|CSDI]] 眼中没有区别[^src-pristi]。
2. **条件信息使用方式粗糙**：CSDI 直接将干净观测值与加噪目标值拼接，仅靠二进制 mask 区分，导致噪声污染注意力权重计算——尤其在高噪声扩散步（$t$ 接近 $T$）时，噪声目标的时序趋势被破坏殆尽，与干净条件信息之间存在严重的不一致性，增加了时空依赖学习的难度[^src-pristi]。

PriSTI 将问题精确定位于"条件信息的构建与利用"（construction and utilization of conditional information），而非扩散过程本身[^src-pristi]。其核心洞察是：扩散模型的注意力不应该在"黑暗中摸索"（用噪声输入算注意力权重），而应先在一个干净的环境中"画出地图"（用条件信息算注意力权重），再沿着这张地图去"走"[^src-pristi]。

## 核心设计

### 条件扩散框架

正向扩散仅作用于插补目标 $\tilde{X}^0$[^src-pristi]：

$$q(\tilde{X}^t | \tilde{X}^{t-1}) = \mathcal{N}(\tilde{X}^t; \sqrt{1-\beta_t}\tilde{X}^{t-1}, \beta_t I)$$

条件逆向过程受插值条件信息 $\bar{X}$ 和地理邻接矩阵 $A$ 条件化[^src-pristi]：

$$p_\theta(\tilde{X}^{0:T-1} | \tilde{X}^T, \bar{X}, A) = \prod_{t=1}^T p_\theta(\tilde{X}^{t-1} | \tilde{X}^t, \bar{X}, A)$$

采用与 [[ddpm|DDPM]] 一致的 $\varepsilon$ 预测参数化，训练目标是对 DDPM $L_{\text{simple}}$ 的直接条件推广[^src-pristi]：

$$L(\theta) = \mathbb{E}_{\tilde{X}^0, \varepsilon, t}\left[ \|\varepsilon - \varepsilon_\theta(\tilde{X}^t, \bar{X}, A, t)\|^2 \right]$$

### 创新一：线性插值增强条件信息

训练时，PriSTI 不对观测值直接使用，而是先对每个节点的时序做线性插值得到增强条件信息 $\bar{X}$[^src-pristi]。线性插值的价值不在于充填精度，而在于：确定性（无随机噪声）、快速（适配训练时实时构造）、不引入模型偏差、在连续缺失（block-missing）场景下提供粗粒度但一致的时空趋势骨架[^src-pristi]。

### 创新二：条件特征提取模块（Conditional Feature Extraction Module）

将 $\bar{X}$ 通过 $1 \times 1$ 卷积投影到隐空间 $H \in \mathbb{R}^{N \times L \times d}$（$d=64$），然后通过单层宽网络、三路并行提取全局上下文先验 $H_{pri}$[^src-pristi]：

$$\begin{aligned}
\phi_{TA}(H) &= \text{Norm}(\text{Attn}_{tem}(H) + H) \\
\phi_{SA}(H) &= \text{Norm}(\text{Attn}_{spa}(H) + H) \\
\phi_{MP}(H, A) &= \text{Norm}(\text{MPNN}(H, A) + H) \\
H_{pri} &= \text{MLP}(\phi_{SA} + \phi_{TA} + \phi_{MP})
\end{aligned}$$

模块仅堆 1 层——"宽"而非"深"——确保 $H_{pri}$ 是粗粒度的全局概括而非精细细节，避免过拟合条件数据的噪声[^src-pristi]。MPNN 采用 [[gwnet|Graph WaveNet]] 的图卷积：双向距离矩阵 + 自适应可学矩阵作为邻接矩阵[^src-pristi]。

### 创新三：先验引导注意力（Noise Estimation Module）

这是 PriSTI 最核心的创新——在噪声估计模块的注意力计算中，Q 和 K 来自干净的 $H_{pri}$，仅 V 来自混合输入 $H_{in}$[^src-pristi]：

$$Q_T = H_{pri} \cdot W_T^Q,\quad K_T = H_{pri} \cdot W_T^K,\quad V_T = H_{in} \cdot W_T^V$$

$$A_T = \text{softmax}(Q_T K_T^T / \sqrt{d}),\quad \text{Output} = A_T \cdot V_T$$

空间注意力同理。这一设计实现了"看哪儿由先验决定，看到什么由当前输入提供"——无论扩散步 $t$ 多大、噪声多强，注意力权重始终反映干净的时空依赖结构[^src-pristi]。噪声只影响 V（被注意的内容），不影响 A（注意的分布）[^src-pristi]。

两模块的架构互补[^src-pristi]：
- **条件特征提取模块**：1 层宽网络（三路并行），因为 $H_{pri}$ 需要同时概括时/空/图多种依赖类型
- **噪声估计模块**：4 层深网络（先时间注意，后空间+图聚合），因为去噪需要逐层精细化和深层抽象

### 虚拟节点降采样空间注意力

在大规模传感器网络（$N$ 大）中，将 $N$ 个节点的键值映射到 $k$ 个虚拟节点（$k \ll N$）[^src-pristi]：

$$K_S = H_{pri} \cdot P_S^K \cdot W_S^K,\quad V_S = H_{tem} \cdot P_S^V \cdot W_S^V$$

复杂度从 $O(N^2 d)$ 降至 $O(Nkd)$。AQI-36 上 $k=16$（36 节点），METR-LA/PEMS-BAY 上 $k=64$（207-325 节点）[^src-pristi]。

### 架构流程

```
观测值 X + 邻接矩阵 A
    │
    ▼ 线性插值
增强条件信息 X̄
    │
    ▼ Conditional Feature Extraction Module (1层宽网络)
全局上下文先验 H_pri  ──────────────────────┐
    │                                        │
    ▼ Noise Estimation Module (4层深网络)     │
H_in = Conv(X̃^t || X̄)                       │
    │                                        │
    ▼ γ_T: 时域注意力 (Q,K←H_pri, V←H_in)  ◄─┘
H_tem
    │
    ▼ γ_S: 空间注意力 + MPNN (Q,K←H_pri, V←H_tem)
H_spa → 残差连接(下一层) + skip connection(汇总)
    │
    ▼ 门控激活 → 2层1×1卷积 → ε_θ 输出
```

## 实验结果

### 确定性插补（MAE）

在三个数据集上全面优于 CSDI[^src-pristi]：

| 数据集 | PriSTI MAE | CSDI MAE | 提升 |
|--------|-----------|----------|------|
| AQI-36 (模拟故障, 24.6%) | 9.03 | 9.51 | -5.0% |
| METR-LA (block-missing, 16.6%) | 1.86 | 1.98 | -6.1% |
| METR-LA (point-missing, 31.1%) | 1.72 | 1.79 | -3.9% |
| PEMS-BAY (block-missing, 9.2%) | 0.78 | 0.86 | -9.3% |
| PEMS-BAY (point-missing, 25.0%) | 0.55 | 0.57 | -3.5% |

Block-missing 场景下提升更显著——插值条件信息在连续缺失时价值被放大[^src-pristi]。

### 概率插补（CRPS）

CRPS 同样全面超越 CSDI：AQI-36 0.0997 vs 0.1056（-5.6%），METR-LA block 0.0244 vs 0.0260（-6.2%），PEMS-BAY block 0.0093 vs 0.0127（-26.8%）[^src-pristi]。

### 高缺失率鲁棒性

90% 缺失率下，PriSTI 相比 BRITS/GRIN/CSDI 的 MAE 提升 4.67%-34.11%（block-missing）和 3.89%-43.99%（point-missing）[^src-pristi]。缺失率越高，PriSTI 相对优势越大——线性插值提供的粗粒度先验在高稀疏度下成为关键骨架[^src-pristi]。

### 消融实验

| 变体 | AQI-36 MAE | METR-LA Block MAE | 说明 |
|------|-----------|-------------------|------|
| mix-STI (无插值增强) | 9.83 | 1.93 | 插值增强在连续缺失中最有效 |
| w/o CF (无先验引导注意力) | 9.28 | 1.95 | 先验引导独立于插值带来增益 |
| w/o tem | 10.95 | 2.43 | 时间依赖贡献最大 |
| w/o spa | 10.07 | 3.51 | 交通数据对空间依赖极敏感 |
| w/o MPNN | 9.10 | 1.92 | 显式地理依赖与隐式空间注意力互补 |
| w/o Attn | 9.15 | 1.91 | 同上 |
| **PriSTI (完整)** | **9.03** | **1.86** | — |

## 与相关方法的对比

| 方法 | 任务 | 扩散域 | 空间建模 | 条件信息利用 | 推理模式 |
|------|------|--------|---------|-------------|---------|
| [[csdi|CSDI]] | 时序插补 | 原始域 | 无（全连接特征Transformer） | 拼接+mask | 非自回归 |
| PriSTI | 时空插补 | 原始域 | 空间注意力+MPNN | 先验提取→引导注意力 | 非自回归 |
| [[timegrad|TimeGrad]] | 时序预测 | 原始域 | 无 | RNN隐状态注入 | 自回归 |
| [[diffstg|DiffSTG]] | 时空预测 | 原始域 | GCN | 历史mask条件 | 非自回归 |
| [[specstg|SpecSTG]] | 时空预测 | 图谱域 | 谱域自然嵌入 | SG-GRU编码 | 非自回归 |

PriSTI 与 [[diffstg|DiffSTG]] 虽然都是"条件扩散+空间建模"，但任务不同（插补 vs 预测），空间建模方式不同（注意力+MPNN vs GCN），条件信息处理方式不同（先验引导注意力 vs mask/拼接条件）[^src-pristi]。

## 超参数

| 参数 | AQI-36 | METR-LA | PEMS-BAY |
|------|--------|---------|----------|
| 扩散步数 $T$ | 100 | 50 | 50 |
| 最小噪声 $\beta_1$ | 0.0001 | 0.0001 | 0.0001 |
| 最大噪声 $\beta_T$ | 0.2 | 0.2 | 0.2 |
| 噪声调度 | 二次 | 二次 | 二次 |
| 隐维度 $d$ | 64 | 64 | 64 |
| 噪声估计层数 | 4 | 4 | 4 |
| 注意力头数 | 8 | 8 | 8 |
| 虚拟节点 $k$ | 16 | 64 | 64 |
| 时间窗口 $L$ | 36 | 24 | 24 |

采用二次噪声调度 $\beta_t = \left(\frac{t-1}{T-1}\sqrt{\beta_T} + \frac{T-t}{T-1}\sqrt{\beta_1}\right)^2$，与 [[csdi|CSDI]] 一致[^src-pristi]。

## 局限性

1. **线性插值的双刃剑**：在高度非线性时序模式（如交通速度早晚高峰的剧烈脉冲）中，线性插值可能"错误平滑"真实突变信号[^src-pristi]
2. **静态图假设**：邻接矩阵 $A$ 是不随时间变化的距离矩阵，无法捕捉风向输送、早晚高峰路网连通性变化等动态空间关系[^src-pristi]
3. **单变量验证**：三个实验数据集均为每个节点仅一个特征（PM2.5 或速度），多变量场景未经验证[^src-pristi]
4. **计算效率开销**：条件特征提取模块（三层并行+MLP）和虚拟节点降采样增加约 20-25% 的训练和推理时间[^src-pristi]
5. **训练-评估代理差距**：$L_{\text{simple}}$ 是逐元素噪声预测 MSE，CRPS 是生成 100 条轨迹的经验 CDF 计算——训练目标与评估指标之间仅间接相关[^src-pristi]

## 后续影响

PriSTI 建立了"条件先验与去噪过程分离"的设计范式——这个核心洞察在 2023 年后逐渐成为高层次时空扩散模型的标准配置，其影响可见于 [[diffstg|DiffSTG]]、[[specstg|SpecSTG]] 后续时空扩散方法[^src-pristi]。更广泛地，"粗先验提取 + 精去噪建模"的信息处理结构在图像修复、3D 点云补全、视频预测等条件生成任务中反复出现[^src-pristi]。

从 CSDI 到 PriSTI 再到 [[fence|FENCE]]（AAAI 2026）的演化反映了扩散插补方法的改进路径[^src-fence]：
- **CSDI → PriSTI**：条件信息处理方式的升级（简单拼接 → 分离式先验引导）
- **PriSTI → FENCE**：引导机制的升级（固定 CFG 尺度 → 动态反馈引导 + 聚类感知）

FENCE 进一步指出 PriSTI 的固定引导尺度在低条件信息场景（如高缺失率节点）下同样面临漂移到先验分布的问题，并通过 [[feedback-diffusion-guidance|反馈扩散引导]] 和 [[cluster-aware-guidance|聚类感知引导]] 两个机制解决[^src-fence]。

## 关联页面

- [[ddpm]] — DDPM，PriSTI 的扩散数学基础
- [[csdi]] — CSDI，PriSTI 的直接前身，条件扩散时序插补
- [[diffusion-model]] — 扩散模型概念总览
- [[timegrad]] — 首个扩散时间序列预测方法
- [[diffstg]] — 首个扩散时空图预测框架
- [[specstg]] — 谱域扩散时空图预测
- [[generative-time-series-forecasting]] — 生成式时间序列预测范式
- [[std-mae]] — 时空解耦掩码预训练
- [[traffic-forecasting]] — 时空图交通预测总览
- [[cofill]] — 基于 PriSTI 的几何增强条件扩散插补
- [[imputeformer]] — 基于 Transformer 的扩散插补
- [[fence]] — FENCE，动态反馈引导扩散插补，PriSTI 的后续升级
- [[loft]] — LOFT (KDD 2026)，流匹配 + 低秩先验 + 轨迹一致性的插补模型，以 PriSTI 为生成式基线对比并作者报告占优[^src-loft]
- [[giflow]] — GiFlow (ICML 2026)，图信息先验流匹配插补，以 PriSTI 为扩散基线；作者报告 Air-36 推理时间 0.28 min vs PriSTI 9.30 min，且确定性 ODE 积分无需多次采样平均[^src-giflow]
- [[rdpi]] — RDPI (AAAI 2025)，两阶段残差条件扩散插补，协议沿用 PriSTI 但未将其列为基线[^src-rdpi]

## 相关工作

- [[maginet|MagiNet]] (arXiv 2024) 将 PriSTI 列为基线，并发现 PriSTI 仅在低方差的 PEMS-BAY 数据集上略优于 MagiNet（归因于扩散多步生成更适合低方差数据）；其余四个数据集 MagiNet 更优[^src-maginet]。
- [[rdpi|RDPI]] (AAAI 2025 / arXiv 2024)：其数据划分与缺失协议沿用 GRIN 与 PriSTI（RDPI 论文 Settings 节自述引 Liu et al. 2023），但其基线清单不含 PriSTI；两篇论文各自报告自己设置下的结果，数字不可直接混用[^src-rdpi]。

[^src-pristi]: [[source-pristi]]
[^src-fence]: [[source-fence]]
[^src-maginet]: [[source-maginet]]
[^src-loft]: [[source-loft]]
[^src-giflow]: [[source-giflow]]
[^src-rdpi]: [[source-rdpi]]
