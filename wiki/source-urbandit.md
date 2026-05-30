---
title: "UrbanDiT: Diffusion Transformers as Open-World Spatiotemporal Foundation Models"
type: source-summary
tags:
  - spatiotemporal
  - diffusion-model
  - transformer
  - foundation-model
  - traffic-forecasting
  - rectified-flow
created: 2026-05-12
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

**UrbanDiT** 是由清华大学电子工程系 FIB Lab（Yuan Yuan, Chonghua Han, Jingtao Ding, Guozhen Zhang, Depeng Jin, Yong Li）提出的开放世界城市时空基础模型，发表于 **NeurIPS 2025**[^src-urbandit]。其核心思想是将 Diffusion Transformer（DiT）架构扩展到城市时空领域，通过统一提示学习（Unified Prompt Learning）框架实现**多数据类型**和**多任务**的统一建模。

代码仓库：https://github.com/tsinghua-fib-lab/UrbanDiT [^src-urbandit].

## 核心贡献

1. **首个多数据类型、多任务统一的城市时空基础模型** — 同时处理 grid-based 和 graph-based 时空数据，支持 5 种任务[^src-urbandit]
2. **基于 DiT + Rectified Flow 的架构设计** — 使用 temporal attention + spatial attention 的 transformer decoder，训练采用 InstaFlow 提出的 rectified flow（straightened ODE trajectory），相比原始 DDPM 获得 25 倍加速[^src-urbandit]
3. **统一提示学习框架** — 创新性地使用三个 memory pool（时域、频域、空域）生成 data-driven prompts，加上从任务 mask 生成的 task-specific prompts[^src-urbandit]
4. **强大的零样本泛化** — 在未见城市上零样本性能超越多数有训练数据的基线模型[^src-urbandit]

## 架构设计

UrbanDiT 包含四个核心组件：

### 1. 数据统一化
将不同类型时空数据转换为统一的序列化格式，适配 Transformer 输入[^src-urbandit]：
- **Grid-based 数据**（Crowd Flow, Taxi Demand 等）：使用 3D CNN（kernel size = (pt, ps, ps)）处理 $X \in \mathbb{R}^{T \times H \times W}$，reshape 为一维序列
- **Graph-based 数据**（Traffic Speed 等）：使用 1D CNN（时间维度）+ GCN（空间图卷积），reshape 为序列
- 统一表示：$X^{N \times T}$，$N$ 为空间分区数（grid: $H \times W$，graph: 节点数）

### 2. 掩码策略统一多任务
通过不同的掩码 $M$ 将各类任务统一为"重建被掩码数据"的形式，输入去噪网络 $X_t = X_t \odot (1-M) + X_0 \odot M$[^src-urbandit]：
- **Forward Prediction**：掩码未来时间步
- **Backward Prediction**：掩码过去时间步
- **Temporal Interpolation**：掩码特定中间时间点
- **Spatial Extrapolation**：掩码部分空间区域
- **Spatio-Temporal Imputation**：随机掩码时空维度

### 3. 时空 Transformer Block
每个 transformer block 包含独立的 temporal attention 和 spatial attention（独立操作以降低复杂度，attention 复杂度随序列长度平方增长）[^src-urbandit]。时间步 $t$ 通过 adaLN（adaptive layer normalization）注入，prompt 与输入序列直接拼接。

### 4. 统一提示学习（Unified Prompt Learning）
这是 UrbanDiT 的核心创新[^src-urbandit]：

**Data-Driven Prompt**：使用三个 learnable 的 key-value memory pool 捕捉不同维度的模式：
- **时间域 Memory Pool** $(K_t, V_t)$：对每个空间位置独立做 temporal attention 提取时序模式
- **频域 Memory Pool** $(K_f, V_f)$：支持 4 种 FFT 配置（无阈值 / 均值阈值 / 分位数阈值 / Top-k 过滤），保留主要频率分量
- **空间域 Memory Pool** $(K_s, V_s)$：对每个时间 patch 独立做 spatial attention 提取空间模式

Prompt 通过 cosine similarity 检索最匹配的 patterns：$P_x = \sum \alpha_x \cdot V_x$，其中 $\alpha_x = \text{softmax}(X_x, K_x)$。

**Task-Specific Prompt**：从 mask 图 $M$ 通过 attention 机制生成 $P_m$，使模型感知当前任务类型。

最终输入：$X = \text{Concat}(P_t, P_f, P_m, X)$，与原始数据拼接后馈入 transformer blocks。

### 消融实验
移除任一类 prompt 均显著降低性能，其中**频域 prompt 影响最大**（对 RMSE 贡献最显著），完全移除所有 prompt 设计后模型性能最差[^src-urbandit]。

## 训练策略

采用 **rectified flow**（InstaFlow）训练方式，通过 straightened ODE trajectory 对齐噪声和数据分布，相比传统扩散模型的弯曲路径更高效[^src-urbandit]。训练过程在多个数据集和任务间交替采样，每轮随机选取一个数据集 $D_i \sim \text{Uniform}(D)$ 和一个任务 $T_i \sim \text{Uniform}(T)$ 进行梯度下降。

**推理效率**：扩散步数 500，推理步数 20 时达到最优平衡，相比原始 DDPM 实现 25 倍加速。模型采用概率预测，每个样本运行 20 次推理取平均[^src-urbandit]。

## 模型规模

三种配置[^src-urbandit]：
- **UrbanDiT-S**（small）：4 层 transformer，hidden size 256，4 attention heads
- **UrbanDiT-M**（medium）：6 层 transformer，hidden size 384，6 attention heads
- **UrbanDiT-L**（large）：12 层 transformer，hidden size 384，12 attention heads

每个 memory pool 包含 512 个 embedding，维度与 hidden size 一致。training epochs 上限 500（early stopping）。学习率 1e-4。

## 实验与性能

### 数据集
**Grid-based（6 个）**[^src-urbandit]：FlowSH（上海/15min）、PopBJ（北京/1h）、TaxiBJ（北京/30min）、CrowdNJ（南京/1h）、TaxiNYC（纽约市/30min）、PopSH（上海/1h）

**Graph-based（3 个）**[^src-urbandit]：SpeedSH、SpeedBJ、SpeedNJ（均为 traffic speed / 15min / 百万级节点）

训练/验证/测试按 6:2:2 沿时间维度切分。

### 基线对比
与 20+ 基线模型对比，涵盖传统方法（HA, ARIMA）、时空深度学习模型（STResNet, ACFM, STNorm, STGSP, MC-STL, STID）、视频预测模型（SimVP, TAU, MAU, MIM）、时序预测模型（PatchTST, iTransformer, Time-LLM）、图网络模型（STGCN, DCRNN, GWN, MTGNN, AGCRN, GTS, STEP）、扩散模型（CSDI）、插补模型（ImputeFormer, Grin, BriTS）、以及 UniST[^src-urbandit]。

### 关键结果

| 任务 | UrbanDiT 表现 | 对标基线 |
|------|-------------|---------|
| Forward Prediction（grid） | 最佳，相对提升 11.3% | UniST / CSDI 次优 |
| Forward Prediction（graph） | 最佳 | 见 Appendix |
| Backward Prediction | 超越专门训练的 CSDI **30.4%** | CSDI |
| Temporal Interpolation（50% missing） | 多数数据集最佳 | 各独立训练的基线 |
| Spatial Extrapolation（50% spatial masked） | 多数数据集最佳 | CSDI, ImputeFormer, Grin, BriTS |
| Spatio-Temporal Imputation（50% random mask） | 多数数据集最佳 | CSDI, ImputeFormer |

### 零样本与小样本

**零样本**：UrbanDiT 在未见过的目标数据集上直接推理，性能超越几乎所有有训练数据的基线模型[^src-urbandit]。

**小样本**（5%/1%）：在 5% 和 1% 训练数据下，UrbanDiT 持续超越基线[^src-urbandit]。

### 扩展性

UrbanDiT-L 在 data size 从 0.8 增至 1.0 时性能提升斜率（0.011）显著高于 M（0.0015）和 S（0.0019），表明大模型具有更强的扩展潜力[^src-urbandit]。

## 与现有模型的对比

| 维度 | GPD | UniST | UrbanGPT | CityGPT | **UrbanDiT** |
|------|-----|-------|----------|---------|-------------|
| 初始化方式 | Scratch | Scratch | LLMs | LLMs | Scratch |
| 数据类型 | Graph | Grid | Grid | Language | **Graph/Grid** |
| 多数据源 | ✗ | ✓ | ✓ | ✗ | **✓** |
| 任务灵活性 | ✗ | ✗ | ✗ | ✓ | **✓（5种任务）** |
| 零样本 | ✗ | ✓ | ✓ | ✗ | **✓** |

## 局限性

论文明确指出的局限[^src-urbandit]：当前聚焦于人类活动数据（出行、交通），尚未包含环境变量（空气污染、气候指标、微气候动态）。未来工作应整合这些环境维度以实现更全面的城市建模。

[^src-urbandit]: [[source-urbandit]]