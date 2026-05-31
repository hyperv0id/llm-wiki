---
title: "xCPD"
type: entity
tags:
  - time-series
  - forecasting
  - graph-spectral
  - channel-dependency
  - plugin
  - mixture-of-experts
  - ICLR-2026
created: 2026-05-31
last_updated: 2026-05-31
source_count: 1
confidence: medium
status: active
---

# xCPD

xCPD 是一个通用的、模型无关的轻量级 plugin，由 Li et al. 提出（ICLR 2026），通过**图频谱分解**（graph spectral decomposition）在 patch 级别自适应路由通道间依赖。"x" 指频谱分解，"CPD" 指 Channel-Patch Dependencies [^src-xcpd]。

## 核心思想

多元时间序列预测中，[[channel-independence|Channel Independence (CI)]] 策略逐通道建模，避免跨通道噪声但丢失有价值的相关性；Channel Dependence (CD) 策略联合建模所有通道，但可能引入无关信息导致过平滑 [^src-xcpd]。现有的 Channel Partiality (CP) 方法在粗粒度的通道级别操作（聚类或通道级注意力），无法捕获 patch 级局部交互和频率解耦的依赖关系 [^src-xcpd]。

xCPD 将建模单元从整个通道下沉到 **channel-patch**——即将每个变量（channel）的局部时间段（patch）视为图中的一个节点，在**频谱域**而非时间域中显式建模三类频率依赖 [^src-xcpd]：

| 频段 | 物理含义 | 滤波器建模目标 |
|------|---------|-------------|
| 低频 | 平滑趋势、季节性周期 | 长程规律 |
| 中频 | 局部波动、半平稳过程 | 交通模式转换、地区天气迁移 |
| 高频 | 快速变化、突变 | 系统异常、市场冲击 |

## 三个核心组件

### 1. 频谱 Channel-Patch 嵌入
- 将 backbone 预测输出切分为 patch，每个 (channel, patch) 对 → 一个图节点 [^src-xcpd]
- 通过共享图傅里叶基 (shared graph Fourier basis) 将节点嵌入统一投影到频谱域，确保跨 batch 一致性 [^src-xcpd]
- 共享基学习由 Davis-Kahan 定理保证误差上界 [^src-xcpd]

### 2. 频谱节点分组
- 使用可学习频率边界 $\tau_1, \tau_2$ 自动划分低频/中频/高频三个波段 [^src-xcpd]
- 通过频谱能量响应量化每个节点对不同频率的响应强度 [^src-xcpd]
- 在每个节点的 ego-graph 内构建频率感知子图（同频节点相连）[^src-xcpd]

### 3. 频谱 Channel-Patch 路由 (DyMoE)
- 三个频率专属滤波器（低/中/高）构建强调不同频谱成分的邻接矩阵 [^src-xcpd]
- **动态 MoE (DyMoE)**：根据累计概率阈值自适应选择 1-3 个 expert（不等同于传统 Top-K 固定选择），路由分数由确定性分量 + 可学习的噪声分量组成 [^src-xcpd]
- GNN 消息传递 + 门控双路径残差校正（GNN 路径捕获跨变量频谱依赖，Linear 路径保留 CI 细化）[^src-xcpd]

## 关键特性

- **模型无关**：可作为 plugin 集成到 CI（DLinear, PatchTST）和 CD（TSMixer, TimesNet）backbone，包括基础预测模型（TimesFM, Chronos），无需重训练 backbone [^src-xcpd]
- **效率优越**：相比 CCM（迭代聚类，增加 60-100% 显存、4-5× 训练时间），xCPD 显存恒定 ~7GB、每 iter <10ms [^src-xcpd]
- **零样本迁移**：学习通用频谱表示而非数据集特定变换，支持跨域零样本部署 [^src-xcpd]
- **非平稳鲁棒**：共享傅里叶基 + 动态 MoE 路由的双重机制下，对时间偏移和幅度扰动仍保持性能 [^src-xcpd]
- **骨干频谱域依赖建模**：xCPD 与现有 CP 方法的三个关键维度区分——粒度（channel-patch vs channel）、建模域（频谱 vs 时间）、自适应能力（频率专属路由 vs 混合注意力）[^src-xcpd]

## 性能亮点

- 长期预测：9 数据集、4 backbone、144 设置中一致提升。高变量数据集（Electricity 321 维 ↓4-7% MSE, Traffic 862 维 ↓4-7% MSE）增益最大，规律性强的数据集（ETT, Solar ↓1-3%）增益适中 [^src-xcpd]
- xCPD 在附加 backbone 上也一致提升：iTransformer 平均 ↓3.7% MSE, TimeMixer ↓2.8%, DUET (CP backbone) 也获提升 [^src-xcpd]
- 零样本迁移 48 设置全提升，CI 模型获益更大（DLinear 12.0%, PatchTST 15.2%）[^src-xcpd]
- 消融：移除共享傅里叶基性能下降最大（非平稳设置下 +8.3% MSE），三波段分解本身提供强归纳偏置 [^src-xcpd]

## 与其他方法对比

| 方法 | 建模粒度 | 建模域 | 自适应路由 | 模型无关性 |
|------|---------|-------|-----------|-----------|
| [[crossformer|Crossformer]] | patch-to-patch (2D) | 时间 | 两阶段注意力 | Transformer only |
| CCM | 通道聚类 | 频率（原型） | 聚类分配 | 有限 |
| DUET | 通道聚类 | 时间 | 聚类分配 | 有限 |
| TimeFilter | patch | 时间 | 时空滤波 | 有限 |
| PCD | patch (仅 Transformer) | 时间 | 通道注意力 | Transformer only |
| **xCPD** | **channel-patch** | **频谱 (频谱分解)** | **DyMoE (1-3 experts)** | **全模型无关** |

## 相关页面

- [[source-xcpd]] — 源文件摘要
- [[channel-independence]] — CI 策略
- [[cross-dimension-dependency]] — 跨维度依赖
- [[patch-based-tokenization]] — patch 化处理
- [[mixture-of-experts]] — MoE 架构
- [[patchtst]] — CI + patching 的开创模型
- [[crossformer]] — 首个全 CD Transformer
- [[itransformer]] — 反转注意力范式
- [[timesnet]] — CNN backbone
- [[timemixer]] — MLP-mixing backbone

[^src-xcpd]: [[source-xcpd]]
