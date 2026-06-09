---
title: "CSDI"
type: technique
tags:
  - diffusion-models
  - time-series
  - data-imputation
  - self-supervised-learning
  - probabilistic-modeling
  - neurips-2021
created: 2026-05-31
last_updated: 2026-06-09
source_count: 8
confidence: medium
status: active
---

# CSDI (Conditional Score-based Diffusion models for Imputation)

**CSDI** 是首个将条件扩散模型显式用于多元时间序列缺失值插补的方法，由 Tashiro, Song, Song & Ermon 发表于 NeurIPS 2021[^src-csdi]。核心创新在于：不是事后用无条件扩散模型去近似条件分布，而是在训练阶段就让去噪网络 $\epsilon_\theta$ 直接学习 $p(x^{\text{ta}} \mid x^{\text{co}})$——将[[ddpm|DDPM]] 的去噪函数从 $\epsilon_\theta(x_t, t)$ 扩展为条件形式 $\epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}})$[^src-csdi]。

## 问题形式化

给定多元时间序列 $X \in \mathbb{R}^{K \times L}$（$K$ 个特征，$L$ 个时间步）和观测掩码 $M \in \{0,1\}^{K \times L}$。令观测值 $x^{\text{co}} = M \odot X$，缺失值 $x^{\text{ta}} = (1-M) \odot X$。目标是估计缺失值的条件分布 $p(x^{\text{ta}} \mid x^{\text{co}})$[^src-csdi]。

## 核心机制

### 条件扩散模型

CSDI 将 DDPM 的反向过程参数化直接扩展到条件场景[^src-csdi]：

$$\mu_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}}) = \mu^{\text{DDPM}}(x_t^{\text{ta}}, t, \epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}}))$$

训练目标保持与 DDPM 相同的 $L_{\text{simple}}$ 形式：$\min_\theta \mathbb{E}[\| \epsilon - \epsilon_\theta(x_t^{\text{ta}}, t \mid x_0^{\text{co}}) \|^2]$[^src-csdi]。唯一的区别在于 $\epsilon_\theta$ 额外接收干净的观测值 $x_0^{\text{co}}$ 和条件掩码 $m^{\text{co}}$ 作为输入。推理时，从纯噪声 $x_T^{\text{ta}} \sim \mathcal{N}(0,I)$ 出发，$T=50$ 步逆扩散，每步去噪中 $\epsilon_\theta$ 都能"看见"观测值作为锚点，引导去噪过程向与已知值一致的分布收敛[^src-csdi]。

### 自监督训练策略

受 BERT 掩码语言建模启发，CSDI 在训练时从观测值中人工构造监督信号[^src-csdi]：从未缺失的部分中随机选取一部分作为"伪插补目标" $x_0^{\text{ta}}$（假装缺失），其余作为"伪条件观测" $x_0^{\text{co}}$（假装只看到这些）。在伪目标上按标准扩散流程加噪 $x_t^{\text{ta}} = \sqrt{\bar{\alpha}_t} x_0^{\text{ta}} + \sqrt{1-\bar{\alpha}_t} \epsilon$，训练网络去噪恢复。

四种目标选择策略[^src-csdi]：

| 策略 | 适用场景 | 方法 |
|------|---------|------|
| Random | 未知缺失模式 | 从观测值中随机选取 0%-100% 作为目标 |
| Historical | 结构性缺失（如传感器连续故障块） | 借用训练集中另一样本的缺失模式 |
| Mix | 部分结构化缺失（PhysioNet + 空气质量） | Random + Historical 混合 |
| Test pattern | 已知测试集缺失模式（如预测任务） | 直接使用测试集的缺失模式 |

### 双轴 Transformer 注意力架构

CSDI 将 DiffWave 残差层中的膨胀卷积替换为两个 1 层 Transformer 编码器[^src-csdi]：

- **时间 Transformer**：沿特征维度迭代，每个特征内长度为 $L$ 的序列独立过一层 self-attention——学习"这个特征内的时间模式"
- **特征 Transformer**：沿时间维度迭代，每个时间步上 $K$ 个特征的值独立过一层 self-attention——学习"这个时刻各特征间的跨通道相关性"

两层叠加可完整捕获 $K \times L$ 矩阵中的全部二阶依赖。输入通过零填充固定到完整形状 $K \times L$，配合条件掩码 $m^{\text{co}}$ 指示哪些位置是条件观测值[^src-csdi]。

**架构规格**[^src-csdi]：
- 骨架：DiffWave 风格 4 层残差层，残差通道 $C=64$
- 每层：时间 Transformer（1 层 TransformerEncoder，8 头注意力）+ 特征 Transformer（1 层 TransformerEncoder，8 头注意力）
- 侧信息：128 维扩散步嵌入（正弦编码）+ 128 维时间戳嵌入 + 16 维特征类别嵌入
- 输出掩码：在条件观测位置乘以 $(1 - m^{\text{co}})$ 屏蔽输出
- 参数量：约 415,000
- 优化器：Adam，初始 lr=0.001，在 75%/90% epoch 时衰减至 0.0001/0.00001，batch size=16，200 epochs
- 扩散参数：$T=50$，$\beta_1=0.0001$，$\beta_T=0.5$，二次方调度

### 推理

所有观测值作为条件 $x_0^{\text{co}}$，缺失位置从纯噪声出发 $T=50$ 步逆扩散。生成 100 个样本近似分布，取中位数作为确定性插补值[^src-csdi]。

## 性能

### 概率插补（CRPS）

| 数据集 | CSDI (50% 缺失) | GP-VAE | 提升 |
|--------|----------------|--------|------|
| PhysioNet 医疗 | 0.330 | 0.774 | -57% |
| 北京空气质量 | 0.108 | 0.397 | -73% |

CSDI 在所有缺失率下全面碾压所有概率插补基线（Multitask GP, GP-VAE, V-RIN），CRPS 改善幅度 40-65%[^src-csdi]。无条件扩散模型已优于 GP-VAE（PhysioNet 50%：0.458 vs 0.774），CSDI 进一步从 0.458 降至 0.330——条件建模贡献额外约 28% 改善[^src-csdi]。

### 确定性插补（MAE）

| 数据集 | CSDI | BRITS | GLIMA | 提升 |
|--------|------|-------|-------|------|
| PhysioNet 10% 缺失 | 0.217 | 0.284 | 0.265 | -18~24% |
| 北京空气质量 | 9.60 | 14.11 | — | -32% |

缺失率越低，CSDI 的相对优势越大（条件信息越多，双重注意力的信息提取能力越强）[^src-csdi]。

### 不规则采样插值

在相同 PhysioNet 数据的不规则采样测试中，CSDI (CRPS 0.380/0.418/0.556) 大幅领先 Latent ODE (0.700/0.676/0.761) 和 mTANs (0.526/0.567/0.689)[^src-csdi]。

### 概率预测

在 5 个预测基准上，CSDI 在 electricity 和 traffic 上超越 [[timegrad|TimeGrad]]，整体竞争力相当。优势不如插补任务显著——预测数据集几乎没有缺失值，RNN 方法在处理完整序列时没有劣势[^src-csdi]。

## 与相关方法的关系

### 继承自

- **[[ddpm|DDPM]]**：$\epsilon$ 预测参数化、$L_{\text{simple}}$ 简化目标、$\beta$ 调度——全部直接继承，区别仅在于添加条件输入 $x_0^{\text{co}}$[^src-csdi]
- **DiffWave**：残差层架构骨架（4 层、C=64、T=50），将膨胀卷积替换为双轴 Transformer 注意力[^src-csdi]
- **BERT 掩码语言建模**：自监督训练的核心灵感——"从已知中人造未知"[^src-csdi]

### 与同期方法对比

| 方法 | 任务 | 扩散形式 | 架构 |
|------|------|---------|------|
| [[timegrad|TimeGrad]] (ICML 2021) | 预测 | 条件扩散（以 RNN 隐状态为条件） | RNN + WaveNet 膨胀卷积 |
| **CSDI** (NeurIPS 2021) | **插补** | **条件扩散（以观测值为条件）** | **双轴 Transformer 注意力** |

TimeGrad 使用 RNN 处理历史序列，无法直接处理含缺失值的数据；CSDI 通过条件扩散机制"绕开"缺失值、直接从观测值中提取信息[^src-csdi][^src-timegrad]。

### 后续影响

CSDI 的三个核心设计——(1) 观测值作为条件直接注入去噪网络，(2) 自监督掩码训练，(3) 双轴注意力——成为后续扩散插补工作的标准范式[^src-csdi]：

- **SSSD** (Alcaraz & Strodthoff, 2023)：用 S4 状态空间模型替代 Transformer 注意力，解决 $O(L^2)$ 复杂度问题
- **[[ssd-ts|SSD-TS]]** (Gao et al., KDD 2025)：用 [[mamba|Mamba]] 选择性 SSM 替代 S4/Transformer 作为去噪 backbone，引入 [[bam|BAM]]（双向 Mamba + temporal attention，通道内）和 [[cmb|CMB]]（单向 Mamba 通道间）模块。在所有高缺失率场景下超越 CSDI 和 SSSD [^src-ssdts]
- **[[cofill|CoFILL]]** (2025)：将 CSDI 的单流架构扩展为时域+频域双流 Cross-Attention，添加图卷积空间建模
- **[[fence|FENCE]]** (AAAI 2026)：将 CSDI 的固定 CFG 引导尺度升级为动态反馈引导，解决高缺失率节点的漂移问题
- **[[lscd|LSCD]]** (ICML 2025)：将 CSDI 的条件扩散框架扩展为频谱条件化——用可微 [[lomb-scargle-periodogram|Lomb–Scargle 周期图]]替代 FFT，消除缺失值预处理带来的频谱失真，并引入 [[spectral-consistency-loss|频谱一致性损失]]强制频域对齐[^src-lscd]
- **[[sadi|SADI]]** (AAAI 2025)：将 CSDI 的双轴分离式 Transformer 替换为联合建模的 [[feature-dependency-encoder|FDE]]（时间感知特征依赖）+ [[gated-temporal-attention|GTA]]（自注意力时间依赖），并引入 [[partial-blackout|partial blackout]] 这一更通用的缺失模式。SADI 在所有 4 个数据集上超越 CSDI，证明了联合建模优于分离式建模[^src-sadi]
- **[[ratd|RATD]]** (NeurIPS 2024)：在 CSDI 的双轴 Transformer / DiffWave 架构基础上加入检索增强——从外部数据库检索 k 个最近邻历史样本作为参照，经 Reference Modulated Attention (RMA) 注入去噪过程，并改用 $x_0$-预测；在罕见/复杂任务（如 Wind、MIMIC 罕见病例）上显著超越 CSDI[^src-ratd]
- **[[s2dbm|S²DBM]]** (arXiv 2024)：直接沿用 CSDI 的去噪网络架构但移除其原有掩码条件机制，改用独立的线性先验预测器 F 与条件编码器 E，并把 CSDI 作为点/概率预测的主要扩散基线全面对比[^src-s2dbm]
- **CSDI 的 SDE 连续化**：一些后续工作将离散时间框架扩展到连续时间 SDE

## 局限性

1. **推理速度**：$T=50$ 步串行去噪，每次需跑完整 4 层残差 + 双 Transformer 网络，实时插补不可行
2. **长序列 $O(L^2)$ 成本**：双轴 Transformer 的 self-attention 复杂度为 $O(KL^2 + LK^2)$，长序列场景下计算膨胀——这是后续 SSSD 工作的直接动机[^src-csdi]
3. **假设观测值可靠**：传感器故障、离群点等错误观测值被当作条件直接喂给模型会污染插补结果
4. **非时序场景未验证**：声称框架不限于时间序列，但所有实验均在时序数据上进行[^src-csdi]
5. **预测任务优势不显著**：在没有缺失值的标准预测数据集上，CSDI 相比 RNN 方法的优势不明显[^src-csdi]

## 关联页面

- [[diffusion-model]] — 扩散模型概念总览
- [[ddpm]] — DDPM，CSDI 的扩散模型基础
- [[ssd-ts|SSD-TS]] — Mamba 作为扩散 backbone 的替代方案 (KDD 2025)
- [[bam|BAM]] — 双向注意力 Mamba，替代 CSDI 的 Time Transformer
- [[cmb|CMB]] — 通道 Mamba 块，替代 CSDI 的 Feature Transformer
- [[timegrad]] — TimeGrad，同期另一扩散+时序工作（预测方向）
- [[diffstg]] — DiffSTG，时空图扩散预测
- [[cofill]] — CoFILL，后续扩散时序插补（双流架构）
- [[fence]] — FENCE，动态反馈引导扩散插补，解决固定 CFG 尺度问题
- [[lscd]] — LSCD，频谱条件化扩散插补，CSDI 的后续发展
- [[lomb-scargle-periodogram]] — Lomb–Scargle 周期图，LSCD 的频谱条件信号源
- [[feedback-diffusion-guidance]] — 反馈扩散引导技术
- [[sadi]] — SADI，针对 partial blackout 设计的双阶段扩散插补
- [[feature-dependency-encoder]] — FDE，SADI 中替代 CSDI Feature Transformer 的特征依赖建模

[^src-csdi]: [[source-csdi]]
[^src-timegrad]: [[source-timegrad]]
[^src-fence]: [[source-fence]]
[^src-lscd]: [[source-lscd]]
[^src-ssdts]: [[source-ssdts]]
[^src-sadi]: [[source-sadi]]
[^src-ratd]: [[source-ratd]]
[^src-s2dbm]: [[source-s2dbm]]
