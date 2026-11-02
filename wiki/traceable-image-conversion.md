---
title: "Traceable Image Conversion（TIC）"
type: technique
tags:
  - time-series
  - anomaly-detection
  - time-series-to-image
  - periodicity
  - multimodal
created: 2026-11-02
last_updated: 2026-11-02
source_count: 1
confidence: low
status: active
---

# Traceable Image Conversion（TIC）

可追溯图像转换（Traceable Image Conversion, TIC）是 [[vetime|VETime]] 框架中将 1D 连续时间序列映射为 2D 高密度图像流形的拓扑转换技术（arXiv:2602.16681，§3.1）[^src-2602-16681-vetime]。

## 动机与设计背景

在时序异常检测中，将 1D 数值转换为 2D 图像有助于借助预训练视觉骨干捕获宏观周期模式，但现有转换方法存在显著瓶颈：
1. **传统数学变换的不可逆性**：Gramian Angular Fields（GAF）、Markov Transition Fields（MTF）与 Recurrence Plots（RP）属于不可逆变换（non-invertible），破坏了点级异常定位所需的时序相位信息，无法支持逆向精确重排（§2, 附录 D.2）[^src-2602-16681-vetime]。
2. **折线图渲染的信息稀疏性**：将时序绘制为折线图（Line plot）在画布中产生大量无信息的背景空白，导致信息密度极低；图像 patch 与时序原始时间戳之间的显式映射被稀释，弱化了时序对齐（weak temporal alignment），破坏点级时间戳定位精度，且依赖 VLM 提示推理导致推理计算和 token 消耗巨大（§1, §2）[^src-2602-16681-vetime]。
3. **TIC 的设计定位**：TIC 构建了一条无损、信息密集且显式可逆的 1D 到 2D 拓扑映射流水线，在浓缩时序多尺度模式的同时，为下游 [[patch-level-temporal-alignment|PTA]] 模块提供可逆追踪的时序拓扑结构与相位映射基础 [^src-2602-16681-vetime]。

## 三步流水线与维度变换

输入为单变量时间序列 $X \in \mathbb{R}^L$（极长序列在滑窗下默认长度 $L=5000$）。

$$\begin{aligned}
\{X_{\text{trend}}, X_{\text{rem}}\} &= \operatorname{Decompose}(X) && \text{// } [L] \xrightarrow{\text{移动平均分解}} [L], [L]\ \text{——提取宏观趋势 } X_{\text{trend}} \text{ 与高频残差 } X_{\text{rem}}\\
I_{\text{raw}} &= [\operatorname{Norm}(X); \operatorname{Norm}(X_{\text{trend}}); \operatorname{Norm}(X_{\text{rem}})] && \text{// } [L] \times 3 \xrightarrow{\text{MinMax 独立归一化堆叠}} [1, L, 3]\ \text{——分别对应 R、G、B 三通道}\\
T_{\text{fold}} &= \operatorname{Align16}(\operatorname{ACF}(X)) && \text{// } [L] \xrightarrow{\text{自相关估计周期并取 } 16 \text{ 的倍数}} \mathbb{Z}^+\ \text{——若无显著周期默认 } \sqrt{L}\\
I_{\text{fold}} &= \operatorname{Fold}(I_{\text{raw}}, T_{\text{fold}}) && \text{// } [1, L, 3] \xrightarrow{\text{尾部末周期均值填充折叠}} [T_{\text{fold}}, \lceil L/T_{\text{fold}} \rceil, 3]\ \text{——重塑为 2D 矩形网格}\\
I &= \operatorname{DecoupledScale}(I_{\text{fold}}) && \text{// } [T_{\text{fold}}, \lceil L/T_{\text{fold}} \rceil, 3] \xrightarrow[\text{周期垂直轴复制填充}]{\text{时间水平轴线性插值}} [224, 224, 3]\ \text{——解耦缩放至标准输入分辨率}
\end{aligned}$$

### 1. 多通道强度映射（Multi-Channel Intensity Mapping）
遵循 DLinear 风格采用移动平均进行季节-趋势分解（Season-Trend Decomposition）：
$$X = X_{\text{trend}} + X_{\text{rem}}$$
其中 $X_{\text{trend}}$ 表示低频宏观趋势分量，$X_{\text{rem}}$ 表示高频残差分量。三元组 $\{X, X_{\text{trend}}, X_{\text{rem}}\}$ 分别独立进行 Min-Max 归一化至 $[0, 255]$ 并分配至 RGB 通道：
$$\text{R 通道} \leftarrow X, \quad \text{G 通道} \leftarrow X_{\text{trend}}, \quad \text{B 通道} \leftarrow X_{\text{rem}}$$
输出尺寸为 $1 \times L \times 3$ 的多通道张量，显式解耦并保留全局趋势与高频残差信息，使视觉模型可直接在色彩空间感知低频趋势畸变与高频孤立突变（§3.1）[^src-2602-16681-vetime]。

### 2. 自适应折叠（Adaptive Folding）
利用自相关函数（ACF）动态估计序列的主周期 $T_{\text{fold}}$；对无显著周期的样本默认设为 $T_{\text{fold}} = \sqrt{L}$。为防止时序信息在 ViT patch 切割处发生相位断裂，算法强制将 $T_{\text{fold}}$ 调整为 ViT patch 大小（16）的整数倍。将 1D 序列重构为尺寸为 $(T_{\text{fold}}, \lceil L/T_{\text{fold}} \rceil, 3)$ 的二维网格；针对末尾无法整除片段，采用最后一个周期的均值（mean of the final period）做均值填充（mean-padding），降低零填充引发的分布偏移（§3.1）[^src-2602-16681-vetime]。

### 3. 解耦的维度感知缩放（Dimension-Aware Scaling）
将网格尺寸从 $(T_{\text{fold}}, \lceil L/T_{\text{fold}} \rceil, 3)$ 调整为预训练 ViT 标准输入分辨率 $(224, 224, 3)$：
- 水平时间轴（time axis）：采用线性插值（linear interpolation）保持波形连续性；
- 垂直周期轴（period axis）：采用复制填充策略（copy-padding），避免跨周期数值插值产生伪周期伪模式（§3.1）[^src-2602-16681-vetime]。

## 与 TimesNet 2D 折叠的机制对照

TIC 与 [[timesnet|TimesNet]] 均采用将 1D 序列折叠为 2D 张量的思想，但在建模目标、周期处理及通道表征上存在明显区别：
- **周期提取与拓扑重组**：TimesNet 面向全任务（预测、分类等）的多周期建模，采用快速傅里叶变换（FFT）选取振幅最大的前 $k$ 个频点周期，将序列并行转换为多个不同尺寸的 2D 张量，再通过 TimesBlock 中的 2D-Inception 卷积在模型内部同时捕获周期内（intra-period）与周期间（inter-period）变化；而 TIC 面向无监督零样本异常检测及对接预训练冻结 ViT，采用自相关函数（ACF）动态估计单一主周期，并严格约束折叠宽度为 ViT patch 大小（16）的整数倍，确保每个视觉补丁封装严格连续的 1D 时间戳 [^src-2602-16681-vetime]。
- **通道信息解耦与标准化**：TimesNet 在单通道或多变量原始数值上直接折叠；TIC 则引入季节-趋势分解构造 R（原始）、G（趋势）、B（残差）三通道流形，并引入「时间轴线性插值 + 周期轴复制填充」的解耦缩放策略，将任意长度序列标准化为固定 $(224, 224, 3)$ 图像，为下游 [[patch-level-temporal-alignment|PTA]] 模块的逆折叠重采样提供了完全可逆的双向索引 [^src-2602-16681-vetime]。

## 多变量时序扩展适配（附录 A）

在多变量时序场景下，TIC 包含以下配套扩展（附录 A）[^src-2602-16681-vetime]：
1. **通道独立（[[channel-independence|CI]]）**：每个变量独立进行季节-趋势分解，避免零样本场景下受到未见域复杂跨通道噪声的干扰；
2. **实例自适应 Gamma 校正**：各变量仅依据当前输入实例的统计量动态计算 Gamma 参数并进行非线性映射，严格防止目标域全局先验泄漏；
3. **最大公约数统一周期**：独立估计每个变量的周期后，取所有变量周期的最大公约数（GCD）作为全局统一折叠周期 $T_{\text{fold}}$；
4. **拼接与自适应池化**：各变量热图沿时间轴横向拼接为 $1 \times (N_v L) \times 3$ 复合热图。折叠后垂直维度超过 224 时，采用自适应池化（adaptive pooling）动态下采样至 224，替代单变量场景下的复制填充。

## 实证证据与消融结果

- **图像映射策略对比（附录 D.2, 表 14）**：在 11 个单变量数据集的 Contextual（上下文异常）下，TIC（Ours）取得 Affiliation-F1 80.54% / VUS-PR 28.35%，显著优于不可逆变换 GAF（77.01% / 23.57%）、RP（77.07% / 23.87%）、折线图 Line plot（78.20% / 24.92%）、单通道灰度映射 Gray mappings（79.50% / 26.65%）以及打乱时序的 Shuffled RGB mappings（78.94% / 25.51%）。
- **模块消融证据（附录 D.5, 表 17）**：移除 TIC（变体 V6 / V7）引发最严重的性能崩溃，WSD 数据集上的 VUS-PR 从完整模型的 33.85% 暴跌至 20.20%（V7），NEK 数据集上的 VUS-PR 从 57.54% 跌至 45.05%（V7），证实无损图像映射是后续时间轴对齐与多模态融合的基石。

## 失效模式与局限（附录 D.5）

1. **非周期信号下的负迁移失效**：在确定性混沌（MGAB）或随机游走金融数据（Stock）中，序列本身无固定周期。TIC 强制将其折叠为 2D 矩形网格，生成充满伪模式的随机噪点，导致变体 V1 在 MGAB 上的 Affiliation-F1 从 Base 的 70.69% 跌至 67.51%，且后续融合模块无法挽回 [^src-2602-16681-vetime]。
2. **周期性尖峰异常引发的模式混淆**：在 Power 数据集中，正常背景无尖峰，异常表现为周期性出现的尖峰脉冲。TIC 的周期折叠机制误将此类周期性脉冲编码为正常的背景纹理；时序分支正确识别了该异常（V3 达到 21.48% VUS-PR），但零样本路由器（变体 V4）错误赋予受骗的视觉表征较高权重，导致严重负迁移（VUS-PR 降至 11.54%，显著低于纯时序占优变体 V3 的 21.48% 与 Base 的 21.25%）[^src-2602-16681-vetime]。

[^src-2602-16681-vetime]: [[source-2602-16681-vetime]]
