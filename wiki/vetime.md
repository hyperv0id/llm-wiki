---
title: "VETime"
type: entity
tags:
  - time-series
  - anomaly-detection
  - zero-shot
  - multimodal
  - vision-transformer
  - foundation-model
created: 2026-11-02
last_updated: 2026-11-02
source_count: 1
confidence: low
status: active
---

# VETime

VETime（Vision Enhanced Zero-Shot Time Series Anomaly Detection）是由清华大学、华为 2012 实验室与 Datadog AI Research 提出的零样本时序异常检测模型（arXiv:2602.16681，NeurIPS 2026 投稿格式）[^src-2602-16681-vetime]。针对 1D 数值模型因滑动窗口感受野限制难以识别长程结构违背（上下文异常），以及 2D 视觉与 VLM 模型因非可逆映射或粗粒度窗口导致时间索引丢失与高昂推理延迟的问题，VETime 建立了 1D 数值模态与 2D 视觉流形特征级的有序对齐与动态融合机制 [^src-2602-16681-vetime]。

在时序多模态领域坐标上，现有面向预测的多模态模型（如 [[time-vlm|Time-VLM]] 与 [[aurora|Aurora]]）关注宏观演化趋势，缺乏异常检测所需的逐点密集定位敏感度与预测残差零样本泛化能力；VETime 确立了特征级密集对齐驱动的 [[multimodal-time-series-anomaly-detection|多模态时序异常检测]] 新路线 [^src-2602-16681-vetime]。

在体系结构上，VETime 采用双流骨干架构：视觉分支采用冻结的预训练 [[mae|MAE]] ViT-Base 编码器（85.64M 参数）；时序分支采用基于预训练 TimeRCD 初始化的 8 层 Transformer 编码器（8 个注意力头，通过秩 $r=8$、$\alpha=16$ 的 LoRA 微调适配）[^src-2602-16681-vetime]。

## 数据流与维度

> **维度符号**：$L$ 为输入时序滑动窗口物理长度（极长序列默认滑窗 $L=5000$）；$P$ 为 1D 时序 patch 大小与 2D ViT patch 空间边长（$P=16$）；$N_V$ 为图像视觉 patch 总数（$N_V = (224/16) \times (224/16) = 196$）；$N_{TS}$ 为 1D 时序 patch 总数（$N_{TS} = L/P = 5000/16 = 313$）；$D_V$ 为视觉编码器特征维度（$D_V = 768$）；$D_{TS}$ 为时序编码器特征维度（$D_{TS} = 512$）；$D$ 为统一映射后的共享特征维度（$D = 512$）。张量形状记作 $[\text{行}, \text{列}]$ 或几何维度。

### Stage 1 · TIC 图像转换与时序分块

输入为单变量时间序列 $X \in \mathbb{R}^L$。

- **支路 A：时序分块预处理**
$$\begin{aligned}
X_{\text{norm}} &= \operatorname{InstanceNorm}(X) && \text{// } [L] \xrightarrow{\text{实例归一化}} [L]\ \text{——消除量纲差异与基准漂移}\\
X_{\text{patch}} &= \operatorname{Reshape}(X_{\text{norm}}) && \text{// } [L] \xrightarrow{\text{按步长 } P=16 \text{ 切分}} [N_{TS}, P]\ \text{——划分为 } N_{TS}=313 \text{ 个局部 patch}
\end{aligned}$$

- **支路 B：TIC 图像生成（[[traceable-image-conversion|TIC]]）**
$$\begin{aligned}
\{X, X_{\text{trend}}, X_{\text{rem}}\} &= \operatorname{Decompose}(X) && \text{// } [L] \xrightarrow{\text{移动平均分解}} [L] \times 3\ \text{——分解为低频趋势与高频残差}\\
I_{\text{raw}} &= \operatorname{Stack}(\operatorname{MinMaxNorm}(X, X_{\text{trend}}, X_{\text{rem}})) && \text{// } [L] \times 3 \xrightarrow{\text{独立归一化至 } [0, 255]} [1, L, 3]\ \text{——映射至 RGB 三通道}\\
I_{\text{fold}} &= \operatorname{Fold}(I_{\text{raw}}, T_{\text{fold}}) && \text{// } [1, L, 3] \xrightarrow{\text{主周期折叠与均值填充}} [T_{\text{fold}}, \lceil L/T_{\text{fold}} \rceil, 3]\ \text{——折叠为 2D 矩形网格}\\
I &= \operatorname{DecoupledScale}(I_{\text{fold}}) && \text{// } [T_{\text{fold}}, \lceil L/T_{\text{fold}} \rceil, 3] \xrightarrow[\text{周期轴复制填充}]{\text{时间轴线性插值}} [224, 224, 3]\ \text{——标准化输入图像}
\end{aligned}$$

### Stage 2 · 双分支特征编码

- **支路 A：冻结视觉分支编码**
$$\begin{aligned}
F_V^0 &= \operatorname{ViT}_{\text{frozen}}(I) && \text{// } [224, 224, 3] \xrightarrow{\text{MAE ViT-Base 编码}} [N_V, D_V]\ \text{——抽取 } N_V=196 \text{ 个视觉 patch 特征}
\end{aligned}$$

- **支路 B：可微调时序分支编码**
$$\begin{aligned}
F_{TS} &= \operatorname{TransformerEncoder}(X_{\text{patch}}) && \text{// } [N_{TS}, P] \xrightarrow{\text{8层 Transformer 编码}} [N_{TS}, D_{TS}]\ \text{——抽取 } N_{TS}=313 \text{ 个时序特征}
\end{aligned}$$

### Stage 3 · PTA 视觉特征时间轴对齐（[[patch-level-temporal-alignment|PTA]]）

$$\begin{aligned}
F_V^{\text{grid}} &= \operatorname{Reshape}(F_V^0) && \text{// } [N_V, D_V] \xrightarrow{\text{恢复空间几何网格}} [14, 14, D_V]\ \text{——还原为 } 14 \times 14 \text{ 空间拓扑}\\
F_V^{\text{time}} &= \operatorname{Interp1D}(F_V^{\text{grid}}, N_{TS}) && \text{// } [14, 14, D_V] \xrightarrow{\text{沿时间水平轴线性插值}} [N_{TS}, 14, D_V]\ \text{——时间 patch 数上采样至 } N_{TS}=313\\
\hat{F}_V &= \operatorname{AvgPool2D}(F_V^{\text{time}}) && \text{// } [N_{TS}, 14, D_V] \xrightarrow{\text{沿周期垂直轴平均池化}} [N_{TS}, D_V]\ \text{——规约跨周期冗余并展平}\\
\tilde{F}_V &= \hat{F}_V + E_{\text{pos}} && \text{// } [N_{TS}, D_V] + [N_{TS}, D_V] \to [N_{TS}, D_V]\ \text{——注入绝对位置编码恢复序列因果依赖}\\
F_V &= \operatorname{SelfAttn}(\tilde{F}_V W_P) && \text{// } [N_{TS}, \bcancel{D_V}] \times [\bcancel{D_V}, D] \xrightarrow{\text{投影与自注意力建模}} [N_{TS}, D]\ \text{——映射至统一共享维度 } D=512
\end{aligned}$$

### Stage 4 · AWCL 交叉注意力与异常增强（[[anomaly-window-contrastive-learning|AWCL]]）

$$\begin{aligned}
A_{TS\to V} &= \operatorname{Softmax}\left(\frac{(F_{TS} W_Q) (F_V W_K)^\top}{\sqrt{d}}\right) && \text{// } [N_{TS}, \bcancel{d}] \times [\bcancel{d}, N_{TS}] \to [N_{TS}, N_{TS}]\ \text{——时序查询视觉注意力矩阵}\\
Z_{TS}^0 &= F_{TS} + A_{TS\to V} (F_V W_V) && \text{// } [N_{TS}, D] + [N_{TS}, \bcancel{N_{TS}}] \times [\bcancel{N_{TS}}, D] \to [N_{TS}, D]\ \text{——跨模态视觉上下文注入}\\
A_{V\to TS} &= \operatorname{Softmax}\left(\frac{(F_V W_Q) (F_{TS} W_K)^\top}{\sqrt{d}}\right) && \text{// } [N_{TS}, \bcancel{d}] \times [\bcancel{d}, N_{TS}] \to [N_{TS}, N_{TS}]\ \text{——视觉查询时序注意力矩阵}\\
Z_V^0 &= F_V + A_{V\to TS} (F_{TS} W_V) && \text{// } [N_{TS}, D] + [N_{TS}, \bcancel{N_{TS}}] \times [\bcancel{N_{TS}}, D] \to [N_{TS}, D]\ \text{——跨模态时序细节注入}\\
Z_{TS} &= Z_{TS}^0 + \operatorname{FFN}(Z_{TS}^0) && \text{// } [N_{TS}, D] \xrightarrow{\text{前馈残差更新}} [N_{TS}, D]\ \text{——更新后时序特征}\\
Z_V &= Z_V^0 + \operatorname{FFN}(Z_V^0) && \text{// } [N_{TS}, D] \xrightarrow{\text{前馈残差更新}} [N_{TS}, D]\ \text{——更新后视觉特征}\\
F_A &= \operatorname{FFN}([Z_{TS}; Z_V]) && \text{// } [N_{TS}, 2\bcancel{D}] \xrightarrow{\text{拼接并经 FFN 投影}} [N_{TS}, D]\ \text{——输出多模态异常增强表征}
\end{aligned}$$

### Stage 5 · TMF 动态权重路由与模态融合（[[task-adaptive-multimodal-fusion|TMF]]）

定义候选专家空间 $\mathcal{M} = \{TS, V, A\}$，对应特征矩阵为 $\{F_{TS}, F_V, F_A\} \in \mathbb{R}^{N_{TS} \times D}$。

$$\begin{aligned}
W^{AD} &= \operatorname{Softmax}(\operatorname{MLP}(F_A) + b^{AD}) && \text{// } [N_{TS}, \bcancel{D}] \xrightarrow{\text{MLP 映射与偏置}} [N_{TS}, 3]\ \text{——异常检测 3 专家动态权重}\\
W^{Rec} &= \operatorname{Softmax}(\operatorname{MLP}(F_A) + b^{Rec}) && \text{// } [N_{TS}, \bcancel{D}] \xrightarrow{\text{MLP 映射与偏置}} [N_{TS}, 3]\ \text{——序列重构 3 专家动态权重}\\
F_{AD} &= \sum_{m \in \mathcal{M}} W_m^{AD} \odot F_m && \text{// } \sum_{m=1}^3 [N_{TS}, 1] \odot [N_{TS}, D] \to [N_{TS}, D]\ \text{——检测特化加权融合特征}\\
F_{Rec} &= \sum_{m \in \mathcal{M}} W_m^{Rec} \odot F_m && \text{// } \sum_{m=1}^3 [N_{TS}, 1] \odot [N_{TS}, D] \to [N_{TS}, D]\ \text{——重构特化加权融合特征}
\end{aligned}$$

### Stage 6 · 双头预测与输出

$$\begin{aligned}
\hat{y} &= \operatorname{Sigmoid}(\operatorname{MLP}_{AD}(\operatorname{TokenProj}(F_{AD}))) && \text{// } [N_{TS}, D] \xrightarrow{\text{Token 线性投影与 MLP 分类}} (0, 1)^L\ \text{——逐时间戳异常概率}\\
\hat{x} &= \operatorname{MLP}_{Rec}(\operatorname{TokenProj}(F_{Rec})) && \text{// } [N_{TS}, D] \xrightarrow{\text{Token 线性投影与 MLP 重构}} \mathbb{R}^L\ \text{——辅助序列数值重构输出}
\end{aligned}$$

## 优化目标与推理流程

### 联合优化目标

模型在 0.5B 数学方程合成数据上进行端到端多任务预训练，总损失函数为（公式 6 / 9）[^src-2602-16681-vetime]：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(y, \hat{y}) + \mathcal{L}_{\text{MSE}}(x, \hat{x}) + \lambda_{\text{aw}} \mathcal{L}_{\text{aw}} + \lambda_e \mathcal{L}_e$$
其中：
- $\mathcal{L}_{\text{BCE}} = -\frac{1}{L} \sum_{t=1}^L [y_t \log \hat{y}_t + (1 - y_t) \log (1 - \hat{y}_t)]$ 为逐点二元交叉熵损失；
- $\mathcal{L}_{\text{MSE}} = \frac{1}{L} \sum_{t=1}^L \|x_t - \hat{x}_t\|_2^2$ 为序列数值重构均方误差损失；
- $\mathcal{L}_{\text{aw}}$ 为异常窗口对比学习损失，权重设为 $\lambda_{\text{aw}} = 0.1$；
- $\mathcal{L}_e$ 为路由负载均衡负熵正则化损失，权重设为 $\lambda_e = 0.2$。

### 零样本推理协议

1. **逐点异常打分**：测试阶段仅需单次前向传播，通过异常检测特征 $F_{AD}$ 经分类头输出逐点异常概率 $S_t = \hat{y}_t \in (0, 1)$，不计算亦不依赖重构误差，无需构建对比窗口 [^src-2602-16681-vetime]。
2. **无监督动态阈值**：采用无监督极值理论峰值超阈值法（POT），分位数风险参数固定为 $q=10^{-4}$，根据当前测试序列的分数分布自适应计算判定阈值 $\delta$，完全脱离真实标签与目标域调参 [^src-2602-16681-vetime]。
3. **极长序列推理**：对于极长序列，采用固定长度为 5000 的滑动窗口分段推理，保证推理显存与分辨率可扩展性 [^src-2602-16681-vetime]。

## 多变量时序扩展适配（附录 A）

论文在附录 A 说明了面向多变量时序场景的适配机制（注明只在附录给出）：
1. **通道独立（[[channel-independence|CI]]）**：对多变量输入按通道独立处理，各变量单独进行季节-趋势分解，避免零样本场景下受到未见域复杂跨通道噪声的干扰。
2. **实例自适应 Gamma 校正**：每个变量仅依据当前输入样本自身的统计分布动态计算 Gamma 参数并进行非线性映射，禁止跨样本或泄漏目标域全局先验。
3. **公约数统一折叠周期**：独立估计每个变量的主周期，取所有变量周期的最大公约数（GCD）作为全局统一折叠周期 $T_{\text{fold}}$。
4. **横向拼接与自适应池化**：各变量热图沿时间轴横向拼接为 $1 \times (N_v L) \times 3$。折叠后若垂直维度超过 224，采用自适应池化（Adaptive Pooling）动态下采样至 224，替代单变量场景下的复制填充。
5. **多变量 PTA 与 AWCL 适配**：对齐特征形式为 $\hat{F}_V \in \mathbb{R}^{N_{TS} \times N_v \times D_V}$，注入相同位置编码 $E_{\text{pos}}$ 后，利用 Any-variate Attention 独立处理各变量的时序动态矫正，重组为 $F_V \in \mathbb{R}^{(N_{TS} N_v) \times D_{TS}}$；AWCL 针对各变量独立构建窗口与对比计算。TMF 模块保持原结构不变。

## 实验证据

### 1. 零样本对比时序基础模型（表 1）

在 TSB-AD 基准的 26 个数据集评测中，VETime 在 104 项指标实例中取得 58 项第一，表现如下（表 1）[^src-2602-16681-vetime]：

| 评测场景 | 对比总项数 | VETime 第 1 项数 | VETime 第 2 项数 | VETime 平均排名 | 次优基线与均排 | 出处 |
|---|---|---|---|---|---|---|
| 单变量（11 个数据集） | 44 | 24 | 14 | **2.05** | TimeRCD (3.27), Chronos (3.27) | 表 1 |
| 多变量（15 个数据集） | 60 | 34 | 20 | **1.58** | DADA (2.40), TimeRCD (2.60) | 表 1 |

代表性单变量数据集具体数值对比（表 1）[^src-2602-16681-vetime]：

| 数据集 | 指标 | VETime | 最强对照基线 | 提升幅度/出处 |
|---|---|---|---|---|
| IOPS | Affiliation-F1 | **90.53%** | Chronos: 90.12% | 表 1 |
| IOPS | Standard-F1 | **34.56%** | TimesFM: 34.28% | 表 1 |
| IOPS | VUS-PR | **30.79%** | TimeRCD: 20.23% | 表 1 |
| SED | F1_T | **74.35%** | TimeRCD: 69.43% | 表 1 |
| SED | Standard-F1 | **75.02%** | TimeRCD: 69.88% | 表 1 |
| SED | VUS-PR | **89.05%** | TimeRCD: 80.75% | 表 1 |
| TODS | F1_T | **68.33%** | TimeRCD: 65.89% | 表 1 |
| TODS | Standard-F1 | **69.13%** | TimeRCD: 67.02% | 表 1 |
| TODS | VUS-PR | **94.20%** | TimeRCD: 93.46% | 表 1 |
| YAHOO | Affiliation-F1 | **97.15%** | TimeRCD: 96.65% | 表 1 |
| YAHOO | F1_T | **91.54%** | TimeRCD: 85.86% | 表 1 |
| YAHOO | Standard-F1 | **88.93%** | TimeRCD: 87.02% | 表 1 |
| YAHOO | VUS-PR | **88.61%** | TimesFM: 86.78% | 表 1 |
| WSD | Affiliation-F1 | **94.31%** | TimeRCD: 90.33% | 表 1 |
| WSD | F1_T | **50.42%** | MOMENT: 41.98% | 表 1 |
| WSD | Standard-F1 | **46.55%** | DADA: 45.06% | 表 1 |
| WSD | VUS-PR | **33.85%** | DADA: 33.42% / TimeRCD: 21.77% | 表 1 |
| NAB | Affiliation-F1 | **88.57%** | Chronos: 86.66% | 表 1 |
| NAB | F1_T | **44.31%** | TimeRCD: 38.85% | 表 1 |
| NAB | Standard-F1 | **37.87%** | TimeRCD: 27.70% | 表 1 |
| NAB | VUS-PR | **32.98%** | TimeRCD: 24.32% | 表 1 |

多变量数据集典型指标表现（表 1, 表 8）[^src-2602-16681-vetime]：
- Exathlon：Affiliation-F1 **99.90%**（vs DADA 93.72%），F1_T **97.45%**（vs MOMENT 96.00% / DADA 74.65%），Standard-F1 **97.43%**（vs DADA 73.11%），VUS-PR **99.77%**（vs MOMENT 96.19% / DADA 73.94%）；
- SWaT：Standard-F1 **55.87%**（vs DADA 34.78% / TimeRCD 28.73%），VUS-PR **42.93%**（vs DADA 21.13% / TimeRCD 17.58%）。

### 2. 相对视觉与 VLM 基线对比及运行时间（表 2, 附录表 11）

在 4 个单变量数据集上遵循 VLM4TS 协议对比，VETime 各项指标均优于对比基线且耗时减少约 100 倍（表 2、表 11）[^src-2602-16681-vetime]：

| 数据集 | 指标 | VETime | VLM4TS | ViT4TS | VisualTimeAnomaly | AnomLLM | 出处 |
|---|---|---|---|---|---|---|---|
| NAB | Affiliation-F1 | **88.57%** | 70.37% | 61.12% | 62.46% | 58.53% | 表 2 |
| NAB | F1_T | **44.31%** | 30.75% | 28.29% | 41.64% | 17.76% | 表 2 |
| NAB | Standard-F1 | **37.87%** | 34.78% | 33.81% | 31.72% | 23.70% | 表 2 |
| NAB | VUS-PR | **32.98%** | 27.89% | 25.78% | 26.87% | 19.50% | 表 2 |
| NAB | 单序列耗时 | **0.04 s (36.54 ms)** | 6.78 s | 1.72 s | 2.52 s | 3.59 s | 表 2, 表 11 |
| YAHOO | Affiliation-F1 | **97.15%** | 64.25% | 60.66% | 61.03% | 36.43% | 表 2 |
| YAHOO | F1_T | **91.54%** | 9.10% | 7.61% | 3.02% | 1.67% | 表 2 |
| YAHOO | Standard-F1 | **88.93%** | 9.16% | 7.74% | 3.63% | 2.36% | 表 2 |
| YAHOO | VUS-PR | **88.61%** | 33.09% | 31.63% | 13.13% | 10.44% | 表 2 |
| YAHOO | 单序列耗时 | **0.02 s (24.64 ms)** | 9.27 s | 5.86 s | 2.35 s | 3.91 s | 表 2, 表 11 |
| SMAP | Affiliation-F1 | **98.83%** | 94.76% | 91.55% | 91.65% | 42.52% | 表 2 |
| SMAP | F1_T | **70.21%** | 57.13% | 50.97% | 14.80% | 4.17% | 表 2 |
| SMAP | Standard-F1 | **64.18%** | 57.56% | 51.65% | 16.19% | 6.65% | 表 2 |
| SMAP | VUS-PR | **60.67%** | 52.28% | 46.44% | 11.94% | 4.92% | 表 2 |
| SMAP | 单序列耗时 | **0.04 s (47.06 ms)** | 7.00 s | 2.23 s | 2.34 s | 3.40 s | 表 2, 表 11 |
| MSL | Affiliation-F1 | **92.78%** | 66.15% | 65.80% | 86.20% | 46.08% | 表 2 |
| MSL | F1_T | **50.38%** | 37.50% | 23.48% | 13.84% | 3.32% | 表 2 |
| MSL | Standard-F1 | **52.18%** | 42.21% | 27.05% | 16.96% | 10.70% | 表 2 |
| MSL | VUS-PR | **42.81%** | 33.57% | 26.52% | 17.91% | 9.77% | 表 2 |
| MSL | 单序列耗时 | **0.02 s (74.11 ms)** | 7.39 s | 2.75 s | 2.85 s | 3.95 s | 表 2, 表 11 |

### 3. 细分异常类型检测对比（表 3）

在 NAB 与 YAHOO 拆分出的异常子集上对比（表 3）[^src-2602-16681-vetime]：

| 异常形态 | 评测模型 | Affiliation-F1 | F1_T | Standard-F1 | VUS-PR | 出处 |
|---|---|---|---|---|---|---|
| **Point（点异常）** | **VETime** | **93.20%** | **66.17%** | **62.33%** | 59.33% | 表 3 |
| Point | TimeRCD | 74.84% | 31.51% | 28.34% | **61.85%** | 表 3 |
| Point | VLM4TS | 67.11% | 21.60% | 23.15% | 30.57% | 表 3 |
| **Context（上下文异常）** | **VETime** | **91.79%** | **63.32%** | **59.74%** | **57.59%** | 表 3 |
| Context | TimeRCD | 73.84% | 54.06% | 32.18% | 44.93% | 表 3 |
| Context | VLM4TS | 61.94% | 35.90% | 35.90% | 32.10% | 表 3 |
| **Mixed（混合异常）** | **VETime** | **87.78%** | **52.61%** | **44.28%** | **44.61%** | 表 3 |
| Mixed | TimeRCD | 73.03% | 42.12% | 26.20% | 42.90% | 表 3 |
| Mixed | VLM4TS | 60.79% | 29.63% | 34.52% | 25.61% | 表 3 |

### 4. 对比全样本目标域训练模型（表 7, 表 8）

- **单变量（表 7）**：VETime 以严格零样本设定取得 23 项第一、7 项第二，平均排名 2.02，优于在目标域全监督/无监督训练的全样本模型（TranAD 3.45–4.18, USAD 2.73–4.36, OmniAnomaly 2.64–3.36, LOF 3.09–4.18, IForest 5.00–5.09）。
- **多变量（表 8）**：VETime 在 60 项对比中取得 30 项第一、12 项第二，显著优于 TranAD、USAD 与 OmniAnomaly。

### 5. 模型计算开销与双分支对照（表 12）

| 模型配置 | 参数量 | 显存占用 | 平均单序列耗时 | NAB (Aff-F1 / VUS-PR) | WSD (Aff-F1 / VUS-PR) | YAHOO (Aff-F1 / VUS-PR) | 出处 |
|---|---|---|---|---|---|---|---|
| Base（纯时序编码器+分类头） | 37.06M | 223.73 MB | 24.49 ms | 82.48% / 24.32% | 90.33% / 21.77% | 96.65% / 84.41% | 表 12 |
| Dual-TS（时序双分支替代视觉） | 160.49M | 928.52 MB | 48.19 ms | 80.84% / 30.28% | 88.87% / 27.24% | 96.81% / 87.25% | 表 12 |
| VETime（时序+视觉双模态） | 157.96M | 787.41 MB | 51.02 ms | **88.57% / 32.98%** | **94.31% / 33.85%** | **97.15% / 88.61%** | 表 12 |

在参数量与显存占用相近的条件下，VETime 显著超越 Dual-TS，证实性能收益源自时空双模态设计而非单纯参数量增加 [^src-2602-16681-vetime]。

## 边界与局限

论文在第 5 节及附录 C.3、D.5 自述了以下边界与局限：
1. **高频噪声敏感性**：依赖季节-趋势分解算法。当真实时间序列存在不规则高频剧烈波动且结构与噪声紧密缠绕时，分解模块易将噪声误划分为趋势或残差分量，导致视觉 RGB 映射失真 [^src-2602-16681-vetime]。
2. **高维多变量物理画布瓶颈**：通道独立假设忽略了变量间复杂的相互依赖关系；在固定 $224 \times 224$ 画布与 5000 长度滑窗下，每个数据点占至少一个像素的物理上限约为 10 个变量（$224 \times 224 / 5000 \approx 10$）。超过此限制会引发像素重叠与混叠（aliasing），导致中高维（25–38 变量）性能回落；但在高相关性冗余数据集（如 59 变量的 SMAP）上因信息重合度高未出现回落（§5, 附录 C.3）[^src-2602-16681-vetime]。
3. **非周期信号下的负迁移失效**：在确定性混沌（MGAB）或随机游走金融数据（Stock）中，序列本身无固定周期。TIC 强制折叠产生伪模式噪点，导致变体 V1 在 MGAB 上的 Affiliation-F1 从 Base 的 70.69% 跌至 67.51%，且后续融合模块无法挽回（附录 D.5）[^src-2602-16681-vetime]。
4. **周期性尖峰异常引发的路由受骗**：在 Power 数据集中，正常背景无尖峰，异常表现为周期性出现的尖峰脉冲。TIC 周期折叠误将此类周期性脉冲编码为正常的背景纹理；时序分支虽正确识别了该异常（V3 达到 21.48% VUS-PR），但零样本路由器（V4）错误赋予受骗的视觉表征较高权重，导致严重负迁移（VUS-PR 降至 11.54%，显著低于纯时序占优变体 V3 的 21.48% 与 Base 的 21.25%）（附录 D.5）[^src-2602-16681-vetime]。
5. **计算开销与理论可解释性缺口**：相比轻量纯 1D 时序模型，引入 ViT 骨干增加了约 1.2 亿参数，单序列推理耗时从 24.49 ms 增至 51.02 ms（表 12）；多模态时序表征在数理理论层面仍缺乏严格的形式化解释证明（§5）[^src-2602-16681-vetime]。

[^src-2602-16681-vetime]: [[source-2602-16681-vetime]]
