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
last_updated: 2026-11-22
source_count: 1
confidence: low
status: active
---

# VETime

时序异常检测的两种主流做法各有盲点：1D 数值模型靠滑动窗口，感受野有限，看不出「整段上下文形状被破坏」这类长程结构违背；把序列折叠成图像交给 2D ViT 或 VLM 虽能看到全局形态，却在折叠与分块中丢掉时间索引，推理也慢。VETime（Vision Enhanced Zero-Shot Time Series Anomaly Detection）把两条路的表征按时间轴逐 patch 对齐，再按样本动态分配模态权重，让视觉特征只在自己可靠的区间发言。该模型由清华大学、华为 2012 实验室与 Datadog AI Research 提出[^src-2602-16681-vetime]。

在时序多模态的坐标上，面向预测的模型（[[time-vlm|Time-VLM]]、[[aurora|Aurora]]）关注宏观演化趋势，缺少异常检测所需的逐点定位敏感度与零样本泛化能力；VETime 把对齐粒度压到 patch 级，用图像转换、时间轴对齐与跨模态增强三段拼出特征级融合，服务于 [[multimodal-time-series-anomaly-detection|多模态时序异常检测]] [^src-2602-16681-vetime]。

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

视觉分支是冻结的 [[mae|MAE]] ViT-Base（85.64M 参数），时序分支是 TimeRCD 初始化、以秩 $r=8$、$\alpha=16$ 的 LoRA 微调的 8 层 8 头 Transformer[^src-2602-16681-vetime]。

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

模型在 0.5B 数学方程合成数据上端到端多任务预训练，总损失是检测、重构、对比与负载均衡四项的加权和[^src-2602-16681-vetime]：
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

## 证据

1. **定位：严格零样本设定下确实赢了同代基线，不只是追平。** TSB-AD 的 26 个数据集、104 项指标中取得 58 项第一；单变量 11 个数据集 44 项平均排名 2.05、多变量 15 个数据集 60 项 1.58，次优基线分别是 TimeRCD/Chronos 的 3.27 与 DADA 的 2.40。在单变量指标上它甚至优于在目标域训练的全样本模型（23 项第一、平均排名 2.02）。对 4 个单变量数据集上的视觉与 VLM 基线（VLM4TS、ViT4TS、VisualTimeAnomaly、AnomLLM）全指标领先，单序列耗时 0.02–0.04 s 对 6.78–9.27 s，低两个数量级（表 1、表 2、表 7–8）[^src-2602-16681-vetime]。
2. **归因：收益来自视觉侧的跨模态对齐，不是参数量。** 把视觉分支换成第二个时序分支（Dual-TS，160.49M）后，参数量、显存与耗时都与完整模型（157.96M、787 MB、51.02 ms）相当，但 NAB 的 Affiliation-F1 只有 80.84 对 88.57、WSD 88.87 对 94.31、YAHOO 96.81 对 97.15；纯时序 Base 仅 37.06M，对应 82.48 / 90.33 / 96.65。按异常形态拆分后 Point、Context、Mixed 三类在多数指标上领先，但 Point 的 VUS-PR 59.33 低于 TimeRCD 的 61.85（表 3、表 12）[^src-2602-16681-vetime]。
3. **缺口与负结果：非周期与周期性尖峰数据上会负迁移。** 无固定周期的 MGAB 上 Affiliation-F1 从纯时序 Base 的 70.69 掉到变体 V1 的 67.51，后续融合模块救不回来；Power 的周期性尖峰被折叠成背景纹理，零样本路由器给受骗的视觉表征过高权重，VUS-PR 从 Base 的 21.25 掉到 11.54，低于纯时序占优变体 V3 的 21.48。25–38 变量的中高维数据因 224×224 画布像素上限性能回落，59 变量的 SMAP 因通道冗余反而没有。VLM 对照只覆盖 4 个单变量数据集，异常形态拆分只在 NAB 与 YAHOO 上做，多变量侧没有同类分析（附录 C.3、D.5）[^src-2602-16681-vetime]。

## 边界与局限

1. **高频噪声敏感性**：依赖季节-趋势分解，噪声与结构缠绕时分解会把噪声误划进趋势或残差，视觉 RGB 映射随之失真。
2. **高维物理画布上限**：通道独立假设忽略变量间依赖；$224 \times 224$ 画布配 5000 长度滑窗，每点至少占一个像素的上限约 10 个变量（$224\times224/5000\approx10$），超过即像素重叠与混叠。
3. **非周期信号与周期性尖峰的负迁移**：TIC 强制折叠在无固定周期的混沌序列与随机游走金融序列上造出伪模式；周期性尖峰异常被折成背景纹理，零样本路由器会把权重让给受骗的视觉表征（§5、附录 C.3、D.5）[^src-2602-16681-vetime]。
4. **开销与可解释性**：ViT 骨干比轻量纯 1D 模型多约 1.2 亿参数，单序列推理耗时从 24.49 ms 增至 51.02 ms；多模态表征仍缺形式化解释（§5）[^src-2602-16681-vetime]。

[^src-2602-16681-vetime]: [[source-2602-16681-vetime]]
