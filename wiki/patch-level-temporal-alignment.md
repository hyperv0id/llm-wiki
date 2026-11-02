---
title: "Patch-level Temporal Alignment（PTA）"
type: technique
tags:
  - time-series
  - anomaly-detection
  - temporal-alignment
  - multimodal
  - vision-transformer
created: 2026-11-02
last_updated: 2026-11-02
source_count: 1
confidence: low
status: active
---

# Patch-level Temporal Alignment（PTA）

补丁级时序对齐（Patch-level Temporal Alignment, PTA）是 [[vetime|VETime]] 框架中将预训练 Vision Transformer（ViT）提取的 2D 空间补丁表征逆映射并投影回 1D 共享时间轴的核心时间轴对齐技术（arXiv:2602.16681，§3.2）[^src-2602-16681-vetime]。

## 动机与设计背景

在多模态时序建模中，视觉与时序分支存在严重的结构与语义失配：
1. **拓扑与数量失配**：冻结的 [[mae|MAE]] ViT-Base 编码器在标准 $224 \times 224$ 输入下，输出 $N_V = (224/16) \times (224/16) = 196$ 个二维空间 patch 表征（通道维 $D_V=768$）；而 1D 时序编码器在窗口长度 $L=5000$、patch 边长 $P=16$ 条件下，输出 $N_{TS} = L/16 = 313$ 个时序 token（通道维 $D_{TS}=512$）[^src-2602-16681-vetime]。
2. **时空割裂与噪声引入**：直接采用晚期决策融合或粗暴特征拼接，会丢失时间戳与图像局部区域之间的显式时空对应关系，引入严重的时空噪声，破坏逐点异常定位精度 [^src-2602-16681-vetime]。
3. **PTA 的设计定位**：PTA 通过严格逆转 [[traceable-image-conversion|TIC]] 的折叠几何逻辑，对视觉特征在时间轴和周期轴上执行解耦的线性重采样与平均池化，并注入绝对位置编码与自注意力机制，恢复显式的时序序列属性，构建与 1D 时序特征严格对齐的共享时间线 [^src-2602-16681-vetime]。

## 逆折叠重采样流水线与维度变换

输入为冻结视觉编码器抽取的初始特征 $F_V^0 \in \mathbb{R}^{N_V \times D_V}$，其中 $N_V = 196, D_V = 768$。

$$\begin{aligned}
F_V^{\text{grid}} &= \operatorname{Reshape}(F_V^0) && \text{// } [N_V, D_V] \xrightarrow{\text{恢复空间几何网格}} [14, 14, D_V]\ \text{——还原为 } 14 \times 14 \text{ 空间拓扑}\\
F_V^{\text{time}} &= \operatorname{Interp1D}(F_V^{\text{grid}}, N_{TS}) && \text{// } [14, 14, D_V] \xrightarrow{\text{沿时间水平轴线性插值}} [N_{TS}, 14, D_V]\ \text{——时间 patch 数上采样至 } N_{TS}=313\\
\hat{F}_V &= \operatorname{AvgPool2D}(F_V^{\text{time}}) && \text{// } [N_{TS}, 14, D_V] \xrightarrow{\text{沿周期垂直轴平均池化}} [N_{TS}, D_V]\ \text{——聚合周期冗余信息并展平}\\
\tilde{F}_V &= \hat{F}_V + E_{\text{pos}} && \text{// } [N_{TS}, D_V] + [N_{TS}, D_V] \to [N_{TS}, D_V]\ \text{——注入可学习绝对位置编码恢复序列顺序}\\
F_V^{\text{proj}} &= \tilde{F}_V W_P && \text{// } [N_{TS}, \bcancel{D_V}] \times [\bcancel{D_V}, D] \to [N_{TS}, D]\ \text{——线性投影消去 } D_V=768 \text{，映射至统一维度 } D=512\\
F_V &= \operatorname{SelfAttn}(F_V^{\text{proj}}) && \text{// } [N_{TS}, D] \xrightarrow{\text{自注意力建模 patch 间依赖}} [N_{TS}, D]\ \text{——输出与 } F_{TS} \text{ 完全对齐的视觉表征}
\end{aligned}$$

### 流水线关键步骤分析
1. **空间网格重构**：将 196 个二维 patch 特征向量重塑为二维几何网格 $(N_T, N_T / T_{\text{fold}}, D_V)$，对应 $14 \times 14 \times 768$。
2. **时间轴线性插值**：由于 TIC 编码时沿水平轴延展时间信息，PTA 沿水平时间轴执行一维线性插值，将特征点数量从 14 精准对齐至时序补丁数 $N_{TS}=313$，实现与 1D 时序补丁序列长度的点对点对应。
3. **周期轴冗余聚合**：沿垂直周期轴执行平均池化（average pooling），对跨周期的重复信息进行平滑降维聚合，得到初步对齐的视觉补丁特征 $\hat{F}_V \in \mathbb{R}^{N_{TS} \times D_V}$。
4. **位置编码注入与自注意力建模**：视觉编码与周期轴池化削弱了序列的绝对时间顺序。PTA 引入可学习的绝对位置编码 $E_{\text{pos}} \in \mathbb{R}^{N_{TS} \times D_V}$ 恢复时间顺序，经线性投影层映射至时序统一维度 $D=512$，最后送入自注意力层（self-attention）建模 intra-patch 与 inter-patch 依赖关系，输出与时序特征 $F_{TS}$ 完全对齐的视觉表征 $F_V \in \mathbb{R}^{N_{TS} \times D_{TS}}$（§3.2）[^src-2602-16681-vetime]。

## 多变量扩展适配（附录 A）

在多变量设置下，PTA 处理输入形式为 $\hat{F}_V \in \mathbb{R}^{N_{TS} \times N_v \times D_V}$ 的张量（其中 $N_v$ 为变量数）。PTA 为每个变量的 token 序列注入相同的绝对位置编码 $E_{\text{pos}}$，并利用 Any-variate attention [Woo et al., 2024] 独立处理各变量的时序动态矫正，重组为 $F_V \in \mathbb{R}^{(N_{TS} N_v) \times D_{TS}}$，使模型能够在通道独立前提下解耦重排各变量的时序因果关系（附录 A）[^src-2602-16681-vetime]。

## 实证证据与消融分析

### 1. 模块级消融表现（附录 D.5, 表 17）
移除 PTA 模块（变体 V5，仅保留 TIC 与 TMF）导致跨模态交互退化为粗粒度语义碰撞，丢失细粒度局部对应：
- 在 IOPS 数据集上，VUS-PR 从完整模型的 30.79% 显著下滑至 26.26%；
- 在 NAB 数据集上，Affiliation-F1 / VUS-PR 从完整模型的 88.57% / 32.98% 降至 83.23% / 24.81%（甚至低于 Base 的 24.32%）。

### 2. 子组件微观消融（附录 D.6, 表 18）
在 NAB、WSD 与 YAHOO 数据集上针对 PTA 内部微观子组件的消融结果如下（表 18）[^src-2602-16681-vetime]：

| PTA 内部变体 | NAB (Aff-F1 / VUS-PR) | WSD (Aff-F1 / VUS-PR) | YAHOO (Aff-F1 / VUS-PR) | 出处 |
|---|---|---|---|---|
| 去除自适应折叠（w/o adaptive folding） | 87.81% / 29.99% | 91.49% / 30.41% | 97.10% / 86.70% | 表 18 |
| 去除位置编码（w/o positional encoding） | 88.16% / 30.27% | 92.29% / 31.18% | 97.25% / 86.81% | 表 18 |
| 去除自注意力机制（w/o attention） | **85.69% / 29.88%** | **91.57% / 29.46%** | **96.92% / 85.53%** | 表 18 |
| PTA 完整结构（Full Model） | **88.57% / 32.98%** | **94.31% / 33.85%** | **97.15% / 88.61%** | 表 18 |

结果表明，自注意力机制对捕获长程时序依赖和对齐视觉-时序语义至关重要（移除后 NAB Aff-F1 降幅最大，跌至 85.69%）；去除自适应折叠与位置编码亦均导致显著的指标滑落。

### 3. 多模态融合层级对比（附录 D.3, 表 15）
补丁级特征对齐融合（Patch-level Alignment feature-level fusion）相比其他层级在 11 个单变量数据集上全面胜出：
- 在 YAHOO 上，特征级对齐融合的 VUS-PR 达到 83.82%，而决策级融合（Decision-level）仅为 79.43%，分数级融合（Score-level）仅为 82.34%，直接特征拼接（Direct feature-level）仅为 80.33%；
- 在 IOPS 上，特征级对齐融合的 VUS-PR 达到 25.75%，决策级仅为 16.07%，分数级为 20.35%，直接特征拼接为 22.67%。

[^src-2602-16681-vetime]: [[source-2602-16681-vetime]]
