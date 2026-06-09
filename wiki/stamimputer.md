---
title: "STAMImputer"
type: entity
tags:
  - spatio-temporal
  - traffic
  - data-imputation
  - mixture-of-experts
  - graph-attention
  - low-rank
  - block-missing
  - arxiv-2025
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# STAMImputer

**STAMImputer** (Spatio-Temporal Attention Mixture-of-experts Imputer) 是 Wang 等人 (Beihang University 等, arXiv 2025 / IJCAI 2025) 提出的交通数据填补模型，号称首次将 [[mixture-of-experts|MoE]] 框架应用于交通数据填补任务[^src-stamimputer]。其核心是用一个**外层 MoE** 动态平衡时间与空间注意力专家的贡献，以应对块状缺失（[[partial-blackout|block missing]]）与非平稳交通数据的分布偏移[^src-stamimputer]。

## 设计动机

STAMImputer 旨在同时克服两个长期问题[^src-stamimputer]：

- **time-to-space 顺序框架在块缺失下失效**：先时间后空间的串行处理在连续多步缺失时会传播无效特征[^src-stamimputer]。
- **静态图 vs. 全局注意力的两难**：静态图捕获不了局部社区外的全局依赖，纯全局注意力在稀疏数据下又缺乏先验引导[^src-stamimputer]。

作者的回答是用 MoE 做**受控解耦**：让时间、空间专家各自捕获维度内特征，再由观测专家依据实时数据特性裁决两者权重[^src-stamimputer]。

## 架构：外层 MoE 三阶段

与传统将专家放在 Transformer FFN 内层的 MoE 不同，STAMImputer 把专家网络上移到框架外层[^src-stamimputer]。整个流程分三阶段[^src-stamimputer]：

| 阶段 | 组件 | 作用 |
|------|------|------|
| 注意力阶段 | T-Expert (MSAT) + S-Expert (LrSGAT) | 分别提取时间/空间相关性 |
| 观测阶段 | O-Expert (FFN) | 对注意力专家输出打置信分并加权 |
| 读出阶段 | MLP | 产生最终填补结果 |

### 时间专家 (T-Expert)

采用经典自注意力 Transformer 编码器，即 Multi-head Self-ATtention (MSAT)。时间专家擅长恢复稳定的低频序列趋势，对低缺失率/简单缺失模式（如随机点缺失）能快速预重建，其输出又为空间专家提供更丰富的信息[^src-stamimputer]。

### 空间专家 (S-Expert)

即 [[lrsgat|LrSGAT]]（Low-rank guided Sampling Graph ATtention），STAMImputer 的核心创新。它通过基于显著度评分的混合采样选出"交通枢纽"节点，再用低秩引导的 re-attention 压缩-重构空间矩阵，同时生成**半自适应动态图**[^src-stamimputer]。详见 [[lrsgat]]。

### 观测专家 (O-Expert)

观测专家是一个前馈网络，扮演"宏观控制者/仲裁者"。它聚焦原始数据（含缺失）与稀疏度特征，对时空注意力输出做可信度评估，并通过 softmax 生成置信分矩阵 $E^{ST}$ 来动态加权各注意力专家[^src-stamimputer]。这一设计的依据是：随机缺失下时间注意力更优，块缺失下空间注意力更优——观测专家据此实时调整偏好[^src-stamimputer]。除路由外，它还兼具动态残差连接的作用，提升稀疏模式下的稳定性[^src-stamimputer]。

> [!note] 与 TESTAM 的对比
> [[testam|TESTAM]] (ICLR 2024) 同为 MoE-based 时空注意力模型，但用于**交通预测**，以 [[memory-augmented-gating|记忆增强门控]] 在 3 个异质 expert（恒等/静态图/动态图）间路由[^src-stamimputer]。STAMImputer 面向**填补**，专家分工是按维度（时间 vs 空间），路由由观测专家依据稀疏度特征裁决，二者机制不同。

## 时空表征学习

进入 MoE 前，输入先做特征增强[^src-stamimputer]：

- **离散小波变换 (DWT/IWT)**：将序列分解为低频分量 $X^l$ 与高频分量 $X^h$，为嵌入提供额外频域维度（消融显示去掉 DWT 会损害效果）[^src-stamimputer]。
- **时空嵌入**：拼接采集时间+工作日信息 $P^u$、可学习时空位置编码 $P^{st}$、以及时间/空间双向稀疏度 $P^{sp}$（供观测专家评估置信分）[^src-stamimputer]。

## 实验

- **基准**：PemsD8、SZ-Taxi、DiDi-SZ、NYC-Taxi（节点数 156–627，交通流/速度，C=1）[^src-stamimputer]。
- **缺失模式**：点缺失 (25%/60%) 与块缺失 (0.2%/1% 故障概率，对应约 10%/30% 总缺失率, $S\sim U(12,48)$ 步)[^src-stamimputer]。
- **结果**：以 MAE 衡量，总体优于 [[imputeformer|ImputeFormer]]、SPIN、SAITS、Transformer、BRITS、rGAIN 及统计基线，城市出行数据集优势更明显[^src-stamimputer]。
- **消融**：注意力专家最关键，块缺失下 LrSGAT→MLP 退化最严重；MoE、采样、投影、DWT 各组件均有正贡献[^src-stamimputer]。
- **下游**：动态图 (DGSL) 接入 Graph-WaveNet 后，在高稀疏与块缺失场景显著改善预测 MAE[^src-stamimputer]。
- **复杂度**：$O(T^2D + N\log N\,D + NTD)$；DGSL 层为 $O(N^2\log N)$[^src-stamimputer]。

## 局限

- 仅在单通道交通数据上验证，未做多变量/跨域泛化[^src-stamimputer]。
- 动态图构建效率为作者明确的未来工作[^src-stamimputer]。

[^src-stamimputer]: [[source-stamimputer]]
