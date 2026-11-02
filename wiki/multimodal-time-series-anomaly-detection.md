---
title: "Multimodal Time Series Anomaly Detection"
type: concept
tags:
  - time-series
  - anomaly-detection
  - multimodal
  - text-encoding
created: 2026-05-03
last_updated: 2026-11-02
source_count: 2
confidence: high
status: active
---

# Multimodal Time Series Anomaly Detection

## Definition

**Multimodal time series anomaly detection** is the task of identifying anomalous events in time series data by jointly leveraging numerical time series and complementary modalities — primarily text — to improve detection accuracy over unimodal approaches[^src-multimodal-ts-anomaly-detection]. 随后该领域亦拓展至将 1D 数值序列与 2D 视觉图像联合建模的跨模态密集对齐路线[^src-2602-16681-vetime]。

## Motivation

Traditional time series anomaly detection methods (e.g., DCdetector, GDN, Anomaly Transformer, DADA) operate exclusively on numerical data[^src-multimodal-ts-anomaly-detection]. However, real-world scenarios often provide rich contextual information in text form:

| Domain | Time Series | Text Modality |
|--------|-------------|---------------|
| Finance | Stock prices, trading volume | News reports, policy announcements |
| Healthcare | Vital signs, lab results | Clinical notes, patient history |
| Environment | Air quality, temperature | Weather forecasts, pollution reports |
| Network | Traffic, latency | Incident reports, maintenance logs |

Text provides background knowledge and contextual descriptions that can help distinguish true anomalies from expected variations.

## Key Challenges

### 1. Semantic Alignment

Time series (continuous, numerical) and text (discrete, semantic) reside in fundamentally different representation spaces. Naive fusion (concatenation, linear interpolation) fails to capture meaningful cross-modal relationships[^src-multimodal-ts-anomaly-detection].

### 2. Redundant Information

Text often contains lengthy descriptions irrelevant to anomaly detection. Without filtering, redundant content dilutes the contribution of genuinely informative text and can degrade model performance[^src-multimodal-ts-anomaly-detection].

### 3. Visual-Temporal Ordered Alignment and Multimodal Synergy

在视觉-时序双模态路线中，时序转图像过程常面临两个关键障碍：(1) **时序索引丢失与弱对齐**：传统时序转图像（如 GAF、RP）为不可逆变换，或折线图渲染缺乏补丁与原始时间戳的显式映射，导致定位窗口粗糙并破坏点级时间戳分辨率；(2) **多样异常模式的动态协同**：点异常依赖高分辨率局部敏感性，而长程上下文异常依赖宏观全局视野，如何在未知领域中自适应调配 1D 局部相关性与 2D 宏观模式的贡献权重仍具挑战[^src-2602-16681-vetime]。

## MindTS Approach

[[mindts|MindTS]] (ICLR 2026) is the first dedicated model for this task, addressing both challenges through two modules:

1. **[[fine-grained-time-text-semantic-alignment|Fine-grained Time-text Semantic Alignment]]**: Decomposes text into endogenous (per-patch statistics) and exogenous (shared background) views, fuses them via cross-view attention, and aligns with time series using contrastive learning[^src-multimodal-ts-anomaly-detection].

2. **[[content-condenser-reconstruction|Content Condenser Reconstruction]]**: Filters redundant text via mutual information minimization (Information Bottleneck principle), then uses condensed text to reconstruct masked time series, forcing meaningful cross-modal interaction[^src-multimodal-ts-anomaly-detection].

## VETime Approach: Time-Series and 2D Vision Dense Alignment

针对视觉模态对齐模糊与推理开销巨大的问题，[[vetime|VETime]] 提出了首个统一 1D 时序与 2D 视觉特征的零样本时序异常检测框架，包含四个核心组件[^src-2602-16681-vetime]：

1. **[[traceable-image-conversion|Traceable Image Conversion (TIC)]]**：将 1D 序列经季节-趋势分解映射至 RGB 三通道生成 $1 \times L \times 3$ 张量，通过自相关函数（ACF）估计周期并约束折叠宽度为 ViT patch 大小（16）的整数倍，辅以水平时间轴线性插值与垂直周期轴复制填充，生成保留时序相位与全局趋势的 2D 图像输入（§3.1）[^src-2602-16681-vetime]。
2. **[[patch-level-temporal-alignment|Patch-Level Temporal Alignment (PTA)]]**：通过冻结预训练 MAE（ViT-Base）提取 2D 补丁特征，执行反向逆折叠、时间轴插值与周期轴平均池化，将其展平对齐为与时序补丁数严格一致的表征，并注入可学习时间位置编码和自注意力层，恢复时间先后顺序（§3.2）[^src-2602-16681-vetime]。
3. **[[anomaly-window-contrastive-learning|Anomaly Window Contrastive Learning (AWCL)]]**：在双向交叉注意力交互后，针对短点异常（$\le 1$ patch）执行窗内局部对比，针对长上下文异常（$\ge 2$ patches）执行窗间全局池化对比，强化多尺度判别能力（§3.3）[^src-2602-16681-vetime]。
4. **[[task-adaptive-multimodal-fusion|Task-Adaptive Multi-Modal Fusion (TMF)]]**：将时序、视觉与异常增强表征视作三路专家，通过动态权重门控路由至异常检测头与辅助重构头，并引入负熵负载均衡损失 $L_e$ 防止路由塌缩（§3.4）[^src-2602-16681-vetime]。

在 TSB-AD 26 个数据集（11 单变量、15 多变量）的零样本评测中，VETime 取得最优平均排名（单变量 2.05，多变量 1.58，表 1）；并在 NAB、YAHOO、SMAP、MSL 单变量数据集上相比 VLM4TS、ViT4TS 等折线图/VLM 模型取得约 100 倍推理加速（单序列耗时 0.02–0.04s vs 1.72–9.27s，表 2）[^src-2602-16681-vetime]。

## Comparison with Multimodal Time Series Forecasting

| Dimension | [[multimodal-time-series-forecasting|Forecasting]] | Anomaly Detection |
|-----------|----------------|-------------------|
| Goal | Predict future values | Identify abnormal events |
| Output | Continuous values | Binary labels / anomaly scores |
| Key models | UniCA, MoST, ChannelMTS | MindTS |
| Text role | Covariate for prediction | Context for anomaly discrimination |
| Evaluation | MAE, MAPE, MSE | Aff-F1, VUS-PR, VUS-ROC |

### 预测型多模态模型的任务错位

现有多模态时序先驱模型（如 [[time-vlm|Time-VLM]]、[[aurora|Aurora]]）专为预测（forecasting）设计。VETime 论文指出，预测任务优先关注宏观趋势演化，缺乏密集体时间戳级（timestamp-level）判别能力；预测-残差范式在零样本面对未知多样异常时难以泛化，因而无法直接替代原生时序异常检测模型（§1）[^src-2602-16681-vetime]。

## Datasets

MindTS evaluates on 6 real-world multimodal datasets: Weather, Energy, Environment, KR, EWJ, and MDT, each containing both numerical time series and corresponding exogenous text[^src-multimodal-ts-anomaly-detection].

## Related Pages

- [[mindts]] — the MindTS model
- [[fine-grained-time-text-semantic-alignment]] — alignment technique
- [[content-condenser-reconstruction]] — redundancy filtering technique
- [[multimodal-time-series-forecasting]] — related multimodal TS task
- [[source-multimodal-ts-anomaly-detection]] — source summary
- [[vetime]] — VETime 视觉增强零样本时序异常检测基础模型
- [[traceable-image-conversion]] — 可追溯图像转换技术
- [[patch-level-temporal-alignment]] — 补丁级时序对齐技术
- [[anomaly-window-contrastive-learning]] — 异常窗口对比学习机制
- [[task-adaptive-multimodal-fusion]] — 任务自适应多模态融合技术
- [[time-vlm]] — 预测型多模态时序模型
- [[aurora]] — 生成式多模态时序基础模型
- [[source-2602-16681-vetime]] — VETime 论文源摘要

[^src-multimodal-ts-anomaly-detection]: [[source-multimodal-ts-anomaly-detection]]
[^src-2602-16681-vetime]: [[source-2602-16681-vetime]]
