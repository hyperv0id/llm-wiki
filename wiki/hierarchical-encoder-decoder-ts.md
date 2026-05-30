---
title: "Hierarchical Encoder-Decoder for Time Series (HED)"
type: technique
tags:
  - time-series
  - transformer
  - hierarchical
  - encoder-decoder
  - multi-scale
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# Hierarchical Encoder-Decoder for Time Series (HED)

HED 是 [[crossformer|Crossformer]] 的编码器-解码器架构，利用多尺度信息进行预测 [^src-crossformer-2023]。

## 编码器

每层（除第一层）将相邻两个时间段向量合并为更粗粒度的表示，然后用 [[two-stage-attention|TSA layer]] 捕获该尺度的依赖 [^src-crossformer-2023]：

$$\hat{\mathbf{Z}}^{\text{enc},l}_{i,d} = M[\mathbf{Z}^{\text{enc},l-1}_{2i-1,d} \cdot \mathbf{Z}^{\text{enc},l-1}_{2i,d}]$$
$$\mathbf{Z}^{\text{enc},l} = \text{TSA}(\hat{\mathbf{Z}}^{\text{enc},l})$$

其中 $M \in \mathbb{R}^{d_\text{model} \times 2d_\text{model}}$ 为可学习的段合并矩阵。上层向量覆盖更长的时间范围，捕获更粗粒度的依赖 [^src-crossformer-2023]。

## 解码器

$N+1$ 个解码器层对应 $N+1$ 个编码器输出 [^src-crossformer-2023]：

1. TSA 处理解码器输入
2. MSA 以解码器输出为 query、编码器输出为 key/value，建立编码器-解码器连接
3. 每层通过线性投影生成该尺度的预测

最终预测为所有层预测之和 [^src-crossformer-2023]：
$$\mathbf{x}^{\text{pred}}_{T+1:T+\tau} = \sum_{l=0}^{N} \mathbf{x}^{\text{pred},l}_{T+1:T+\tau}$$

## 消融实验

HED 的效果取决于预测长度 [^src-crossformer-2023]：
- 短期预测：HED 略降低精度（多尺度信息对短预测帮助不大）
- 长期预测：HED 提升精度（粗粒度信息有助于长期趋势捕获）

这与直觉一致：不同尺度的信息对长期预测更有价值 [^src-crossformer-2023]。

## 与其他层次结构对比

- **Informer**：单尺度编码器-解码器，ProbSparse attention [^src-crossformer-2023]
- **Pyraformer**：金字塔注意力模块，不同分辨率的特征 [^src-crossformer-2023]
- **Swin Transformer**（CV）：层级特征图 + 移窗注意力，启发 HED 的段合并 [^src-crossformer-2023]

## Connections

- 属于：[[crossformer]] — 架构组件
- 相关：[[dsw-embedding]] — 提供 2D 输入
- 相关：[[two-stage-attention]] — 编码器和解码器的核心
- 相关：[[lstf]] — HED 主要提升长期预测

[^src-crossformer-2023]: [[source-crossformer-2023]]
