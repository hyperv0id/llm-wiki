---
title: "Space-Filling Curves for Spatio-Temporal Scanning"
type: technique
tags:
  - serialization
  - spatio-temporal
  - mamba
  - scanning-path
created: 2026-07-21
last_updated: 2026-07-21
source_count: 1
confidence: medium
status: active
---

# Space-Filling Curves for Spatio-Temporal Scanning

空间填充曲线是一种将高维空间点映射到 1D 序列的双射函数 Φ: Z³ → N，使得序列中相邻的点在空间上也保持近邻关系[^src-rivermamba]。RiverMamba 首次将这一概念引入河流流量和洪水预报的时空建模中，作为 Mamba 块处理前的关键预处理步骤。

## 在 RiverMamba 中的应用

RiverMamba 使用空间填充曲线将全球 0.05° 网格上采样的 P 个点序列化为 1D 序列，供 [[mamba|Mamba]] 块做双向选择性 SSM 扫描[^src-rivermamba]。这一设计解决了两个核心挑战：

1. **计算可行性**：Transformer 的二次复杂度在全球洪水预报中不可行（P 可达数十万），而 Mamba 的线性复杂度结合序列化扫描使全球尺度时空建模成为可能。实验证明 Flash-Attention 替代方案在推理速度和精度上均不如序列化+Mamba。
2. **感受野覆盖**：通过在多个 Hindcast 层中交替使用不同曲线，每个采样点被从多个空间视角扫描，确保完整河网（如亚马逊河）的上下游路由关系被模型捕获。

## 四条曲线交替策略

RiverMamba 使用四条空间填充曲线，按 Hindcast 层顺序交替[^src-rivermamba]：

| 层 | 曲线 | 扫描方向 |
|---|------|---------|
| 1 | Sweep | 水平 |
| 2 | Sweep | 垂直（水平曲线的转置） |
| 3 | Generalized Hilbert (Gilbert) | — |
| 4+ | Gilbert 转置 | — |

消融实验表明，Sweep + Gilbert 组合优于单独使用任一种曲线。时间维度通过连接相邻时间步的曲线端点实现：曲线在时间 t 的最后一点连接到时间 t+1 的第一点。

## 序列化与反序列化

序列化编码 Φ : Z³ → N 将每个时空点映射为唯一序列索引，反序列化解码 Φ⁻¹ : N → Z³ 将 Mamba 输出恢复为原始空间布局。由于相邻层使用不同曲线，每层结束时必须反序列化，下一层再按新曲线重新序列化。

## 历史背景

空间填充曲线的概念由 Peano (1890) 和 Hilbert (1935) 首次提出，最初用于证明连续曲线可以填满平面区域[^src-rivermamba]。RiverMamba 将其现代化应用于深度学习中的时空序列建模。

## 相关页面

- [[rivermamba|RiverMamba]] — 该技术的首个水文应用
- [[mamba|Mamba]] — 选择性状态空间模型
- [[mamba-block-design|Mamba Block Design]]

[^src-rivermamba]: [[source-rivermamba]]
