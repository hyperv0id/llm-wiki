---
title: "Feature Dependency Encoder"
type: technique
tags:
  - self-attention
  - time-series
  - data-imputation
  - diffusion-models
  - dilated-convolution
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: high
status: active
---

# Feature Dependency Encoder (FDE)

**FDE** (Feature Dependency Encoder) 是 [[sadi|SADI]] 中用于显式建模多元时间序列中特征间时间感知依赖关系的组件[^src-sadi]。与 [[csdi|CSDI]] 将每个时间步视为独立同分布进行特征维度注意力的做法不同，FDE 通过学习时间序列级别的特征关联来捕获更丰富的跨通道信息[^src-sadi]。

## 机制

FDE 由 $N_{FDE}$ 层堆叠组成，每层执行以下操作[^src-sadi]：

1. **1-D 膨胀卷积**：核大小 $(1 \times 3)$，在时间维度上膨胀。每层的膨胀率从 $n=1$ 开始，逐层递增（第 $n$ 层 dilation=$n$），扩展感受野以捕获从短程到长程的局部模式
2. **Self-Attention on Feature Dimension**：在卷积输出转置后，沿特征维度做自注意力，学习跨特征的时间序列级别关系
3. **LayerNorm + FFN**：标准 Transformer 风格后处理

形式化定义[^src-sadi]：

$$FDE_n(X) = \text{self-attn}(\text{conv}_{1\times3}(X^T,\text{dilation}=n)^T)$$

$$\hat{X} = \begin{cases} FDE_n(X) & n=1 \\ FDE_n(\hat{X}) & 1 < n \leq N_{FDE} \end{cases}$$

## 与 CSDI Feature Transformer 的区别

| 维度 | CSDI Feature Transformer | SADI FDE |
|------|--------------------------|----------|
| 时间感知 | 否（每时间步独立处理） | **是**（膨胀卷积 + 自注意力） |
| 局部模式 | 无 | **膨胀卷积**逐层扩大感受野 |
| 全局依赖 | 自注意力 on 特征 | 自注意力 on 特征（时间序列级） |
| 输入处理 | 分割为 $K$ 个实例 | 联合处理（单通道架构） |

## 消融证据

SADI 消融实验表明[^src-sadi]：

- FDE 对 NACSE 和 Electricity 数据集**尤其关键**——这两个数据集的特征间相关性高，移除 FDE 后性能急剧下降
- 在 AgAID 数据集上，移除 FDE 后 MSE 从 $2.93\times10^{-4}$ 升至 $1.43\times10^{-3}$（~4.9×）
- 在 Electricity 数据集上（10 缺失特征），移除 FDE 后 MSE 从 0.107 飙升至 >9.5（~89×），说明高维特征相关场景下 FDE 不可或缺

## 设计直觉

膨胀卷积的递增膨胀率设计使 FDE 能够在不同粒度上同时提取局部模式——低层捕获相邻时间步的短期特征共变，高层捕获远距离的长期特征间依赖模式[^src-sadi]。这种多尺度特征依赖建模是应对 [[partial-blackout|partial blackout]] 的关键——当部分特征缺失时，模型需要从观测特征的完整时间序列模式中推断缺失值。

## 关联页面

- [[sadi]] — SADI，FDE 的宿主模型
- [[partial-blackout]] — FDE 针对的缺失场景
- [[gated-temporal-attention]] — GTA，FDE 的互补组件（建模时间依赖）
- [[csdi]] — CSDI，FDE 替代的 baseline 特征建模方式
- [[dilated-convolution]] — 膨胀卷积技术

[^src-sadi]: [[source-sadi]]
