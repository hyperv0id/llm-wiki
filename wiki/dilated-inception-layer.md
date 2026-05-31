---
title: "Dilated Inception Layer"
type: technique
tags:
  - temporal-convolution
  - dilated-convolution
  - inception
  - time-series
created: 2026-05-30
last_updated: 2026-05-30
source_count: 1
confidence: medium
status: active
---

# Dilated Inception Layer

Dilated Inception Layer（扩张初始层）是 [[mtgnn|MTGNN]] (KDD 2020) 提出的时间卷积模块核心组件，通过组合多尺度滤波器和指数扩张实现极长序列的高效时间依赖捕获 [^src-mtgnn]。

## 机制

扩张初始层融合两个来自卷积神经网络的成熟策略 [^src-mtgnn]：

### 1. 多尺度滤波器 (Inception)

使用四种滤波器尺寸：1×2、1×3、1×6、1×7，而非传统 Inception 的 1×1/1×3/1×5 [^src-mtgnn]。设计理由：时间序列信号具有固有周期（如 7、12、24、28、60），这些滤波器尺寸的组合可覆盖所有常见周期 [^src-mtgnn]。例如，周期 12 可通过第一层的 1×7 滤波器接第二层的 1×6 滤波器得到 [^src-mtgnn]。

四个滤波器的输出按最大长度截断对齐，在通道维度拼接 [^src-mtgnn]。

### 2. 指数扩张 (Exponential Dilation)

扩张因子 d 随层数以 q 的速率指数增长 [^src-mtgnn]。m 层扩张卷积网络（核大小 c）的感受野为：

R = 1 + (c-1)(q^m - 1)/(q - 1)

这意味着感受野随层数增加*指数级*扩展（而标准卷积仅线性扩展），使浅层网络即可处理极长序列 [^src-mtgnn]。

## 在时间卷积模块中的使用

MTGNN 的时间卷积模块包含两个扩张初始层，构成门控机制 [^src-mtgnn]：

- **滤波层** — Tanh 激活 → 提取高层时间特征
- **门控层** — Sigmoid 激活 → 控制信息流动

即 H_out = Tanh(Filter(H_in)) ⊙ Sigmoid(Gate(H_in))，等价于 GLU (Gated Linear Unit) 风格的门控 [^src-mtgnn]。

## 与前作的区别

- 标准卷积：O(Nlc_co/d)，感受野线性增长；扩张初始层感受野指数增长，在相同深度下覆盖更长时间跨度 [^src-mtgnn]
- WaveNet (Oord et al., 2016)：使用单一滤波器尺寸 + 扩张，MTGNN 加入多尺度滤波器以适应时间序列的多周期特性
- Inception (Szegedy et al., 2015)：2D 图像滤波器尺寸 (1×1, 3×3, 5×5)，MTGNN 改造为时间序列适配的 (1×2, 1×3, 1×6, 1×7)

## 消融结果

|| MAE || RMSE || MAPE ||
| 完整 MTGNN | 2.7715 | 5.8070 | 0.0778 |
| 移除 Inception（仅用 1×7 滤波器） | 2.7772 | 5.8251 | 0.0778 |

移除 inception 后 RMSE 上升但 MAE 几乎不变——因为 1×7 滤波器参数更多（为了保持输出通道数一致），容量更大但缺乏多尺度分解能力 [^src-mtgnn]。

## 相关页面

- [[mtgnn]] — MTGNN 模型上下文
- [[mix-hop-propagation-layer]] — 并行的图卷积模块
- [[graph-learning-layer]] — 上游图学习模块

[^src-mtgnn]: [[source-mtgnn]]
