---
title: "Sliding-Window Diffusion"
type: technique
tags:
  - diffusion-models
  - time-series
  - forecasting
  - generalized-diffusion
  - sliding-window
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Sliding-Window Diffusion

**Sliding-Window Diffusion**（滑动窗口扩散）是 [[armd|ARMD]] 提出的中间状态生成机制：用**沿时间轴滑动序列窗口**取代传统扩散的逐步加噪，从而把"历史序列→未来序列"的演化直接构造成一条确定性扩散链[^src-armd]。

## 定义

设未来序列为扩散初始状态 $X^0_{1:T}$、历史序列为最终状态 $X^T_{-T+1:0}$（历史长度 = 预测长度 = $T$）。第 $t$ 步中间态由未来序列朝历史方向滑动 $t$ 步得到[^src-armd]：

$$X^t_{1-t:T-t}=\mathrm{Slide}(X^0_{1:T}, t),$$

其中 $\mathrm{Slide}(X,k)$ 表序列窗口朝历史方向移动 $k$ 步。单步 $X^{t-1}_{2-t:T-t+1}\to X^t_{1-t:T-t}=\mathrm{Slide}(X^{t-1}_{2-t:T-t+1},1)$ 类比 DDPM 的 $q$ 过程[^src-armd]。

## 与噪声扩散的对应

借用 DDPM 的闭式前向公式，滑动可写成与加噪同构的形式[^src-armd]：

$$X^t_{1-t:T-t}=\sqrt{\bar\alpha_t}\,X^0_{1:T}+\sqrt{1-\bar\alpha_t}\,z^t,$$

其中**演化趋势** (evolution trend) $z^t$ 占据 DDPM 中"高斯噪声"的位置，但因每个中间态都是确定的真实子序列，$z^t$ 可被闭式解出 $z^t=\big(\tfrac{1}{\sqrt{\bar\alpha_t}}X^t_{1-t:T-t}-X^0_{1:T}\big)/\sqrt{\tfrac{1}{\bar\alpha_t}-1}$，作为训练的 ground truth[^src-armd]。最大扩散步数 $T$ 因此等于待预测序列长度[^src-armd]。

## 为什么有效

- **确定性**：滑动消除了加噪引入的随机性，使训练目标 $z^t$ 有解析真值，并允许采样时移除噪声项 $\sigma_t\epsilon_t$[^src-armd]。
- **保持序列连续性**：每个中间态都是真实时间序列的一个连续子窗口，保留了序列演化中的中间信息——这是噪声扩散把真实序列毁成白噪声所丢失的[^src-armd]。
- **目标对齐**：采样从历史序列出发逐步变换为未来序列，扩散采样过程本身即预测过程，无需条件生成[^src-armd]。

## 消融证据

ARMD 对比了用线性**插值**生成中间态（$X^t_{1-t:T-t}=X^0_{1:T}+(X^T_{-T+1:0}-X^0_{1:T})\cdot t/T$）的方案；滑动法在全部 7 个数据集上一致优于插值法，印证其在保持序列连续性上的优势[^src-armd]。

## 与广义扩散的关系

滑动窗口扩散属于"用退化/中间态替代噪声"的广义扩散家族，与 [[cold-sampling|Cold Diffusion]]（任意退化算子）、[[dyffusion|DYffusion]]（时间插值作为扩散步）思想相通；区别在于 ARMD 的退化算子是时间轴上的窗口平移，直接把历史与未来嵌入同一扩散链[^src-armd]。

[^src-armd]: [[source-armd]]
