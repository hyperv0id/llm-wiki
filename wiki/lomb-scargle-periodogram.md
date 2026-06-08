---
title: "Lomb–Scargle Periodogram"
type: concept
tags:
  - frequency-domain
  - signal-processing
  - irregular-sampling
  - spectral-analysis
  - time-series
created: 2026-06-08
last_updated: 2026-06-08
source_count: 1
confidence: medium
status: active
---

# Lomb–Scargle Periodogram

**Lomb–Scargle 周期图**是一种从不均匀采样数据中估计功率谱密度（PSD）的方法，最初由 Lomb (1976) 和 Scargle (1982) 提出，主要用于天文学中分析非均匀间隔的观测数据[^src-lscd]。与 FFT 不同，Lomb–Scargle **不需要**数据在均匀网格上采样，因此避免了缺失值插值或零填充带来的频谱失真[^src-lscd]。

## 数学原理

给定非均匀采样信号 $x = (x_{t_1}, \ldots, x_{t_N})$，在时间点 $t_1, \ldots, t_N$ 上观测。对每个候选频率 $f$，Lomb–Scargle 通过最小二乘拟合正弦波来估计该频率的功率[^src-lscd]：

$$x_{t_n} = A \cos(2\pi f t_n + \phi) + \epsilon_{t_n}$$

其中 $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ 为高斯白噪声。等价地可表示为线性模型：

$$x_{t_n} = \alpha_1 \cos(2\pi f t_n) + \alpha_2 \sin(2\pi f t_n) + \epsilon_{t_n}$$

每个频率 $f$ 的功率估计 $P(f)$ 通过最小二乘解得到[^src-lscd]：

$$P(f) = \frac{1}{2}\left[\frac{(\sum_n x_{t_n} \cos(2\pi f(t_n - \tau)))^2}{\sum_n \cos^2(2\pi f(t_n - \tau))} + \frac{(\sum_n x_{t_n} \sin(2\pi f(t_n - \tau)))^2}{\sum_n \sin^2(2\pi f(t_n - \tau))}\right]$$

其中 $\tau$ 是使周期图平移不变的时移参数[^src-lscd]：

$$\tau = \frac{1}{4\pi f} \tan^{-1}\left(\frac{\sum_n \sin(4\pi f t_n)}{\sum_n \cos(4\pi f t_n)}\right)$$

对一组候选频率 $\{f_1, \ldots, f_K\}$ 重复计算得到完整的周期图。

## False Alarm Probability (FAP)

在高斯噪声假设下，$P(f)$ 服从自由度为 2 的 $\chi^2$ 分布[^src-lscd]。这允许估计每个频率的虚假警报概率（FAP）：

$$P_{\text{FA}}(\omega) = 1 - [1 - \exp(-P_f(\omega))]^{J_{\text{eff}}}$$

FAP 可以用于过滤周期图中的虚假频率分量——在 [[lscd|LSCD]] 中，权重函数 $w(\omega_k) = 1/(P_{\text{FA}}(\omega_k) + \epsilon)$ 被用于增强真实频率的影响[^src-lscd]。

## 与 FFT 的对比

| 特性 | FFT | Lomb–Scargle |
|------|-----|-------------|
| 采样要求 | 必须均匀采样 | 支持非均匀采样 |
| 缺失值处理 | 需插值/零填充 | 直接处理 |
| 高频缺失率鲁棒性 | 差（频谱严重扭曲） | 好 |
| 计算 | $O(N \log N)$ | $O(N \cdot K)$ |
| 统计基础 | 确定性变换 | 最小二乘 + $\chi^2$ 检验 |

在 [[lscd|LSCD 论文]]中，75% 随机缺失的合成正弦波数据显示：FFT + 线性插值产生偏移或虚假的频谱峰值，而 Lomb–Scargle 的频谱分布与真实值保持良好对齐[^src-lscd]。

## 在机器学习中的应用

尽管 Lomb–Scargle 在天文学和信号处理中已有广泛应用，在机器学习中仍属未充分探索的方法[^src-lscd]。[[lscd|LSCD]] (ICML 2025) 首次将其以**可微层**的形式集成到扩散模型中：

1. **作为条件信号**：Lomb–Scargle 周期图经频谱编码器处理后注入扩散去噪网络，为时间域重建提供频率引导[^src-lscd]
2. **作为损失函数**：频谱一致性损失 $L_{\text{SCons}}$ 比较观测信号与重建信号的 Lomb–Scargle 周期图，强制频谱对齐[^src-lscd]

## 局限性

- **正弦假设**：Lomb–Scargle 假设信号可分解为正弦波之和——对于非周期性或高度非线性的时间序列，这一假设可能不准确[^src-lscd]
- **计算成本**：每个候选频率需要独立的最小二乘拟合，对于大量频率的细粒度频谱分析可能存在计算瓶颈
- **白噪声假设**：$\chi^2$ 分布和 FAP 的有效性依赖高斯白噪声假设

## 关联页面

- [[lscd]] — LSCD，首次将 Lomb–Scargle 引入扩散模型
- [[source-lscd]] — LSCD 源文件
- [[frequency-aware-conditioning]] — 频率感知条件化范式
- [[spectral-consistency-loss]] — 频谱一致性损失
- [[csdi]] — CSDI，传统 FFT 依赖的插补方法
- [[diffusion-model]] — 扩散模型

[^src-lscd]: [[source-lscd]]
