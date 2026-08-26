---
title: "反馈扩散引导"
type: technique
tags:
  - diffusion-models
  - dynamic-guidance
  - classifier-free-guidance
  - posterior-likelihood
  - aaai-2026
created: 2026-06-08
last_updated: 2026-08-26
source_count: 1
confidence: medium
status: active
---

# 反馈扩散引导 (Feedback Diffusion Guidance)

**反馈扩散引导**是一种动态调整扩散模型引导尺度的技术，由 [[fence|FENCE]]（AAAI 2026）首次引入时空插补领域[^src-fence]。与 [[classifier-free-guidance|无分类器引导]]（CFG）使用固定超参数 $\lambda$ 不同，反馈引导将引导尺度 $\lambda(x_k, k)$ 作为去噪步 $k$ 和当前样本 $x_k$ 的函数，通过后验似然 $p(c|x_k)$ 的近似估计动态调整引导强度[^src-fence]。

## 核心原理

### 加性误差假设

反馈引导的理论基础是加性误差假设（Koulischer et al., 2025）：学习的条件分布 $p_{\theta,k}(x_k|c)$ 是真实条件分布 $p_k(x_k|c)$ 和真实无条件分布 $p_k(x_k)$ 的线性组合：

$$p_{\theta,k}(x_k|c) = (1-\pi)p_k(x_k) + \pi p_k(x_k|c)$$

其中 $\pi \in [0,1]$ 表示对条件模型学习效果的事先置信度[^src-fence]。通过推导，引导尺度可表达为后验似然的函数：

$$\lambda(x_k, k) \approx \frac{p_{\theta,k}(c|x_k)}{p_{\theta,k}(c|x_k) - (1-\pi)}$$

### 引导尺度行为

- 当 $p(c|x_k)$ 高（生成值与条件观测一致）→ $\lambda \to 1$，避免过校正
- 当 $p(c|x_k)$ 降低（生成值偏离条件）→ $\lambda$ 增大，加强引导以拉回条件分布
- 当 $p(c|x_k)$ 接近 $(1-\pi)$ → $\lambda$ 急剧增大，强制遵循条件[^src-fence]

### 后验似然更新

后验 $p(c|x_k)$ 无法直接获取，通过追踪扩散反向马尔可夫链迭代更新[^src-fence]：

$$\log p_{\theta,k-1}(c|x_{k-1}) = \log p_{\theta,k}(c|x_k) + \log p_\theta(x_{k-1}|x_k, c) - \log p_\theta(x_{k-1}|x_k)$$

由于反向转移分布是高斯分布，对数似然差可简化为条件与无条件模型预测均值的 L2 距离差[^src-fence]。

## 与 CFG 的对比

| 特性 | CFG | 反馈引导 |
|------|-----|---------|
| 引导尺度 | 固定超参数 $\lambda$ | 动态函数 $\lambda(x_k, k)$ |
| 调整依据 | 无（手动调参） | 后验似然 $p(c|x_k)$ |
| 空间适应性 | 全局统一 | 支持聚类级差异化 |
| 高缺失率鲁棒性 | 差（漂移到先验） | 好（动态增强引导） |

## 应用

- [[fence|FENCE]]：将反馈引导首次应用于时空交通数据扩散插补，结合聚类感知机制实现节点级定制化引导[^src-fence]

[^src-fence]: [[source-fence]]