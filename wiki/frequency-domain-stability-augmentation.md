---
title: "Frequency Domain Stability Augmentation"
type: technique
tags:
  - frequency-domain
  - data-augmentation
  - traffic-forecasting
  - pre-training
created: 2026-07-16
last_updated: 2026-07-16
source_count: 1
confidence: medium
status: active
---

# Frequency Domain Stability Augmentation

**Frequency Domain Stability Augmentation (FDA)** 是 [[minitraffic|MiniTraffic]] 中提出的频域数据增强技术，通过在频域对道路级交通信号施加有界扰动，模拟车道级变异性，同时保持全局频谱结构不变[^src-minitraffic]。

## 动机

道路-车道之间存在结构性关联（交通流守恒定律），且频域分析显示道路速度和对应车道速度的频谱高度重叠[^src-minitraffic]。FDA 利用这一特性，从丰富的道路数据生成伪车道模式，缓解车道级标注稀缺对预训练的限制。

## 机制

1. **DFT 变换**：将道路级状态 $X^R \in \mathbb{R}^{N \times T}$ 经矩阵形式 DFT 映射到频域：$\tilde{X}^R = X^R \cdot F_T$，每个频率系数分解为幅值 $A(f)$ 和相位 $\theta(f)$[^src-minitraffic]。

2. **有界扰动**：注入高斯噪声 $\delta_A \sim \mathcal{N}(0, \sigma_A^2)$、$\delta_\theta \sim \mathcal{N}(0, \sigma_\theta^2)$，受两个约束限制[^src-minitraffic]：
   - **幅值约束**：$|\delta_A(f)| \leq \epsilon(f) = \lambda \cdot \max A(f)$，$\lambda \in (0,1)$
   - **选择性掩码**：$\Gamma(f) = \mathbb{I}(A(f)^2 > \tau \cdot \max_{f'} A(f')^2)$，仅扰动主导频带

3. **能量稳定性**：相对频谱能量偏移 $\lesssim \lambda^2 \cdot \frac{\sum_f \Gamma(f)A(f)^2}{\sum_f A(f)^2}$，Parseval 恒等式保证了扰动后信号结构完整性[^src-minitraffic]。

4. **逆变换**：$\tilde{X}^R_d = \Re(\tilde{X}^R_d(f) \cdot F_T^{-1})$，仅取实部，经 Adaptive Head（轻量 MLP）投影至主干网络输入维度[^src-minitraffic]。

## 理论依据

扰动约束确保增强样本保持在原始信号流形的 Lipschitz 有界邻域内，不引入标签噪声；中等 $\lambda$ 在增加数据多样性和保持信号忠实性之间取得平衡[^src-minitraffic]。

## 与其他增强方法对比

不同于 FRAug（通用频域增强，仅用于扩增数据），FDA 的扰动设计针对道路→车道迁移任务，频率约束保证迁移过程中全局趋势不变[^src-minitraffic]。

[^src-minitraffic]: [[source-minitraffic]]
