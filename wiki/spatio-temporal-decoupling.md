---
title: "Spatio-Temporal Decoupling in Conformal Prediction"
type: concept
tags:
  - conformal-prediction
  - uncertainty-quantification
  - time-series
  - spatio-temporal
  - bias-variance-tradeoff
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: high
status: active
---

# Spatio-Temporal Decoupling in Conformal Prediction

时空解耦是在线共形预测中的一种设计理念，核心思想是将预测区间的调整分解为**时间维度**（近期惯性）和**空间维度**（历史模式记忆）两个独立通道，通过可解释的机制动态融合它们。由 SA-BCP 首次形式化并证明其最优性。[^src-sa-bcp]

---

## 问题背景

在线共形预测面临核心困境：

| 方法类 | 代表 | 优势 | 致命缺陷 |
|--------|------|------|---------|
| 纯时间（反馈驱动） | ACI, AgACI, DtACI | 对近期变化敏感 | 系统性覆盖不足，突变期区间方差高 |
| 纯时间（折扣） | Bayesian CP | 最坏情况保护 | 结构滞后，冲击过后区间过宽不回缩 |
| 时空解耦 | SA-BCP | 兼顾两者 | 需调优 $K$ |

核心洞见：仅依赖时间维度忽视了局部状态依赖关系。看似前所未有的突变往往是历史状态的**重现**。[^src-sa-bcp]

---

## 形式化定义

### 时间基线（Recent Inertia）

指数折扣密度，捕获近期波动状态：

$$D_t^T(i) = \beta^{t-1-i}, \quad \forall i < t$$

等价于在线学习的折扣信念机制，也等价于 FTRL 算法。[^src-sa-bcp]

### 空间密度（Pattern Memory）

通过各向异性 KDE 衡量当前状态与历史状态的相似度：

$$D_t^S(i) = \exp\left(-\frac{1}{2}\sum_{j=1}^d \frac{(S_{t,j} - S_{i,j})^2}{h_j^2}\right)$$

带宽 $h_j = \hat{\sigma}_j \cdot N^{-1/(d+4)}$（在线 Scott 规则，通过 Welford 算法维护）。[^src-sa-bcp]

### 门控机制

$$\pi_t^S = \frac{D_t^S}{D_t^S + K}$$

$K$ = 目标匹配数，控制时空耦合强度。这是一个**认知置信门控**——$K$ 充当"安全阈值"防止对虚假噪声过拟合。[^src-sa-bcp]

---

## 最优权衡（定理 4）

设 $V_0$ = 空间特征不可约方差，$M^T$ = 时间基线结构偏差的平方。

$$\text{MSE}(K) = \left(\frac{D_t^S}{D_t^S + K}\right)^2 \frac{V_0}{D_t^S} + \left(\frac{K}{D_t^S + K}\right)^2 M^T$$

由于：
- **空间分量**：Nadaraya-Watson 核回归估计器（Lemma 5），渐近无偏，方差 $\propto V_0/D_t^S$
- **时间分量**：指数移动平均（Lemma 6），方差 $\to 0$（$\beta \to 1$），但持结构偏差 $B^T$，MSE $\approx M^T$

求导得最优 $K^* = V_0 / M^T$。

| 情景 | 含义 | 后果 |
|------|------|------|
| $K \ll K^*$ | 空间主导 | 过拟合局部噪声，区间爆炸（高方差） |
| $K \gg K^*$ | 时间主导 | 结构滞后，不能主动扩区间（高偏差） |
| $K = K^*$ | 最优平衡 | minimax 最优 |

$K$ 不仅是超参数，它是资产"状态-噪声比"的物理表示。[^src-sa-bcp]

---

## 自适应行为

解耦框架产生一种**渐进异常识别**的涌现行为：

1. **正常状态**：频繁出现 → 空间密度高 → $\pi_t^S \approx 1$ → 自信利用历史记忆
2. **首次冲击**：前所未见 → 空间密度骤降 → $\pi_t^S \to 0$ → 回退到时间防御
3. **冲击持续**：两种方法都拉伸适配
4. **冲击结束**：空间密度立即识别熟悉的正常状态 → 瞬间恢复 $\pi_t^S \approx 1$ → 区间弹回最优宽度
5. **冲击反复**：累积密度增长 → $\pi_t^S$ 下降谷变浅 → 提前主动扩缩区间 → 达到 Oracle 条件覆盖率（定理 3）

对比：纯时间折扣方法（BCP）在步骤 4 因受污染的记忆缓慢回缩，导致未校准区间膨胀 10-37%。[^src-sa-bcp]

---

## 与偏差-方差权衡的关系

时空解耦从信息论角度重新诠释了偏差-方差权衡：
- **时间基线**提供低方差但高偏差的估计
- **空间记忆**提供零偏差但高方差的估计（取决于匹配数量）
- 门控机制 $K$ 是决定两者权重的"温度调节器"

这对应于经典统计学习中的模型选择偏差-方差权衡——发生在在线共形预测的语境中。[^src-sa-bcp]

---

## 关联方法与页面

- [[sa-bcp]] — 时空解耦的具体实现（Fang & Lee, 2026）
- [[conformal-prediction]] — 共形预测基础框架
- [[bayesian-conformal-prediction]] — 纯时间基线方法的结构滞���
- [[adaptive-conformal-inference]] — 纯反馈方法的覆盖不足

[^src-sa-bcp]: [[source-sa-bcp]]
