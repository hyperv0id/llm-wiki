---
title: "Conformal Prediction"
type: concept
tags:
  - conformal-prediction
  - uncertainty-quantification
  - distribution-free
created: 2026-05-07
last_updated: 2026-05-07
source_count: 2
confidence: medium
status: active
---

# Conformal Prediction

**共形预测（Conformal Prediction, CP）** 是一种与模型无关的不确定性量化框架，在可交换性假设下提供有限样本的覆盖保证。[^src-sa-bcp][^src-scale]

## 基本设定

对于预测问题，CP 的目标是构建预测区间 $C_t(X_t) = [\hat{f}(X_t) - \hat{q}_t, \hat{f}(X_t) + \hat{q}_t]$，使得 $P(Y_t \in C_t(X_t)) \ge 1 - \alpha$。非一致性评分通常取绝对残差 $E_t = |Y_t - \hat{f}(X_t)|$。[^src-sa-bcp]

## 可交换性挑战

标准 CP 要求数据满足可交换性——联合分布在置换下不变。时间序列中的时序依赖和分布漂移频繁违反此假设，图结构 MTS 中的跨节点耦合进一步加剧了此问题。[^src-scale]

### 应对策略

| 方法 | 策略 | 来源 |
|------|------|------|
| ACI | 在线误差反馈调整 | Gibbs & Candes, 2021 |
| Bayesian CP | 时间折扣历史权重 | Zhang et al., 2024 |
| **SA-BCP** | 时空解耦 + 状态自适应 | Fang & Lee, 2026 |
| **SCALE** | 谱图条件可交换性 + 小波分解 | Guo et al., ICML 2026 |

## SA-BCP 方法

通过将时间基线惯性（指数折扣）与空间状态记忆（KDE）解耦，使用可解释的证据门控 $\pi_t^S = D_t^S / (D_t^S + K)$ 动态融合两者。解决了 ACI 的系统性覆盖不足和 Bayesian CP 的结构滞后。[^src-sa-bcp]

## SCALE 方法

通过谱图小波变换将残差分解为低频和高频分量。高频分量在条件化于低频分量后近似可交换，从而在谱域实现有效的共形预测。[^src-scale]

[^src-sa-bcp]: [[source-sa-bcp]]
[^src-scale]: [[source-scale]]
