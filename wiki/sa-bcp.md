---
title: "SA-BCP"
type: entity
tags:
  - conformal-prediction
  - bayesian
  - uncertainty-quantification
  - spatio-temporal-decoupling
  - online-learning
created: 2026-05-07
last_updated: 2026-05-07
source_count: 1
confidence: high
status: active
---

# SA-BCP

**State-Adaptive Bayesian Conformal Prediction (SA-BCP)** 是一种在线共形预测方法，由 Fang & Lee（台湾大学，2026 年 5 月）提出。通过将近期时间惯性（指数折扣）与历史模式记忆（核密度估计）解耦，解决了 ACI 类方法的系统性覆盖不足和 Bayesian CP 的结构滞后问题。[^src-sa-bcp]

---

## 架构：四步计算流程

### Step 1：时间密度计算

$$D_t^T(i) = \beta^{t-1-i}, \quad \forall i < t$$

$\beta \in (0,1)$ 是折扣因子。严格 $O(1)$ 的几何级数维护，无需遍历历史。[^src-sa-bcp]

### Step 2：空间密度计算

从近期轨迹提取局部空间状态 $S_t \in \mathbb{R}^d$（如过去 $d$ 天的滞后回报）。使用各向异性核密度估计（KDE）[Chen et al., 2024]：

$$D_t^S(i) = \exp\left(-\frac{1}{2}\sum_{j=1}^d \left(\frac{S_{t,j} - S_{i,j}}{h_j}\right)^2\right)$$

其中 $h_j$ 是通过在线 Scott 规则动态更新的带宽：

$$h_j = \hat{\sigma}_j \cdot N^{-1/(d+4)}$$

$\hat{\sigma}_j$ 通过 Welford 在线算法连续维护。总累积空间证据 $D_t^S = \sum_{i=0}^{t-1} D_t^S(i)$。[^src-sa-bcp]

### Step 3：认知置信门控

$$\pi_t^S = \frac{D_t^S}{D_t^S + K}, \quad \pi_t^T = 1 - \pi_t^S$$

其中 $K$ = 目标匹配数（唯一超参数，高可解释）：
- $D_t^S \ll K$（前所未见状态）→ $\pi_t^S \to 0$，保守回退时间基线，防噪声过拟合
- $D_t^S \ge K$（历史重现）→ 快速切���空间记忆，主动基于模式识别扩缩区间[^src-sa-bcp]

### Step 4：自适应分位数构造

空间 CDF：$\hat{F}_t^S(r) = \sum D_t^S(i) I(E_i \le r) / D_t^S$
时间 CDF：$\hat{F}_t^T(r) = \sum \beta^{t-1-i} I(E_i \le r) / \sum \beta^{t-1-i}$

冷启动保护：均匀先验 $\text{Unif}(0,R)$，$\lambda_t = 1/\sqrt{1+t}$：

$$\hat{F}_t(r) = (1-\lambda_t)(\pi_t^S \hat{F}_t^S(r) + \pi_t^T \hat{F}_t^T(r)) + \lambda_t \min(r/R, 1)$$

每步根求解 $\hat{F}_t(\hat{q}_t) = 1 - \alpha$ 得 $\hat{q}_t$。[^src-sa-bcp]

---

## 理论保证

### 定理 1：渐近边际有效性

**条件**：$P_t(E_t)$ Lipschitz 连续，$\sum d_{TV}(P_t, P_{t+1}) = o(T)$。

**结论**：对 $\forall \alpha \in (0,1), K > 0$：

$$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T I(E_t \le \hat{q}_t) = 1 - \alpha \quad \text{a.s.}$$

**证明**：等价于在线 pinball loss 梯度下降。累积遗憾 $O(\sqrt{T})$。覆盖误差构成有界鞅差序列，Azuma-Hoeffding 保证收敛。[^src-sa-bcp]

### 定理 2：动态遗憾界

**条件**：$D_t^S \to 0$（最坏情况）。

**结论**：退化为 FTRL。$\text{Regret}_T \le O(1/\sqrt{1-\beta}) + O(V_T \sqrt{1-\beta})$

$V_T = \sum |q_t^* - q_{t-1}^*|$ 是攻击性分布偏移的总变差。保障即使遭遇前所未见冲击也不出现无界覆盖崩溃。[^src-sa-bcp]

### 定理 3：Oracle 条件覆盖率

**条件**：零噪声 oracle，$D_t^S \to \infty$。

**结论**：$\pi_t^S \to 1$，$P(Y_t \in C_t(X_t) | S_t) = 1 - \alpha$

**渐进异常识别机制**：
正常状态 → $\pi_t^S \approx 1$（自信利用历史）
首次冲击 → 空间密度骤降 → $\pi_t^S \to 0$（回退时间基线）
冲击反复 → 密度积累 → $\pi_t^S$ 下降谷变浅 → 逐步"记住"罕见事件 → 在时间基线反应前就主动扩区间[^src-sa-bcp]

### 定理 4：最优时空解耦

**Lemma 5（空间 CDF 性质）**：$\hat{F}_t^S(r)$ 是 Nadaraya-Watson 核回归估计器。渐近无偏，方差 $\approx V_0/D_t^S$（$V_0$ = 不可约噪声方差）。

**Lemma 6（时间 CDF 性质）**：$\hat{F}_t^T(r)$ 等价于 EMA。当 $\beta \to 1$ 时方差 $\to 0$，但产生持久结构偏差 $B^T$，MSE $= (B^T)^2 \equiv M^T > 0$。

**定理 4**：$\text{MSE}(K) = (\pi_t^S)^2 V_0/D_t^S + (1-\pi_t^S)^2 M^T$

代入 $\pi_t^S = D_t^S/(D_t^S+K)$，求导得：

$$K^* = \frac{V_0}{M^T}$$

- $K < K^*$：过拟合局部噪声，高方差
- $K > K^*$：结构滞后，高偏差[^src-sa-bcp]

---

## 实证表现

### 渐进异常识别（合成实验）
交替序列 T=900：正常 $N(0,0.5^2)$ vs 冲击 $N(3.0,0.5^2)$，冲击 3 次（每次 30 步）。SA-BCP（$K=10$）对比 BCP（$\beta=0.99$），SA-BCP 间歇迅速恢复，BCP 缓慢回缩。[^src-sa-bcp]

### 跨数据集偏差-方差（2016-2026）

基���模型：Fast Online GARCH(1,1)。空间状态：5 步绝对对数回报轨迹。

| 资产 | $K^*$ | 曲线 | 解释 |
|------|-------|------|------|
| GBP/USD | 1.0 | 深抛物线 | 外汇市场稳定 + 剧烈断点，对解耦高度敏感 |
| AMD | 1.0 | 稳健 U 形 | 高 beta 个股特质波动，左陡右缓 |
| Gold | 1000.0 | 渐近 L 形 | 慢周期大宗商品，时间惯性巨大，几乎完全依赖时间基线 |

$K$ 是资产"状态-噪声比"的物理表示。[^src-sa-bcp]

### 主基准（2016-2026，$\alpha \in \{0.3, 0.2, 0.1\}$）

基线：AgACI [Zaffran et al., 2022]、DtACI [Gibbs & Candès, 2024]、BCP [Zhang et al., 2024]

| 指标 | SA-BCP vs ACI | SA-BCP vs BCP |
|------|--------------|--------------|
| 覆盖校准 | 90% 目标 91%+ vs 86-87%（DtACI 系统性不足） | 近完美校准 vs 过度覆盖 |
| 区间宽度 | 可比或更优 | AMD -10.5%, Gold -27.7%, GBP -37.4% |
| Winkler 最优 | 9 中 8 最优 | 全部降低 |

**计算**：每步 $O(1)$。KDE 带宽通过 Welford 在线算法维护。[^src-sa-bcp]

---

## 局限与未来方向

- $K$ 固定 → 提出在线元共形预测（Meta-CP）：$K_t$ 动态调整为二级在线学习任务
- 扩展至多视野预测
- 高频流环境中的主动风险管理[^src-sa-bcp]

## 与在线学习的关系

- 时间基线 → FTRL 算法（享强凸正则化）
- 空间密度 → Nadaraya-Watson 核回归
- 门控机制 → 认知不确定性估计器
- 总框架 → 在线梯度下降 + 核方法 + 贝叶斯先验 的融合

## 关联页面

- [[spatio-temporal-decoupling]] — 核心概念：时空解耦在 CP 中的最优权衡
- [[conformal-prediction]] — 共形预测基础框架
- [[bayesian-conformal-prediction]] — Bayesian CP 基线，时间折扣方法
- [[adaptive-conformal-inference]] — ACI 基线，纯反馈方法

[^src-sa-bcp]: [[source-sa-bcp]]
