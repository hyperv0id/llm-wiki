---
title: "Confidence-Gated Distillation"
type: technique
tags:
  - knowledge-distillation
  - confidence-gating
  - retrieval-augmented
  - time-series-forecasting
  - selective-distillation
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: medium
status: active
---

# Confidence-Gated Distillation

**置信门控蒸馏**（Confidence-Gated Distillation）是 [[ts-memory|TS-Memory]]（KDD 2026）提出的训练技术，通过 advantage gate 和置信加权机制选择性蒸馏检索教师的分布校正，仅在检索提供可靠改善信号时传递知识 [^src-ts-memory-time-series-foundation-models-kdd26]。

## 动机

离线 kNN 检索并非总优于冻结 backbone——当检索邻居与查询上下文不匹配或检索语料与部署域不一致时，教师目标可能引入噪声 [^src-ts-memory-time-series-foundation-models-kdd26]。无条件蒸馏会传递噪声信号，导致负迁移（negative transfer）[^src-ts-memory-time-series-foundation-models-kdd26]。

## 机制

### Step 1 — 优势门控（Advantage Gate）

计算教师和 backbone 的中位数绝对误差 [^src-ts-memory-time-series-foundation-models-kdd26]：

$$\text{err}_t^* = \frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}} |Y_{t,u} - \hat{Q}_{t,j^*,u}^*|, \quad * \in \{T, \text{base}\}$$

其中 $j^* = \arg\min_j |q_j - 0.5|$ 为中位数索引。门控指标 [^src-ts-memory-time-series-foundation-models-kdd26]：

$$\chi_t = \mathbb{I}[\text{err}_t^T + \epsilon_{\text{gate}} < \text{err}_t^{\text{base}}], \quad \omega_t = \chi_t \cdot \text{Conf}_t^\gamma$$

$\chi_t = 1$ 仅当教师严格优于 backbone（含 margin $\epsilon_{\text{gate}}$），$\text{Conf}_t = \max_k w_k$ 为检索置信度，$\gamma$ 为置信缩放超参 [^src-ts-memory-time-series-foundation-models-kdd26]。

### Step 2 — 增量对齐（Incremental Alignment）

仅对通过门控的窗口蒸馏教师分位数，使用 Huber loss [^src-ts-memory-time-series-foundation-models-kdd26]：

$$D_Q(t) = \frac{1}{Q|\mathcal{U}|}\sum_{j,u} \ell_\kappa(\hat{Q}_{t,j,u}^{\text{mem}}, \hat{Q}_{t,j,u}^{\text{teach}})$$

附加 $\Delta$-alignment 对齐教师与记忆模块相对于 backbone 的中位数校正（而非全局偏差），鼓励学习域级时间校正模式 [^src-ts-memory-time-series-foundation-models-kdd26]。

### 稳定性正则

对低置信窗口（$\omega_t$ 小），锚定 PlugMem 中位数至 backbone $(1-\omega_t)$ 权重，防止过度校正 [^src-ts-memory-time-series-foundation-models-kdd26]。另加分位数交叉惩罚保证单调性 [^src-ts-memory-time-series-foundation-models-kdd26]。

## 门控统计

论文报告教师优于 backbone 的训练窗口占比 83.5%，但门控并非始终开启 [^src-ts-memory-time-series-foundation-models-kdd26]：

| 数据集 | 激活率 $\chi_t=1$ | 平均 $\omega_t$ | 平均 Conf | 优势 margin |
|--------|-----------------|---------------|-----------|------------|
| ETT Avg | 87.7% | 0.479 | 0.562 | 0.173 |
| Weather | 96.7% | 0.680 | 0.701 | 0.245 |
| Traffic | 68.1% | 0.067 | 0.100 | 0.025 |
| Electricity | 50.8% | 0.062 | 0.113 | 0.013 |
| Exchange | 99.4% | 0.102 | 0.102 | 0.184 |

在已校准数据集（Electricity、Traffic）上门控激活率和权重低，在漂移域（Weather、Exchange）上高——门控起到域自适应过滤器作用 [^src-ts-memory-time-series-foundation-models-kdd26]。

## 与其他选择性蒸馏的关系

- **标准知识蒸馏**：无门控，无条件传递教师信号
- **[[temporal-semantic-primitives|TESS 置信门控]]**：TESS 用置信门控选择是否将时序原语注入 backbone prefix，门控对象是离散原语而非连续分位数 [^src-ts-memory-time-series-foundation-models-kdd26]
- **选择性蒸馏**：NLP 中已有基于置信度选择教师信号的工作；TS-Memory 的创新在于将检索置信度与优势门控结合，实现双重过滤 [^src-ts-memory-time-series-foundation-models-kdd26]

## 消融证据

移除优势门控或置信加权均导致一致性能退化，证实选择性和置信条件化蒸馏至关重要——过滤不可靠检索目标，防止负迁移 [^src-ts-memory-time-series-foundation-models-kdd26]。

[^src-ts-memory-time-series-foundation-models-kdd26]: [[source-ts-memory-time-series-foundation-models-kdd26]]
