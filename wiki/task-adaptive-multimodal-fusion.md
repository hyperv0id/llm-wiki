---
title: "Task-adaptive Multi-modal Fusion（TMF）"
type: technique
tags:
  - time-series
  - anomaly-detection
  - multimodal-fusion
  - mixture-of-experts
  - routing
created: 2026-11-02
last_updated: 2026-11-02
source_count: 1
confidence: low
status: active
---

# Task-adaptive Multi-modal Fusion（TMF）

任务自适应多模态融合（Task-adaptive Multi-modal Fusion, TMF）是 [[vetime|VETime]] 框架中协调异常检测主任务与序列数值重构辅助任务的多模态动态门控路由机制（arXiv:2602.16681，§3.4）[^src-2602-16681-vetime]。

## 动机与任务失配分析

在端到端多任务学习中，异常检测与辅助任务之间存在固有的特征偏好冲突：
1. **主任务与辅助任务的梯度冲突**：下游异常检测（AD）分类任务偏好高度抽象且具有强区分度的判别性特征；而辅助序列重构（Rec）任务偏好底层连续的局部数值细节。静态融合会导致两任务的优化梯度互相干扰（§3.4）[^src-2602-16681-vetime]。
2. **静态加权的局限性**：简单的特征拼接（Concatenation）或固定加和（Addition）无法针对时间戳局部的动态异常特性调节模态注意力，容易导致多模态交互失效或偏向单一模态 [^src-2602-16681-vetime]。
3. **TMF 的设计定位**：TMF 将时序特征 $F_{TS}$、对齐视觉特征 $F_V$ 与 [[anomaly-window-contrastive-learning|AWCL]] 提取的异常增强特征 $F_A$ 视作专家候选空间 $\mathcal{M} = \{TS, V, A\}$，通过动态门控网络为不同下游任务分配补丁级路由权重，辅以辅助序列重构约束防止过拟合，并引入负熵正则化防止路由过早塌缩（§3.4）[^src-2602-16681-vetime]。

## 机制流程与形式化

### 1. 动态权重路由网络（Dynamic Weight Routing）

定义候选专家集合 $\mathcal{M} = \{TS, V, A\}$，对应特征矩阵为 $F_{TS}, F_V, F_A \in \mathbb{R}^{N_{TS} \times D}$。路由网络以异常增强表征 $F_A \in \mathbb{R}^{N_{TS} \times D}$ 为输入，为异常检测（AD）与序列重构（Rec）两个任务独立生成 patch 级的专家权重（公式 4）[^src-2602-16681-vetime]：
$$W^t = \operatorname{Softmax}(\operatorname{MLP}(F_A) + b^t) \in \mathbb{R}^{N_{TS} \times 3}, \quad t \in \{AD, Rec\}$$
其中 $b^t \in \mathbb{R}^3$ 为可学习的任务特异偏置向量，$\operatorname{Softmax}$ 沿专家维度归一化以保证概率和为 1。

### 2. 任务特化特征融合（Task-Adaptive Fusion）

依据路由网络输出的权重向量，融合模块对三类专家特征执行加权求和，分别生成适配两项任务的特化表征（公式 5）[^src-2602-16681-vetime]：
$$F_t = \sum_{m \in \mathcal{M}} W_m^t \odot F_m \in \mathbb{R}^{N_{TS} \times 512}, \quad t \in \{AD, Rec\}$$
其中 $W_m^t \in \mathbb{R}^{N_{TS} \times 1}$ 为针对专家 $m$ 沿特征维度广播的权重向量，$\odot$ 为逐元素相乘。
融合后的 $F_{AD}$ 与 $F_{Rec}$ 经 Token 线性投影层恢复至原始序列长度 $L$，分别送入分类头 $\text{MLP}_{AD}$ 与重构头 $\text{MLP}_{Rec}$ 输出预测结果。辅助重构任务强制网络保持时序数值连续性，促进深度跨模态交互并防止过拟合极度稀疏的异常标签（§3.4）[^src-2602-16681-vetime]。

### 3. 专家负载均衡负熵正则化（Routing Entropy Regularization）

为避免局部门控网络在训练初期过快塌缩至单一模态，TMF 引入基于负熵最小化的负载均衡正则项（§3.4）[^src-2602-16681-vetime]。
首先计算各专家在全序列时间轴上的平均利用率：
$$\hat{w}_m^t = \frac{1}{N_{TS}} \sum_{i=1}^{N_{TS}} w_{i, m}^t$$
其中 $w_{i, m}^t$ 为任务 $t$ 在补丁 $i$ 上分配给专家 $m$ 的概率。通过最小化负熵（最大化全局专家利用多样性）：
$$\mathcal{L}_e = \sum_{t \in \{AD, Rec\}} \sum_{m \in \mathcal{M}} \hat{w}_m^t \log \hat{w}_m^t$$
该正则项平衡了全局层面的模态资源分配，同时允许局部 patch 依据上下文需求自由进行动态门控。

## 超参数与敏感度分析

- **默认超参**：负熵正则化损失权重设定为 $\lambda_e = 0.2$。
- **敏感度验证（§4.3, 图 6(b)）**：超参数敏感度实验显示，$\lambda_e$ 需维持在 $\ge 0.2$；当 $\lambda_e < 0.2$ 时，门控网络缺乏足够的负载均衡约束，专家多样性不足，易导致模态塌缩并削弱整体判别力。

## 门控权重可视化分工证据

在单变量数据集上的门控权重分布可视化显示出清晰的任务分工解耦（§4.3, 图 7）[^src-2602-16681-vetime]：
- **异常检测任务（AD）**：异常增强特征 $F_A$ 占据主导路由权重（图 7(a)），使分类头能够充分利用判别性高的高层抽象线索以最大化区分度；
- **序列重构任务（Rec）**：时序特征 $F_{TS}$ 占据主导路由权重（图 7(b)），以保留细粒度数值连续性；
- 这种自适应任务分工策略有效隔离了高层语义与底层数值精度，验证了动态路由在缓解任务冲突上的有效性。

## 实证证据与消融分析

### 1. 融合策略对比（附录 D.6, 表 18）
在微观消融实验中，TMF 的动态路由机制显著优于传统静态融合策略（表 18）[^src-2602-16681-vetime]：

| 融合策略 | NAB (Aff-F1 / VUS-PR) | WSD (Aff-F1 / VUS-PR) | YAHOO (Aff-F1 / VUS-PR) | 出处 |
|---|---|---|---|---|
| 简单拼接（Concatenation） | 85.07% / 28.54% | 89.80% / 26.39% | 95.69% / 85.72% | 表 18 |
| 简单相加（Addition） | 83.99% / 29.52% | 90.11% / 26.30% | 96.86% / 85.31% | 表 18 |
| TMF 动态路由（完整模型） | **88.57% / 32.98%** | **94.31% / 33.85%** | **97.15% / 88.61%** | 表 18 |

结果表明，简单的线性拼接或相加无法应对复杂的多模态时序模式，而 TMF 的自适应门控大幅提升了跨域泛化能力。

### 2. 辅助重构头消融（附录 D.6, 表 18）
移除辅助序列重构头（w/o reconstruction head）后，模型在各数据集上均出现性能回退（表 18）[^src-2602-16681-vetime]：
- NAB 数据集：Affiliation-F1 从 88.57% 降至 86.96%，VUS-PR 从 32.98% 降至 28.44%；
- WSD 数据集：Affiliation-F1 从 94.31% 降至 90.87%，VUS-PR 从 33.85% 降至 28.63%；
- YAHOO 数据集：VUS-PR 从 88.61% 降至 86.58%。
证实序列重构任务作为辅助监督，能促使模型学习更加鲁棒的底层时序表征。

### 3. 联合优化目标交互消融（附录 D.4, 表 16）
实验表明，直接同时引入 $\mathcal{L}_{\text{BCE}}$ 与 $\mathcal{L}_{\text{MSE}}$ 会引发细微冲突（NAB VUS-PR 从 27.75% 降至 27.64%）；而引入路由负熵正则 $\mathcal{L}_e$ 能够有效调和两者的梯度矛盾，使完整多目标优化在 NAB 上达到 32.98% 的峰值性能（表 16）[^src-2602-16681-vetime]。

[^src-2602-16681-vetime]: [[source-2602-16681-vetime]]
