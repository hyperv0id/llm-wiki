---
title: "GMF (Geometry-based Multimodal Fusion)"
type: entity
tags:
  - multimodal-fusion
  - schrodinger-bridge
  - rectified-flow
  - reliability
  - trustworthy-ai
created: 2026-07-18
last_updated: 2026-07-18
source_count: 1
confidence: medium
status: active
---

# GMF (Geometry-based Multimodal Fusion)

GMF 是一种基于[[transport-based-reliability-assessment|潜在空间传输几何]]的多模态融合框架，将可靠性评估从分类器输出的[[circular-dependency-in-multimodal-fusion|循环依赖]]中解耦出来。不同于依赖预测置信度或 evidential 不确定性的统计方法，GMF 通过 [[schrodinger-bridge|Diffusion Schrödinger Bridge]] / [[rectified-flow|Rectified Flow]] 在潜在空间中估计传输代价，以此作为与分类器独立的外在可靠性信号[^src-gmf]。

## 动机：循环依赖问题

传统多模态融合方法（如 QMF、PDF、DBF、UAW-EEF）依赖分类器输出的置信度或信念质量来检测错误或分配融合权重。这形成了一个[[circular-dependency-in-multimodal-fusion|循环依赖]]：深度分类器在噪声或 OOD 输入上往往仍给出高置信度，导致可靠性信号与需要检测的错误来自同一来源[^src-gmf]。GMF 通过将可靠性定义为外在几何属性来打破这一循环。

## 方法架构

### 传输代价估计

GMF 使用 Rectified Flow 速度网络估计两类传输代价：

- **模态内传输能量** $E_{\text{intra}}^{(m)}$：从类无关先验分布到模态 $m$ 潜在表示的 rectified flow 的初始速度平方范数。度量单个模态偏离干净流形的程度。
- **模态间传输代价** $E_{\text{inter}}^{(n \to m)}$：跨模态速度场将源模态表示映射到目标模态的端点残差平方范数。度量模态间的语义一致性。

### 融合门控机制

模态 $m$ 的交互门控为：

$$\gamma_{\text{int}}^{(m)} = \lambda \sum_{n \neq m} r^{(n)} \exp\left(-\frac{E_{\text{inter}}^{(n \to m)}}{\kappa}\right)$$

其中 $r^{(n)} = \sigma(\theta_r - E_{\text{intra}}^{(n)})$ 是模态内可靠性评分。稳定化门控 $\tilde{\gamma}_{\text{int}}^{(m)} = \gamma_{\text{int}}^{(m)} + \epsilon_\gamma$ 加上数值地板。

### 最优融合权重

融合权重由熵正则化最小化问题的闭式解给出：

$$w^{*(m)} = \frac{\exp(-C^{(m)}/\tau)}{\sum_j \exp(-C^{(j)}/\tau)}, \quad C^{(m)} = E_{\text{intra}}^{(m)} - \tau \ln \tilde{\gamma}_{\text{int}}^{(m)}$$

### 训练目标

总损失 $L_{\text{total}} = L_{\text{task}} + \lambda_{\text{geo}} L_{\text{geo}} + \lambda_{\text{reg}} L_{\text{reg}}$，其中 $L_{\text{geo}}$ 为 rectified flow 匹配损失，$L_{\text{reg}}$ 为冲突感知正则项（当跨模态偏差大时强制预测分布趋于均匀），且几何分支与分类分支通过分离的梯度路径独立优化[^src-gmf]。

## 理论保证

- **Theorem 4.4（融合权重最优性）**：熵正则化几何代价最小化在概率单纯形上有唯一闭式解。
- **Theorem 4.5（[[geometric-barrier-principle|几何屏障原理]]）**：当模态编码冲突类别时，跨模态传输代价 $\geq (\delta - 2\epsilon)^2 > 0$。
- **Corollary 4.6**：冲突模态的交互门控 $\gamma_{\text{int}}^{(B)} \leq \lambda(M-1)\exp(-(\delta-2\epsilon)^2/\kappa)$，即被指数级抑制。

## 实验表现

在 NYU Depth V2、UPMC Food-101、MVSA-Single、PneumoniaMNIST 四个基准上验证。关键结果：

| 场景 | GMF | 最佳基线 |
|------|-----|---------|
| Food-101 噪声 σ=2.0 准确率 | 58.7% | 53.1% (UAW-EEF) |
| MVSA-Single 安全拒绝率 | 76.8% | 35.2% (DBF) |
| MVSA-Single 冲突检测 AUROC | 89.4 | 71.2 (DBF) |
| PneumoniaMNIST 可靠性 Pearson r | 0.78 | 0.61 (DBF) |

与分类置信度的互信息仅 0.08（vs 统计方法的 0.67），验证了传输能量与分类器输出的独立性[^src-gmf]。

[^src-gmf]: [[source-gmf]]
