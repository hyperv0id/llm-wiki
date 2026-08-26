---
title: "FENCE"
type: technique
tags:
  - diffusion-models
  - spatiotemporal-imputation
  - dynamic-guidance
  - traffic
  - classifier-free-guidance
  - aaai-2026
created: 2026-06-08
last_updated: 2026-08-26
source_count: 2
confidence: medium
status: active
---

# FENCE (Spatial-Temporal Feedback Diffusion GuidaNCE)

**FENCE** 是 Mao, Ding 等人发表于 AAAI 2026 的动态反馈扩散引导方法，用于受控交通数据时空插补[^src-fence]。针对 [[csdi|CSDI]]、[[pristi|PriSTI]] 等扩散插补方法使用统一引导尺度的问题，FENCE 将固定的 [[classifier-free-guidance|无分类器引导]] 尺度 $\lambda$ 替换为基于后验似然动态调整的反馈引导（该机制的理论公式采用自 Koulischer et al. 2025 的 Feedback Guidance 工作[^src-fence]），以应对高缺失率节点因条件信息不足而漂移到先验分布的现象[^src-fence]。

## 问题与动机

现有扩散插补方法（CSDI、PriSTI）使用固定引导尺度 $\lambda$ 进行 [[classifier-free-guidance|CFG]] 引导，但这一策略在高缺失率节点上失效[^src-fence]。当某节点在一个时间段内完全没有观测值时，CFG 的条件引导梯度范数始终很低——学到的条件分布已坍缩到无条件先验，生成过程偏向从 $p(x_k)$ 而非 $p(x_k|c)$ 采样，导致插补值偏离真实值[^src-fence]。

FENCE 的核心洞察：引导尺度不应是超参数，而应是去噪状态的函数——当后验似然 $p(c|x_k)$ 降低时，应增大引导以拉回条件分布；当后验升高时，应减小引导以避免过校正[^src-fence]。

## 核心机制

### 1. 反馈引导循环

在每一步去噪 $k$ 中，FENCE 构建引导评分如下[^src-fence]：

$$\nabla_{x_k} \log \tilde{p}_{\theta,k}(x_k|c) = s_\theta(x_k) + \lambda(x_k, k) \cdot \big(s_\theta(x_k, c) - s_\theta(x_k)\big)$$

其中引导尺度 $\lambda(x_k, k)$ 根据后验似然 $p_{\theta,k}(c|x_k)$ 动态计算[^src-fence]：

$$\lambda(x_k, k) \approx \frac{p_{\theta,k}(c|x_k)}{p_{\theta,k}(c|x_k) - (1-\pi)}$$

- $\pi \in [0,1]$ 是事先后验置信度超参数，$\pi=0.5$ 时性能最佳[^src-fence]
- 当 $p(c|x_k)$ 高 → $\lambda \to 1$（轻度引导）
- 当 $p(c|x_k) \to (1-\pi)$ → $\lambda$ 急剧增大（强引导）[^src-fence]

### 2. 后验似然追踪

后验无法直接获取，FENCE 通过扩散反向过程的马尔可夫性质迭代更新[^src-fence]：

$$\log p_{\theta,k-1}(c|x_{k-1}) = \log p_{\theta,k}(c|x_k) + \log p_\theta(x_{k-1}|x_k, c) - \log p_\theta(x_{k-1}|x_k)$$

引入温度 $\tau$ 和偏移 $\delta$（分别由 $t_0, t_1$ 控制）来调节更新强度和引导激活时机[^src-fence]。

### 3. 聚类感知引导

不同节点对观测的符合程度不同，FENCE 在每步去噪时利用空间注意力分数 $A_{\text{attn}}$ 做 k-means 聚类[^src-fence]。对每个聚类 $C_j$，计算聚类级对数后验均值：

$$\log p_{\theta,k-1,C_j}(c|x_{k-1}) = \frac{1}{|C_j|} \sum_{l \in C_j} \log p_{\theta,k-1,l}(c|x_{k-1})$$

聚类级后验用于计算该聚类内所有节点的共享引导尺度，比全局统一尺度或逐节点尺度都更稳定[^src-fence]。最优聚类数约为 $N/20$[^src-fence]。

### 4. 两阶段训练

- **第一阶段**：训练无条件模型，仅用结构先验（节点嵌入 + 时间嵌入），无观测值输入，学习先验分布 $p_\theta(x)$[^src-fence]
- **第二阶段**：从无条件模型权重初始化，加入条件观测进行微调，学习条件分布 $p_\theta(x|c)$[^src-fence]

## 实验结果

在 PEMS04、PEMS07、PEMS08 三个数据集上，两种缺失模式（SR-TC 和 SC-TC，80% 缺失率）下全面超越 [[csdi|CSDI]]、[[pristi|PriSTI]]、[[imputeformer|ImputeFormer]]、ASTGNN、IGNNK、GCASTN、LCR、mTAN 等 8 个基线[^src-fence]。MAPE 平均提升 6.26%[^src-fence]。

## 与相关方法的关系

| 方法 | 引导机制 | 引导尺度 | 空间信息 |
|------|---------|---------|---------|
| [[csdi|CSDI]] | 固定 CFG | 统一 $\lambda$ | 无（论文未提及其空间处理） |
| [[pristi|PriSTI]] | 固定 CFG | 统一 $\lambda$ | 论文仅述"integrates geographic context"，未详述机制 |
| **FENCE** | 动态反馈引导 | $\lambda(x_k, k)$，聚类级 | 动态注意力分数 k-means 聚类 |

## 局限性

- 论文实验仅在 PEMS04、PEMS07、PEMS08 交通数据集上验证，其他时空数据类型的泛化性未在论文中报告[^src-fence]

## 另见

- [[loft]] — 同组（Mao 等）后续工作，KDD 2026：以流匹配 + 低秩先验 + 轨迹一致性替代扩散路线做交通插补，将 FENCE 列为生成式基线对比（作者报告 PEMS04 SC-TC RMSE 41.67 vs FENCE 44.28）[^src-loft]

[^src-fence]: [[source-fence]]
[^src-loft]: [[source-loft]]