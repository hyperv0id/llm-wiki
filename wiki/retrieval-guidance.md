---
title: "Retrieval Guidance"
type: technique
tags:
  - diffusion-models
  - retrieval-augmented
  - score-estimation
  - inference-time
  - guidance
created: 2026-06-08
last_updated: 2026-06-09
source_count: 2
confidence: medium
status: active
---

# Retrieval Guidance（检索引导）

**Retrieval Guidance** 是 [[middir|MiDDiR]] 提出的推理时扩散采样增强技术——通过从训练集检索相似历史模式，分析性偏置得分估计以提升条件生成质量，尤其改善低密度区域的采样效果[^src-middir]。这是**首个将检索与分析性引导结合到扩散生成的工作**[^src-middir]。

## 动机

扩散模型的最大似然训练本质上对数据流形的**低密度区域估计不足**——罕见但可重复的模式因样本量少而欠拟合，导致在这些区域的预测次优[^src-middir]。传统方案（如 classifier guidance、classifier-free guidance）需要额外的标签或条件训练，而检索引导直接从训练集中提取参照信息，无需额外训练。

## 机制

### 第一步：混合依赖相似度检索

1. **构建检索数据库**：对训练集每个样本，用 CD 编码器 ϕ 编码得到通道级隐向量 e，与目标序列 x^p 配对存储[^src-middir]：
   $$D_\text{retrieval} = \{(e_{1}, x^p_{1}), ..., (e_{M \times C}, x^p_{M \times C})\}$$

2. **检索**：对测试样本，用编码器 ϕ 生成查询向量 e_c，按余弦相似度检索 Top-K 最近邻[^src-middir]：
   $$i_1, ..., i_K = \arg \text{Top-}K \frac{e_i^\top e_c}{\|e_i\|\|e_c\|}$$

3. **加权聚合**：以相似度为权重，平均 K 个检索到的目标序列[^src-middir]：
   $$x^r_c = \frac{\sum_{k=1}^K s_{i_k} \cdot x^p_{i_k}}{\sum_{k=1}^K s_{i_k}}$$

### 第二步：检索引导得分估计

将检索引导视为从指数倾斜分布中采样[^src-middir]：

$$p_\theta(\hat{x}^p|e) = p_\theta(x^p|e) e^{-\lambda E(x^r, x^p)}$$

得分函数被偏置为[^src-middir]：

$$\nabla_{\hat{x}^p_n} \log p_\theta(\hat{x}^p_n|e) = \nabla_{x^p_n} \log p_\theta(x^p_n|e) - \lambda \nabla_{x^p_n} E(x^r, x^p_n)$$

其中：
- λ 为引导强度超参数
- E 为 L2 距离能量函数
- 梯度归一化：每步估计的能量得分标准差缩放至 1

### 关键设计决策

- **通道级检索**：CI 去噪意味着每通道独立生成 → 检索也按通道独立执行，与混合依赖策略一致[^src-middir]
- **一次检索**：检索仅在推理开始时执行（非每个扩散步），开销极小——单变量检索 0.054–0.176 ms，引导仅增加采样步时间 0.51%–0.86%[^src-middir]
- **编码器即检索器**：CD 编码器同时作为检索键提取器，无需额外模型

## 消融分析

| λ | ETTm1 MAE | Traffic MAE | 现象 |
|---|-----------|-------------|------|
| 0 | - | - | 无引导基线 |
| 0.005–0.01 | ↓ 改善 | ↓ 改善 | 适度引导有益 |
| 0.02–0.05 | ↑ 恶化 | ↓ 继续改善 | ETTm1 过拟合，Traffic 持续获益 |

核心发现[^src-middir]：

1. **最优 λ 数据集依赖**：小数据集（ETTm1）过度引导导致过拟合训练集；大数据集/高维（Traffic 862 通道）引导获益更大
2. **引导样本可视化**：λ 增大 → 生成的预测区间逐渐收紧并向检索序列靠拢，验证了引导偏置采样分布的预期效果
3. **与 CD 编码的协同**：无 CD 编码时检索引导效果下降超 50%，说明通道依赖编码提供的序列级信息对检索引导至关重要

## 与其他检索方法对比

| 方法 | 检索对象 | 检索时机 | 集成方式 |
|------|---------|---------|---------|
| [[gtr|GTR]] | 全局时间嵌入 | 训练+推理 | 2D 卷积残差融合 |
| RAST | 时空双维度向量 | 推理 | Cross-attention 融合 |
| [[ratd\|RATD]] | 数据库 k-NN 参照（未来段） | 推理（仅一次检索） | RMA 注意力（条件特征输入） |
| **Retrieval Guidance** | 通道级训练样本 | 推理（仅一次） | **分析性偏置得分估计** |

关键区别：Retrieval Guidance 直接修改采样过程的得分函数（数学分析性），而非将检索结果作为模型输入特征[^src-middir]。

## 相关技术

- [[middir|MiDDiR]] — 检索引导的提出模型
- [[mixed-channel-dependency]] — 混合通道依赖策略（CD 编码器即检索键提取器）
- [[classifier-free-guidance]] — 扩散模型的条件引导方法
- [[gtr|GTR]] — 全局时序检索（训练时学习检索模块）
- [[retrieval-augmented-spatio-temporal-forecasting]] — RAG-for-STF 范式

[^src-middir]: [[source-middir]]
[^src-ratd]: [[source-ratd]]
