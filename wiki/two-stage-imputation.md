---
title: "Two-Stage Imputation"
type: technique
tags:
  - time-series
  - data-imputation
  - diffusion-models
  - refinement
created: 2026-06-08
last_updated: 2026-08-29
source_count: 3
confidence: medium
status: active
---

# Two-Stage Imputation (双阶段插补)

**双阶段插补**是一种插补精炼范式，最早由 [[saits|SAITS]] 的 DMSA (Diagonal Masked Self-Attention) 块引入，被 [[sadi|SADI]] 首次应用于扩散模型插补框架[^src-sadi]。核心思想：第一阶段产生初始插补，第二阶段基于初始插补结合原始信息进行精炼，最终通过可学习加权组合融合两阶段输出[^src-sadi]。

## SADI 的双阶段实现

SADI 在去噪函数中部署两个 [[gated-temporal-attention|GTA]] 块[^src-sadi]：

1. **第一阶段 (GTA₁)**：以 FDE 输出和观测值为输入，输出 $\epsilon_1$（初始噪声预测）和注意力权重 $W_L$
2. **第二阶段 (GTA₂)**：以 GTA₁ 的隐状态 + **重新引入的原始噪声数据**为输入，输出 $\epsilon_2$（精炼的噪声预测）

重新引入原始噪声数据的设计是为了**接地**——防止第二阶段完全依赖第一阶段可能错误的插补而偏离真实分布[^src-sadi]。

## 加权组合

SADI 不直接使用第二阶段输出，而是学习动态加权[^src-sadi]：

$$\tilde{W}_L = \text{sigmoid}(\text{linear}(\text{concat}(W_L, M_0^{co})))$$

$$\epsilon_\theta = (1 - \tilde{W}_L) \odot \epsilon_1 + \tilde{W}_L \odot \epsilon_2$$

关键设计[^src-sadi]：
- $W_L$ 来自 GTA₁ 的注意力权重矩阵（$L\times L$）
- $M_0^{co}$ 缺失掩码告知模型哪些位置是观测值
- 拼接后经 FFN 投影到 $(L, K)$ 维度
- 逐元素加权，允许**每个位置独立决定**两阶段的贡献比例

## 为什么需要加权组合

SADI 作者在实践中观察到第二阶段有时会降低精度[^src-sadi]。加权组合机制允许模型在第二阶段不利于插补时自动降低其权重，避免退化。

消融实验证明[^src-sadi]：移除加权组合（直接用 $\epsilon_2$）导致 AgAID MSE 从 $2.93\times10^{-4}$ 升至 $7.81\times10^{-4}$（~2.7×），Electricity（10 缺失特征）MSE 从 0.107 飙升至 2.10（~19.6×）。

## 与其他方法的双阶段设计

| 方法 | 阶段 1 | 阶段 2 | 融合方式 |
|------|--------|--------|---------|
| [[saits\|SAITS]] | First DMSA | Second DMSA | 加权组合（来自 DMSA 注意力权重） |
| **SADI** | GTA₁ (FDE 后) | GTA₂ (噪声接地) | 动态逐元素加权 |
| [[cofill\|CoFILL]] | 时域 TCN | 频域 DCT | Cross-Attention 融合（非序列化） |
| ImputeFormer | 投影注意力 | 嵌入注意力 | 堆叠（非加权） |
| [[rdpi\|RDPI]] (arXiv 2024) | 确定性插补模型（实验用 [[grin\|GRIN]]） | 残差条件扩散（观测值写入前向过程） | 相减：初值减去预测残差（Algorithm 2） |

SADI 的独特之处在于：(1) 第二阶段重新引入原始噪声数据作为接地信号；(2) 逐元素的独立动态权重允许细粒度控制[^src-sadi]。

## 网络级与框架级两种用法

上表前四行的"两阶段"都发生在**单一去噪网络（或注意力块）内部**：两个阶段是同一模型的两个组件，输出经加权或注意力融合——SADI 即两个 GTA 块部署于同一去噪函数内[^src-sadi]。[[rdpi|RDPI]] 的两阶段则是**框架级**的——初始模型 $f_\theta$ 与扩散模型 $g_\theta$ 是两个独立模型，扩散目标不是缺失值本身，而是初始估计与真值之间的残差 $z_0^m = f_\theta(x_0^c) - x_0^m$，采样时以初值减去预测残差得到最终插补，两阶段以 $L_{joint} = L_{simple} + \lambda L_{init}$ 联合训练[^src-rdpi]。两种用法不同义，跨方法比较时应注意层级。

## 命名辨析：与综述 "impute-then-predict" 范式的区别

Wang & Du 等人的 MTSI 综述在讨论下游任务集成时使用了另一种"两阶段"表述：主流的 "impute-then-predict"（综述原文为 "impute and predict"）范式把插补当作数据预处理，先填补缺失、再将完整数据交给下游任务模型；替代方案是 "encode-and-predict" 端到端范式，把不完整数据编码为表示后做多任务学习（插补 + 分类/预测等），综述认为当缺失模式本身携带下游有用信息时端到端方式更有前景[^src-mts-imputation-survey]。注意这与本页的"双阶段插补"不同义：本页指单一模型内部的两段式精炼结构，综述的 "impute and predict" 指"插补模块 + 下游模型"的流水线级组装——跨文献检索 "two-stage imputation" 时应先区分层级。

## 关联页面

- [[sadi]] — SADI，将双阶段插补引入扩散模型的首次实践
- [[gated-temporal-attention]] — GTA，双阶段的核心计算单元
- [[feature-dependency-encoder]] — FDE，为第一阶段提供特征依赖信息
- [[mixed-partial-blackout-training]] — MPB，增强双阶段在 partial blackout 下的鲁棒性
- [[saits]] — SAITS，双阶段插补的注意力设计灵感来源
- [[rdpi]] — RDPI，框架级两阶段（确定性初值 + 残差扩散精炼）
- [[mts-imputation-taxonomy]] — MTSI 综述的分类框架（其 "impute and predict" 是与本页不同层级的"两阶段"概念）

[^src-sadi]: [[source-sadi]]
[^src-rdpi]: [[source-rdpi]]
[^src-mts-imputation-survey]: [[source-mts-imputation-survey]]
