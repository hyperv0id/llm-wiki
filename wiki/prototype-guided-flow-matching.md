---
title: "Prototype-Guided Flow Matching"
type: technique
tags:
  - flow-matching
  - generative-model
  - time-series-forecasting
  - probabilistic-forecasting
  - arxiv-2026
created: 2026-05-03
last_updated: 2026-08-05
source_count: 2
confidence: high
status: active
---

# Prototype-Guided Flow Matching

**Prototype-Guided Flow Matching** 是 [[aurora|Aurora]] 中提出的生成式概率预测技术，用于在解码阶段生成未来时间序列 token[^src-aurora]。

## 动机

标准 [[flow-matching|Flow Matching]] 通过从噪声到数据的向量场回归实现生成建模，但在时间序列预测中，未来值的生成需要以历史观测和多模态上下文为条件。Aurora 通过引入"原型"（prototypes）来引导流匹配过程，使生成更符合领域特定的未来趋势[^src-aurora]。

## 机制

解码阶段由三个组件协作完成[^src-aurora]：

1. **ConditionDecoder**（DiT 启发，L 层堆叠）：把编码端融合表示 $X^{\text{fuse}}$ 解码为未来 token 的多模态条件 $X^{\text{cond}}$。由 Causal-Transformer（对 $X^{\text{fuse}}$ 末 token 复制 F 份）与集成 RoPE 的 Cross-Transformer（以 $X^{\text{fuse}}$ 为 Key/Value）组成
2. **Prototype Bank**：$M=1000$ 个可学习"未来雏形"向量 $P \in \mathbb{R}^{M \times p^{\text{time}}}$，以三角函数、指数、对数、多项式基初始化——每种雏形代表一种"周期 + 趋势"组合形态
3. **PrototypeRetriever**（Transformer 结构）：接收文本表示 $\tilde{X}^{\text{text}}$ 与图像表示 $\tilde{X}^{\text{image}}$，叠加未来 token 的 Sinusoidal Positional Embedding，经 Softmax 输出 $F$ 个未来 token 对 1000 个原型的分类分布 $D \in \mathbb{R}^{F \times M}$，加权合成面向当前样本的未来原型 $\tilde{P} = D \cdot P$

**Flow-Matching Network**（MLP 结构 + AdaLN 条件注入）：起点设为 $y^{(0)} = \tilde{P}_i + \epsilon_i$（原型 + 高斯噪声，而非纯高斯噪声），沿条件**最优传输（OT）路径**（能量最优、速度场均匀）拟合速度场。token-wise 优化目标：

$$\mathcal{L}(\theta, h_i) = \mathbb{E}_{t, y_i^{(0)}, y_i^{(1)}} \left[\left\| v_t^\theta(y_i^{(t)} | h_i) - (y_i^{(1)} - y_i^{(0)}) \right\|^2\right]$$

其中 $y_i^{(t)} = t y_i^{(1)} + (1-t) y_i^{(0)}$，$h_i = X^{\text{cond}}_i$ 为条件。推理时按 Algorithm 1 进行 $J$ 步离散积分：

```text
Algorithm 1: Prototype-Guided Flow Matching (sampling)
给定条件 X_cond^i、步数 J、原型 P̃_i
采样噪声 ε_i ~ N(0, I);  Δt = 1/J
h_i = X_cond^i;  ŷ_i = P̃_i + ε_i
for j in {0, ..., J-1}:
    ŷ_i ← ŷ_i + v_{jΔt}^θ(ŷ_i | h_i) · Δt
return ŷ_i
```

高斯噪声 $\epsilon_i \sim \mathcal{N}(0, I)$ 提供了概率预测所需的采样多样性；对 $F$ 个未来 token 逐 token 生成后拼接为 $F \times p^{\text{time}}$ 的预测窗口。多次采样即得预测分布。

## 与标准 Flow Matching 的区别

| 维度 | 标准 Flow Matching | Prototype-Guided Flow Matching |
|------|-------------------|-------------------------------|
| 起点 | 标准高斯噪声 | 未来原型 + 高斯噪声（$\tilde{P}_i + \epsilon_i$） |
| 条件 | 无条件或简单条件 | ConditionDecoder 解码的多模态条件（AdaLN 注入） |
| 路径 | 任意插值 | 条件最优传输（OT）路径 |
| 应用 | 图像/音频生成 | 时间序列概率预测 |
| 引导 | 无 | 原型检索 + 条件双重引导 |

## 实验证据

- **消融**：论文报告去掉原型引导（Variant 2，退回标准高斯噪声起点）后，TimeMMD 上 Social Good MSE 0.838→1.425、Traffic 0.161→0.273；与去掉模态引导注意力叠加时性能大幅下降[^src-aurora]
- **采样可扩展性**：论文报告在 ProbTS 上采样数 20→100 时 CRPS/NMAE 持续改善，100 次达良好性能；模型以此支持概率预测（采样 100 次、推理 83.5ms/样本）
- **可视化**：附录 C.2 展示了 1000 个原型的形态（部分形似但幅度/相位不同）；C.3 展示了生成的未来原型贴合 Groundtruth 的周期/趋势骨架

## 与 SimDiff 扩散方法的对比

[[simdiff|SimDiff]] 使用 DDPM 扩散模型进行点预测，通过 Median-of-Means 将概率样本聚合为点估计[^src-simdiff]。论文报告 Aurora 的 Prototype-Guided Flow Matching 直接进行概率预测、保留完整的预测分布信息，并使用更高效的 Flow Matching（OT 路径直线轨迹）替代扩散路径[^src-aurora]。

## 相关页面

- [[aurora]] — Aurora 模型
- [[modality-guided-self-attention]] — 编码阶段的模态引导注意力
- [[flow-matching]] — Flow Matching 理论基础
- [[generative-time-series-forecasting]] — 生成式时间序列预测概念
- [[simdiff]] — 扩散式生成预测对比

[^src-aurora]: [[source-aurora]]
[^src-simdiff]: [[source-simdiff]]
