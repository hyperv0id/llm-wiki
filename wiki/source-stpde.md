---
title: "STPDE: Decoupling Universal Laws and Environmental Heterogeneity for Robust Spatio-Temporal Forecasting"
type: source-summary
tags:
  - spatio-temporal-forecasting
  - pde
  - ood-generalization
  - continual-learning
  - physics-inspired
created: 2026-07-23
last_updated: 2026-07-23
source_count: 0
confidence: low
status: active
---

# STPDE: Decoupling Universal Laws and Environmental Heterogeneity

**Authors:** Aoyu Liu, Liming Wei, Yaying Zhang (Tongji University)
**Venue:** ICML 2026 (Proceedings of the 43rd International Conference on Machine Learning, Seoul)

## 核心贡献

STPDE 提出将时空预测重新形式化为**非齐次偏微分方程（inhomogeneous PDE）的数值演化**问题。该方法将时空动态显式分解为：

1. **Invariant Diffusion Operator**（不变扩散算子）：以拉普拉斯算子 $\nabla^2$ 为核心，捕获跨环境共享的普适物理传输与守恒机制。利用 Green 函数与线性注意力的结构对应关系，将全局扩散建模实现为 O(N) 复杂度的线性注意力形式（通过可分离核近似 + aggregate–distribute 分解）。

2. **Environment Basis Manifold**（环境基底流形）：用 K 个共享的可学习基底 $\{E_k\}_{k=1}^K$ 参数化空间异质性（扩散系数 $\alpha$ 和源项 $S$），通过稀疏 Top-K 路由机制为每个节点合成本地环境嵌入，经 AdaLN 调制扩散算子输出。

3. **Stochastic Environment Perturbation**（随机环境扰动）：训练时对基底进行扰动（原始/平均/置换三种采样），模拟未见环境变化，正则化不变算子以学习更可迁移的动态。

## 实验覆盖

在 **LargeST**（CA-D3/SD/Orange/LA）和 **PEMS-Stream** 上评估四个场景：

- **ID 预测**：标准同分布设置，12 步历史 → 12 步预测
- **OOD 泛化**：1 月训练 → 8 月测试（季节性分布偏移，基于 Wasserstein 距离选择 8 月为最大漂移月）
- **Few-shot 跨城市迁移**：SD/Orange/LA 预训练 → CA-D3，仅用目标域 10% 数据微调
- **持续学习**：PEMS-Stream 7 个增量周期（655→871 节点），仅扩展 Environment Basis Manifold

## 关键结果

- ID/OOD 上一致优于 GWNet、D2STGNN、STID、STAEformer、HimNet、PatchSTG、BiST 等强基线
- 消融实验：w/o PDE（用 MLP 替换 PDE solver）退化最大，OOD 下尤为显著；w/o D/w/o E/w/o P 均持续退化
- 效率：LA（1728 节点）上训练/推理时间和 GPU 内存均优于 PatchSTG 和 BiST
- 跨城市迁移和持续学习中仅需轻量 Manifold 微调即可快速适应

## 局限性

论文未深入讨论 PDE 假设在严重非物理场景（如极端事件、传感器大面积故障）下的适用边界；基底数 K 需手动选择，对参数的敏感性在大规模部署中可能成为工程负担。
