---
title: "Deep Model Fusion"
type: concept
tags:
  - deep-model-fusion
  - model-merging
  - model-ensemble
  - model-mixing
  - multi-task-learning
  - learn-from-model
created: 2026-07-31
last_updated: 2026-07-31
source_count: 1
confidence: medium
status: active
---

# Deep Model Fusion（深度模型融合）

**Deep model fusion** 指把多个深度网络的**预测或参数**（有时是结构组件）统一成一个模型，以提升性能、效率或鲁棒性，而不从零训练。FusionBench 将其放在更广的 “learn from model” 范式下，与微调、蒸馏、剪枝、编辑并列[^src-jmlr-25-1243]。

## 三分法（FusionBench taxonomy）

| 类别 | 操作对象 | 输出形态 | 典型代价 |
|------|----------|----------|----------|
| **Ensemble** | 预测 | 多模型并存 | 推理与存储 ×N |
| **Merging** | 同构参数 | 单组参数、同架构 | 融合时可要数据/TTA；推理≈单模 |
| **Mixing** | 层/专家等组件 | 常异构或扩参（如 MoE） | 常需再训或测试时适配 |

形式化直觉（FusionBench Appendix A）[^src-jmlr-25-1243]：

- Ensemble：\(y = A_{\mathrm{ens}}(x; f_1,\ldots,f_N; w)\)
- Merging：\(\theta = A_{\mathrm{merge}}(\theta_1,\ldots,\theta_N; w)\)，\(f(\cdot;\theta)\)
- Mixing：\(\Theta = A_{\mathrm{mix}}(\ldots)\)，维数可变为 \(d'\)，\(F(\cdot;\Theta)\) 更表达

压缩（剪枝、BitDelta 等）被 FusionBench 挂为**配套**而非第四类融合：融前/融后降本[^src-jmlr-25-1243]。

## 为何重要

1. **数据与算力效率**：复用已有专家检查点，避免多任务联合重训。
2. **基础模型时代**：全量重训越来越贵；参数空间插值、任务向量编辑成为实用工具。
3. **评测曾碎片化**：无统一任务/模型/协议时，无法比较 AdaMerging vs Ties 等；[[fusionbench|FusionBench]] 针对此缺口[^src-jmlr-25-1243]。

## 与相邻概念

| 概念 | 区别 |
|------|------|
| 传统 multi-task learning | 联合数据重训；融合是*后置*装配 |
| 知识蒸馏 | 师生传递；融合多是对称组合已训模型 |
| [[mixture-of-experts\|Mixture of Experts]] | 训时稀疏/稠密路由 vs 事后 multi-checkpoint mixing（WE-MoE/SMILE） |
| 多模态特征融合 | 融的是模态表示，不是多个任务专家的参数 |

子领域 **[[model-merging]]** 专指参数合并；**[[task-arithmetic]]** 是 merging 中以任务向量做加减的代表技术。

## 实证轮廓（FusionBench）

- CLIP 八任务：简单平均远弱于 RegMean / 层间 AdaMerging；WEMoE/SMILE 逼近传统 MTL[^src-jmlr-25-1243]。
- 层间自适应 ≫ 任务级自适应。
- 更大骨干上，融合与 STL 上界的缝更窄。
- **负迁移**与**腐蚀分布**下性能塌陷是一阶风险，不能只看 seen 平均准确率[^src-jmlr-25-1243]。

## 相关页面

- [[fusionbench]] · [[source-jmlr-25-1243]]
- [[model-merging]] · [[task-arithmetic]] · [[mixture-of-experts]]

[^src-jmlr-25-1243]: [[source-jmlr-25-1243]]
