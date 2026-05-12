---
title: "Denoising Reduction：RL 训练中的去噪步数缩减"
type: technique
tags:
  - flow-matching
  - reinforcement-learning
  - denoising-reduction
  - training-efficiency
created: 2026-05-12
last_updated: 2026-05-12
source_count: 1
confidence: high
status: active
---

# Denoising Reduction：RL 训练中的去噪步数缩减

> 在线 RL 训练时使用极少的去噪步数生成样本（T_train = 10），推理时恢复完整步数（T_infer = 40），大幅加速训练流程而不牺牲性能。[^src-2505-05470]

## 动机

在线 RL 依赖频繁的样本生成来收集训练数据，但 Flow Matching 模型通常需要 40-50 步去噪才能生成高质量图像。对于大规模 T2I 模型，单次推理本身就很昂贵，加上 RL 训练需要生成多组样本（组大小 G=24），直接使用全步训练将使计算成本难以承受。

## 方法

**Denoising Reduction 是一种训练-推理解耦策略**：

1. **训练时**：使用大幅缩减的去噪步数（T_train = 10，原步数 T=40）
2. **推理时**：保持原有默认步数（T_infer = 40）
3. **奖励计算**：始终在完整步数生成的高质量图像上评估奖励

## 工作原理

这一策略有效的核心原因在于 RL 训练的特殊性：

- **目标差异化**：RL 训练的目标是优化策略，而不是生成高质量图像。低质量（少步）样本虽然画质差、有模糊和颜色漂移，但已足够承载布局、物体计数、空间关系等结构信息
- **相对偏好信号**：GRPO 使用组内相对优势估计，即使样本质量不够好，只要奖励函数能正确排序组内样本的优劣（高奖励 vs 低奖励），就能提供有效的学习信号
- **边界分布可迁移**：在较少步数下学到的策略改进，能够完好地迁移到更多步数的推理中

## 实验结果

论文在三项任务上验证了 Denoising Reduction 的效果：

| 步数配置 | GenEval 加速比 | OCR 加速比 | PickScore 加速比 |
|---|---|---|---|
| T=40 (baseline) | 1× | 1× | 1× |
| T=10 (proposed) | **>4×** | **>4×** | **>4×** |
| T=5 | 不稳定性出现 | 不稳定性出现 | 不稳定性出现 |

**T=10 收敛速度比 T=40 快 4 倍以上，最终奖励水平相当**。进一步减到 T=5 时出现不稳定（梯度方差增大），因此 T=10 是平衡速度与稳定性的最优选择。

## 与 ODE-to-SDE 的关系

Denoising Reduction 与 [[ode-to-sde-conversion|ODE-to-SDE 转换]] 互补：
- ODE-to-SDE 解决「确定性 vs 随机性」问题，提供探索机制
- Denoising Reduction 解决「采样效率」问题，降低每轮数据收集的成本

两者结合使在线 RL 在流匹配模型上切实可行。

## 可训练性分析

假设全步训练需要 N 个梯度步才能收敛。在 Denoising Reduction 下：

- **每轮采样耗时**：约减少至原有的 T_train/T_infer = 10/40 = 1/4
- **有效学习信号**：虽然少步样本画质差，但相对排序信号依然有效（实验验证）
- **收敛曲线**：以 wall-clock 时间衡量，T=10 的奖励增长速率远超 T=40

[^src-2505-05470]: [[source-flow-grpo]]
