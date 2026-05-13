---
title: "Flow-OPD: On-Policy Distillation for Flow Matching Models"
type: source-summary
tags:
  - flow-matching
  - on-policy-distillation
  - text-to-image
  - grpo
  - arxiv-2026
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: high
status: active
---

# Flow-OPD: On-Policy Distillation for Flow Matching Models

> Fang, Z., Huang, W., Zeng, Y., Zhao, Y., Chen, S., Feng, K., Lin, Y., Chen, L., Chen, Z., Cao, S., & Zhao, F. (2026). Flow-OPD: On-Policy Distillation for Flow Matching Models. arXiv:2605.08063.

## 核心贡献

Flow-OPD 是首个将 On-Policy Distillation (OPD) 集成到 Flow Matching 模型后训练（post-training）框架中的方法，旨在解决多任务对齐中的两大瓶颈：标量奖励稀疏性（reward sparsity）和梯度干扰（gradient interference）。[^src-2605-08063]

### 三个关键发现

1. **GRPO 在单任务上有效** — 通过在线探索突破离线 SFT 的性能上限
2. **GRPO 在多任务上失败** — 标量奖励混合导致 seesaw effect：优化一项指标必然损害其他指标（如 OCR 提升导致画质下降），根源是梯度干扰（⟨∇θJi, ∇θJj⟩ < 0）
3. **简单混合训练不可行** — 逐步叠加 GenEval、OCR、PickScore、DeQA 四个奖励函数，每加一个奖励就导致之前已优化指标下降 3-9%

## 方法架构

Flow-OPD 采用两阶段对齐策略：[^src-2605-08063]

### 阶段一：领域专家教师训练

使用单奖励 GRPO 微调，为每个任务独立培养专用教师模型（GenEval、OCR、PickScore、DeQA 各一个），让每个专家在自己的指标上达到性能上限。

### 阶段二：多教师在线蒸馏

通过三步协调机制将异构专家知识整合到单一学生模型中：

1. **On-Policy Sampling** — 通过 [[ode-to-sde-conversion|ODE-to-SDE 转换]] 将确定性 ODE 采样转为随机 SDE 采样，使学生模型暴露自己的分布偏移，实现充分的状态空间探索
2. **Task-Routing Labeling** — 硬路由机制，每个文本条件 c 映射到对应的领域专家 k，只激活一个教师提供参考速度场 v_ϕk(xt, t, c)
3. **Dense Trajectory-Level Supervision** — 推导出连续时间 Reverse KL 散度的闭式解作为稠密奖励信号（等价于速度场 L2 距离），结合 PPO clip 稳定训练

### Cold-Start 策略

两种初始化方式：[^src-2605-08063]
- **SFT-based** — 用专家教师的轨迹对进行监督微调
- **Model Merging** — 合并多个教师的参数，将学生置于损失景观中已有协同的高能力区域

### Manifold Anchor Regularization (MAR)

引入一个与任务无关的审美教师（DeQA 优化）提供全数据监督，将生成过程锚定在高品质流形上，有效缓解纯 RL 对齐中常见的审美退化（background mode collapse、semantic redundancy）。[^src-2605-08063]

## 实验结果

### 主要指标（SD3.5-Medium 基础）

| 模型 | GenEval | OCR Acc. | Avg |
|------|---------|----------|-----|
| SD3.5-M (base) | 0.63 | 0.59 | 0.7166 |
| GRPO-Mix | 0.73 | 0.83 | 0.8165 |
| **Flow-OPD (Merge)** | **0.92** | **0.94** | **0.9044** |

### 关键现象

- **Teacher-Surpassing 效应** — 学生在某些边缘案例上超越所有单个教师，推测源于知识交叉授粉（knowledge cross-pollination）在潜流形中形成更平滑的表示
- **OOD 泛化** — 在 T2I-CompBench++ 上超越 GRPO-mix 和 Cold-Start+GRPO，在 Color、Shape、3D-Spatial、Numeracy 等维度全面领先
- **Qwen-VL Score** — 4.05 vs DiffusionNFT 的 3.74，展示更强的复杂视觉推理和布局连贯性

## 局限与展望

- **教师性能天花板** — 当专业教师生成语义错误的图像时，错误会通过稠密监督传播
- **架构同质需求** — 教师和学生模型需要相同架构以支持逐步骤的细粒度监督
- 未来方向：共进化蒸馏（co-evolutionary distillation）、自蒸馏、跨词汇蒸馏（cross-vocabulary distillation）[^src-2605-08063]

## 引用

[^src-2605-08063]: [[source-flow-opd]]