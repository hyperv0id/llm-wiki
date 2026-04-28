---
title: "Shortcut Models"
type: technique
tags:
  - shortcut-models
  - one-step-generation
  - flow-matching
  - generative-model
  - uc-berkeley
  - arxiv-2025
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Shortcut Models

**Shortcut Models**（快捷模型）是由 UC Berkeley 的 Kevin Frans 等人于 2025 年提出的生成模型，旨在解决扩散模型推理速度慢的问题[^src-shortcut-models]。它通过让模型学习"捷径"，在单次前向传播中完成从噪声到数据的生成。

## 背景问题

### 扩散模型的迭代去噪

扩散模型和 Flow Matching 模型通过学习一个常微分方程（ODE），将噪声逐步映射到数据：

$$
\frac{d}{dt} x(t) = v_\theta(x(t), t)
$$

采样时需要从 $t=0$ 到 $t=1$ 迭代求解，通常需要数十到数百步网络前向传播。

### 少步生成的困难

1. **歧义问题**：在 $t=0$ 时，模型只能看到纯噪声，而训练时噪声 $x_0$ 和数据 $x_1$ 是随机配对的，因此模型预测的速度方向指向**数据集均值**
2. **曲线轨迹**：Flow Matching 的轨迹是弯曲的，大步长会导致"脱轨"
3. **一步失败**：对于多模态数据分布，单步生成会灾难性失败（只能生成均值附近的图像）

---

## 核心思想

### 步长调节

Shortcut Models 不仅根据当前时间步 $t$ 调节网络，还根据**期望的步长 $d$** 调节网络：

$$
x'_{t+d} = x_t + s_\theta(x_t, t, d) \cdot d
$$

其中 $s_\theta(x_t, t, d)$ 是模型预测的"快捷方向"（shortcut）。

### 关键洞察

- 当 $d \to 0$ 时，$s_\theta(x_t, t, d)$ 等价于 Flow Matching 的瞬时速度 $v_t$
- 当 $d > 0$ 时，模型学习"跳过"多个步骤，直接到达正确位置
- 通过学习不同步长，模型能够适应任意推理预算

---

## 自一致性约束

### 核心性质

Shortcut 模型满足**自一致性**（self-consistency）约束：

$$
\boxed{s(x_t, t, 2d) = \frac{1}{2}s(x_t, t, d) + \frac{1}{2}s(x'_{t+d}, t+d, d)}
$$

其中 $x'_{t+d} = x_t + s(x_t, t, d) \cdot d$ 是沿预测方向的一步。

### 直观理解

- 走两步大小为 $d$ 的捷径，等价于先走一步 $d$，再走一步 $d$
- 这个约束允许我们用小步长的预测来构建大步长的目标

---

## 训练目标

### 组合损失函数

$$
\mathcal{L}_S = \mathcal{L}_{\text{FM}} + \mathcal{L}_{\text{SC}}
$$

其中：

1. **Flow Matching 目标**（$d=0$）：

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0 \sim \mathcal{N}, x_1 \sim D, t} \| s_\theta(x_t, t, 0) - (x_1 - x_0) \|^2
$$

这确保模型在小步长时具有基础的去噪能力。

2. **自一致性目标**（$d>0$）：

$$
\mathcal{L}_{\text{SC}} = \mathbb{E}_{x_0, x_1, t, d} \| s_\theta(x_t, t, 2d) - s_{\text{target}} \|^2
$$

其中：

$$
s_{\text{target}} = \text{stopgrad}\left( \frac{s_\theta(x_t, t, d) + s_\theta(x'_{t+d}, t+d, d)}{2} \right)
$$

### 训练算法

```
while not converged do:
    x0 ~ N(0,I), x1 ~ D, (d,t) ~ p(d,t)
    xt = (1-t)x0 + t*x1
    
    # 小批量: d = 0 (Flow Matching)
    for batch[:k]:
        target = x1 - x0
        loss = ||s_theta(xt, t, 0) - target||^2
    
    # 大批量: d > 0 (自一致性)
    for batch[k:]:
        st = s_theta(xt, t, d)
        xt+d = xt + st * d
        st+d = s_theta(xt+d, t+d, d)
        target = (st + st+d) / 2  # stopgradient
        loss = ||s_theta(xt, t, 2d) - target||^2
    
    theta = theta - lr * grad(loss)
```

---

## 为什么有效？

### 1. 避免歧义

- Flow Matching 在 $t=0$ 时只能预测平均方向（数据集均值）
- Shortcut 模型通过自一致性学习，可以从 $t>0$ 的位置"回溯"到正确的数据点

### 2. 直线轨迹

- OT Flow Matching 的轨迹是直线，Shortcut 模型学习直接跳到目标点
- 不需要跟随弯曲的 ODE 轨迹

### 3. 自举训练

- 自一致性目标使用模型自身的预测作为目标
- 这实际上是一种**自蒸馏**（self-distillation），无需外部教师模型

---

## 实验结果

### CelebA-HQ-256（无条件生成）

| 方法 | 128-Step FID | 4-Step FID | 1-Step FID |
|------|--------------|------------|------------|
| Diffusion | 23.0 | 123.4 | 39.7 |
| Flow Matching | 7.3 | 63.3 | 280.5 |
| Consistency Training | 53.7 | 19.0 | 33.2 |
| Progressive Distillation | 14.8 | 201.9 | 142.5 |
| **Shortcut Models** | **6.9** | **13.8** | **20.5** |

### ImageNet-256（类别条件生成）

| 方法 | 128-Step FID | 4-Step FID | 1-Step FID |
|------|--------------|------------|------------|
| Diffusion | 39.7 | 464.5 | 467.2 |
| Flow Matching | 17.3 | 108.2 | 324.8 |
| Consistency Training | 42.8 | 43.0 | 69.7 |
| **Shortcut Models** | **15.5** | **28.3** | **40.3** |

### 关键发现

1. **Shortcut Models 全面优于端到端方法**：在 128、4、1 步设置下均优于 Consistency Training
2. **与两阶段蒸馏方法可比**：在 4 步和 1 步设置下与 Progressive Distillation 相当或更好
3. **保持多步生成能力**：128 步性能与 Flow Matching 相当（6.9 vs 7.3）
4. **可扩展性**：随着模型规模增大，单步生成质量持续提升

---

## 与其他方法对比

### vs 蒸馏方法（Reflow, Progressive Distillation）

| 方面 | 蒸馏方法 | Shortcut Models |
|------|----------|-----------------|
| 训练阶段 | 两阶段（预训练+蒸馏） | 单阶段 |
| 网络数量 | 教师+学生 | 单一网络 |
| 推理预算 | 固定 | 可变 |

### vs Consistency Models

| 方面 | Consistency Models | Shortcut Models |
|------|-------------------|-----------------|
| 训练调度 | 需要逐渐增加步长的调度 | 无需调度 |
| 损失函数 | 需要 L2 + Perceptual Loss | 仅 L2 |
| 引导 | 需要 bootstrap 路径 | 使用自一致性 |

### vs Flow Matching

| 方面 | Flow Matching | Shortcut Models |
|------|---------------|-----------------|
| 128 步 | 7.3 | 6.9 |
| 1 步 | 280.5 | 20.5 |

---

## 训练技巧

### 1. 经验目标 vs 自一致性目标

- 经验目标（$d=0$）方差大，因为噪声-数据配对有内在不确定性
- 自一致性目标使用确定性自举目标
- 实践中使用 3:1 的经验目标:自一致性目标比例

### 2. 权重衰减

- 权重衰减对训练稳定性至关重要
- 早期不使用权重衰减会导致模型学习噪声目标，产生伪影

### 3. EMA（指数移动平均）

- 使用 EMA 参数生成自一致性目标
- 减少 $d=0$ 层的方差导致的输出振荡

### 4. 离散时间采样

- 训练时只采样与步长��应的离散时间点
- 减少需要学习的表示数量

---

## 机器人控制应用

Shortcut Models 还被应用于机器人控制任务：

| 任务 | Diffusion Policy (100步) | Shortcut Policy (1步) |
|------|-------------------------|----------------------|
| Push-T | 0.95 | 0.87 |
| Transport | 1.00 | 0.80 |

单步 Shortcut Policy 显著优于单步 Diffusion Policy（0.12 和 0.00）。

---

## 相关页面

- [[flow-matching]] — Flow Matching 基础
- [[consistency-models]] — Consistency Models
- [[diffusion-model]] — 扩散模型
- [[progressive-distillation]] — 渐进蒸馏

## 引用

[^src-shortcut-models]: [[source-shortcut-models]]