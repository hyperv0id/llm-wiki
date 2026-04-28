---
title: "Adjoint Sensitivity Method"
type: technique
tags:
  - neural-ode
  - backpropagation
  - optimization
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# 伴随灵敏度方法

**伴随灵敏度方法（Adjoint Sensitivity Method）** 是 Neural ODE 论文中提出的反向传播技术[^src-neural-ode]，通过求解增广 ODE 来计算 ODE 求解器的梯度。

## 动机

传统方法需要对 ODE 求解器的每个操作进行反向传播，内存成本高。伴随方法将 ODE 求解器视为黑盒[^src-neural-ode]。

## 数学形式

设损失函数 L 依赖于 ODE 解 z(t1)[^src-neural-ode]：

$$L = L(z(t_0) + \int_{t_0}^{t_1} f(z(t), t, \theta)dt)$$

### 伴随状态

定义伴随状态 $a(t) = \partial L / \partial z(t)$，其动力学为：

$$\frac{da(t)}{dt} = -a(t)^T \frac{\partial f}{\partial z}(z(t), t, \theta)$$

### 参数梯度

$$\frac{\partial L}{\partial \theta} = -\int_{t_0}^{t_1} a(t)^T \frac{\partial f}{\partial z}(z(t), t, \theta) dt$$

## 实现

通过构造增广状态 $[z(t), a(t), \partial L/\partial \theta]$ 并反向求解 ODE[^src-neural-ode]：

```
def aug_dynamics([z(t), a(t), ·], t, θ):
    return [f(z(t), t, θ), 
            -a(t)^T ∂f/∂z, 
            -a(t)^T ∂f/∂θ]
```

## 优势

- **线性复杂度**：随问题规模线性增长
- **常数内存**：无需存储前向传播中间状态
- **精度可控**：可精确控制数值误差

## 链接

- [[neural-ordinary-differential-equation]] — Neural ODE
- [[continuous-normalizing-flow]] — 连续归一化流

[^src-neural-ode]: [[source-neural-ode]]