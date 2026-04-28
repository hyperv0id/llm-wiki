---
title: "DPM-Solver"
type: entity
tags:
  - diffusion-models
  - sampling
  - fast-inference
  - nips-2022
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# DPM-Solver

**DPM-Solver** 是扩散模型快速采样的里程碑工作，由 Lu Cheng 等人于 2022 年发表在 NeurIPS。该方法利用扩散 ODE 的半线性结构，通过解析计算线性部分实现快速高质量采样[^src-dpm-solver]。

## 核心创新

1. **精确解公式**：利用常数变易公式和变量替换，精确计算扩散 ODE 的线性部分
2. **高阶求解器**：DPM-Solver-1/2/3 分别对应一/二/三阶收敛
3. **DDIM 等价**：证明 DDIM 是 DPM-Solver-1 的特例
4. **训练免费**：无需额外训练，可直接应用于任意预训练模型

## 技术细节

### 半线性 ODE 结构

扩散 ODE：
$$
\frac{dx}{dt} = f(t)x + \frac{g(t)^2}{2\sigma_t}\varepsilon_\theta(x, t)
$$

- 线性项：$f(t)x$
- 非线性项：神经网络预测的噪声 $\varepsilon_\theta$

### 关键变量替换

$\lambda_t = \log(\alpha_t / \sigma_t)$ — 半对数信噪比

### 求解器更新公式

**DPM-Solver-1**:
$$
\tilde{x}_{t_i} = \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{x}_{t_{i-1}} - \sigma_{t_i}(e^{h_i} - 1)\varepsilon_\theta(\tilde{x}_{t_{i-1}}, t_{i-1})
$$

其中 $h_i = \lambda_{t_i} - \lambda_{t_{i-1}}$

## 实验结果

| 数据集 | NFE | FID |
|--------|-----|-----|
| CIFAR-10 | 10 | 4.70 |
| CIFAR-10 | 20 | 2.87 |
| ImageNet 256×256 | 10 | 优秀质量 |

## 相关页面

- [[diffusion-model]] — 扩散模型概念
- [[ddpm]] — DDPM，原始采样需要 ~1000 步
- [[score-based-sde]] — Score-Based SDE，ODE 求解基础
- [[probability-flow-ode]] — 概率流 ODE
- [[ddim]] — DDIM，等价于 DPM-Solver-1

## 引用

[^src-dpm-solver]: [[source-dpm-solver]]