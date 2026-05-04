---
title: "Neural Ordinary Differential Equations"
type: source-summary
tags:
  - neural-ode
  - continuous-depth
  - ode-solver
  - normalizing-flows
  - neurips-2018
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Neural Ordinary Differential Equations

**论文信息**：Chen et al., NeurIPS 2018

## 核心贡献

Neural ODE 将隐藏状态的导数参数化为神经网络，通过黑盒 ODE 求解器计算输出[^src-neural-ode]。主要贡献包括：

1. **连续深度模型**：用 ODE dh(t)/dt = f(h(t), t, θ) 替代离散的残差连接
2. **伴随灵敏度方法**：通过求解增广 ODE 实现反向传播，无需存储中间状态
3. **连续归一化流 (CNF)**：推导出瞬时变量变换公式，避免归一化流的行列式计算瓶颈
4. **潜在 ODE 模型**：处理不规则采样时间序列的连续时间生成模型

## 主要成果

- MNIST 分类：测试误差 0.42%，参数量 0.22M，内存 O(1)
- 密度估计：CNF 在 Two Circles/Two Moons 数据集上优于离散 NF
- 时间序列：潜在 ODE 在不规则采样螺旋数据集上 RMSE 显著低于 RNN

## 局限性

- 小批量训练需要更精细的误差控制
- 需要选择合适的误差容限来平衡精度和速度

## 源文件
[^src-neural-ode]: [[source-neural-ode]]