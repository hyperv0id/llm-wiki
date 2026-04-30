---
title: "Newton-Schulz Iteration"
type: technique
tags:
  - matrix-iteration
  - orthogonalization
  - linear-algebra
  - numerical-methods
created: 2026-04-30
last_updated: 2026-04-30
source_count: 3
confidence: high
status: active
---

# Newton-Schulz Iteration

**Newton-Schulz 迭代**是一种用于将矩阵收敛到正交矩阵的数值方法[^src-muon-optimizer][^src-kexue-muon-analysis]。该方法由 Bernstein & Newhouse (2024) 在 Shampoo 优化器的分析中推广，是 Muon 优化器的核心技术。

---

## 背景：为什么需要矩阵正交化？

### 问题

在神经网络训练中，2D 参数（如线性层的权重矩阵）的梯度更新矩阵通常具有很高的**条件数**——即更新矩阵几乎是**低秩**的，所有神经元的更新都被少数方向主导[^src-muon-optimizer]。

### 解决思路

将更新矩阵 $\boldsymbol{M}$ 替换为最接近的**半正交矩阵**：

$$
\text{msign}(\boldsymbol{M}) = \mathop{\text{argmin}}_{\boldsymbol{O}^{\top}\boldsymbol{O} = \boldsymbol{I}} \Vert \boldsymbol{M} - \boldsymbol{O} \Vert_F^2
$$

这意味着将所有**奇异值变为 1**：

$$
\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} \quad \Rightarrow \quad \text{msign}(\boldsymbol{M}) = \boldsymbol{U}\boldsymbol{V}^{\top}
$$

---

## 数学基础

### 1. 矩阵符号函数的定义

对于任意矩阵 $\boldsymbol{M}$，其 SVD 为 $\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，则：

$$
\text{msign}(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}
$$

其中 $r$ 是矩阵的秩。

### 2. 恒等式

利用 SVD 可以证明[^src-kexue-muon-analysis]：

$$
\text{msign}(\boldsymbol{M}) = (\boldsymbol{M}\boldsymbol{M}^{\top})^{-1/2}\boldsymbol{M} = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}
$$

**标量特殊情况**：当 $\boldsymbol{M}$ 是 $1\times 1$ 矩阵（即标量 $x$）时：
$$
\text{msign}(x) = x(x^2)^{-1/2} = \text{sign}(x)
$$

这正是 $\text{sign}$ 函数的矩阵推广。

### 3. 泰勒展开视角

从恒等式出发，在 $\boldsymbol{M}^{\top}\boldsymbol{M} = \boldsymbol{I}$ 处泰勒展开 $(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}$：

$$
t^{-1/2} = 1 - \frac{1}{2}(t-1) + \frac{3}{8}(t-1)^2 - \frac{5}{16}(t-1)^3 + \cdots
$$

保留到二阶：
$$
(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2} \approx \frac{15}{8}\boldsymbol{I} - \frac{5}{4}\boldsymbol{M}^{\top}\boldsymbol{M} + \frac{3}{8}(\boldsymbol{M}^{\top}\boldsymbol{M})^2
$$

---

## Newton-Schulz 迭代算法

### 标准形式

```python
def newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz 迭代：通过迭代将矩阵趋近于正交矩阵
    
    原理：迭代奇异值映射 g(x) = ax + bx³ + cx⁵
         使得无论初始奇异值大于1还是小于1，都收敛到1
    
    参数:
        G: 输入矩阵 (n×m)
        steps: 迭代步数 (通常用5步)
        eps: 数值稳定性参数
    返回:
        近似正交矩阵
    """
    # 调优系数：经过优化得到，收敛最快
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # 1. 转换为 bfloat16 加速计算（可在 bfloat16 下稳定运行）
    X = G.bfloat16()
    
    # 2. 归一化：确保所有奇异值在 [0,1] 范围内
    X /= (X.norm() + eps)
    
    # 3. 转置处理：如果行数大于列数，转置以优化计算
    if G.size(0) > G.size(1):
        X = X.T
    
    # 4. 迭代更新
    for _ in range(steps):
        A = X @ X.T           # X @ X.T 是 n×n 矩阵
        B = b * A + c * A @ A # 高阶项
        X = a * X + B @ X     # 组合得到新近似
    
    # 5. 转置回去
    if G.size(0) > G.size(1):
        X = X.T
    
    return X
```

### 系数的选择（详细）

系数 (3.4445, -4.7750, 2.0315) 是通过**优化**得到的[^src-kexue-muon-analysis]：

#### 迭代函数形式

设 $g(x) = ax + bx^3 + cx^5$，这是奇异值的迭代映射。

#### 优化目标

选择 $a, b, c$ 使得：
- 无论初始奇异值 $x \in (0, \infty)$ 大于 1 还是小于 1
- 迭代 $T$ 次后都尽可能接近 1

#### 重新参数化

将 $a, b, c$ 重新参数化为：
$$
g(x) = x + \kappa x(x^2 - x_1^2)(x^2 - x_2^2)
$$

其中 $x_1 < 1 < x_2$ 是两个不动点。

#### 损失函数

$$
\mathcal{L} = \mathbb{E}[(g^{(T)}(x) - 1)^2]
$$

对所有可能的奇异值取期望，通过梯度下降优化 $\kappa, x_1, x_2$。

#### 为什么是 5 步？

| 步数 T | mse (1024×1024) | mse (2048×1024) |
|--------|-----------------|-----------------|
| 3 | 0.18278 | 0.06171 |
| **5** | **0.04431** | **0.02954** |

5 步在精度和速度之间取得最佳平衡[^src-kexue-muon-analysis]。

---

## 迭代的数学性质

### 对奇异值的作用

设 $\boldsymbol{X}_t = \boldsymbol{U}\boldsymbol{\Sigma}_t\boldsymbol{V}^{\top}$，则：
$$
\boldsymbol{X}_{t+1} = \boldsymbol{U}(a\boldsymbol{\Sigma}_t + b\boldsymbol{\Sigma}_t^3 + c\boldsymbol{\Sigma}_t^5)\boldsymbol{V}^{\top}
$$

即：**迭代只作用于奇异值，不改变奇异向量**。

### 收敛性

- 如果 $0 < \sigma_0 < \infty$，则 $\sigma_t \to 1$ 当 $t \to \infty$
- 收敛速度由多项式 $g(x)$ 的性质决定
- 调优系数使收敛更快、更稳定

### 与 SVD 的关系

| 方法 | 精度 | 速度 | 稳定性 |
|------|------|------|--------|
| 直接 SVD | 精确 | 慢 $O(n^3)$ | 高 |
| Newton-Schulz | 近似 | 快 $O(nm^2)$ | 高 (bfloat16) |

---

## FLOP 开销分析

对于形状为 $(n, m)$ 的参数（设 $n \geq m$）：

每步迭代：
- $\boldsymbol{X}\boldsymbol{X}^{\top}$：$n^2 m$ FLOPs
- $\boldsymbol{A}^2$：$n^3$ FLOPs
- 矩阵乘法：$n^2 m$ FLOPs

总计约 $O(n^2 m)$ FLOPs，相比前向传播的 $O(nmk)$（$k$ 是 batch size）占比极小。

典型开销：**< 1%**[^src-muon-optimizer]

---

## 为什么选择 Newton-Schulz？

### 1. 计算效率

- 可以在 **bfloat16** 精度下稳定运行
- 矩阵乘法可以利用 GPU 并行
- 比直接 SVD 快几个数量级

### 2. 数值稳定性

- 耦合 Newton 迭代需要 float32 精度
- Newton-Schulz 对初始值不敏感

### 3. 实用优势

- 静态大小的矩阵运算，GPU 优化良好
- 不需要额外的缓存变量（比 Adam 更省显存）

---

## 使用示例

```python
import torch
import torch.nn as nn

class Muon(nn.Module):
    def __init__(self, params):
        self.params = list(params)
        self.m = [torch.zeros_like(p) for p in params]
        self.beta = 0.9
        
    def step(self):
        for i, p in enumerate(self.params):
            if p.ndim < 2:  # 非矩阵参数跳过
                continue
            g = p.grad
            if g is None:
                continue
                
            # 动量更新
            self.m[i] = self.beta * self.m[i] + g
            
            # Newton-Schulz 正交化
            m_normalized = self.m[i] / (self.m[i].norm() + 1e-7)
            m_orthogonal = newtonschulz5(m_normalized, steps=5)
            
            # 参数更新
            p.data -= 0.01 * m_orthogonal
```

---

## 参考代码（���学空间）

```python
# 参考：https://kexue.fm/archives/10592
import jax
import jax.numpy as jnp

n, m, T = 1024, 1024, 5

# 采样奇异值
data = jnp.array([])
for _ in range(1000):
    M = jax.random.normal(key, shape=(n, m))
    S = jnp.linalg.svd(M, full_matrices=False)[1]
    data = jnp.concatenate([data, S / (S**2).sum()**0.5])

# 优化系数
def f(w, x):
    k, x1, x2 = w
    for _ in range(T):
        x = x + k * x * (x**2 - x1**2) * (x**2 - x2**2)
    return ((x - 1)**2).mean()

# 梯度下降优化
w = jnp.array([1, 0.9, 1.1])
for _ in range(100000):
    w = w - 0.01 * jax.grad(f)(w, data)
```

---

## 参考文献

[^src-muon-optimizer]: [[source-muon-optimizer]]
[^src-kexue-muon-analysis]: [[source-kexue-muon-analysis]]