---
title: "Muon Optimizer"
type: entity
tags:
  - optimizer
  - neural-network
  - orthogonalization
  - matrix-optimization
created: 2026-04-30
last_updated: 2026-04-30
source_count: 3
confidence: high
status: active
---

# Muon Optimizer

**Muon** (MomentUm Orthogonalized by Newton-Schulz) 是一种专门用于神经网络**隐藏层 2D 参数**的优化器[^src-muon-optimizer][^src-kellerjordan-muon-blog]。它在当前 NanoGPT 和 CIFAR-10 速度训练纪录中使用，是大模型时代备受关注的优化器[^src-muon-optimizer]。

> [!tip] 为什么 Muon 值得关注
> 
> Muon 不仅仅是对 AdamW 的小修补，而是体现了**向量优化与矩阵优化的本质差异**——这是一个值得深思的根本性原理[^src-kexue-muon-analysis]。

---

## 背景：为什么需要新的优化器？

### AdamW 的局限性

AdamW 是当前大模型训练的主流优化器，但它本质上是 **Element-wise**（逐元素）的优化器：
- 将矩阵参数展平为向量
- 每个分量独立更新
- 通过梯度平方的滑动平均来调整学习率

这种设计对于**向量参数**是合理的，但**矩阵参数**有独特的结构[^src-kexue-muon-analysis]：

1. **矩阵有非对角线元素**：向量只有位置信息，矩阵还有行列交互
2. **奇异值分布**：矩阵的"各向异性"由奇异值体现
3. **谱范数**：矩阵之间更本质的度量是谱范数（最大奇异值），而非逐元素差异

### Muon 的核心洞见

Muon 的关键创新是：**以矩阵为基本单位进行优化**，而非将矩阵展平为向量[^src-kexue-muon-analysis]。

---

## 核心机制

### 1. 矩阵符号函数 (msign)

Muon 的核心是**矩阵符号函数** (matrix sign function, msign)：

$$
\text{msign}(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}
$$

其中 $\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ 是 $\boldsymbol{M}$ 的奇异值分解 (SVD)[^src-kexue-muon-analysis]。

**为什么 msign 重要？**

1. **奇异值置一**：将所有奇异值变为 1，使更新更"各向同性"
2. **自适应学习率**：损失函数乘以 $\lambda$ 不影响优化轨迹
3. **同步更新幅度**：让不同方向的更新幅度尽可能一致

### 2. 更新规则

完整的 Muon 更新规则：

$$
\begin{aligned}
\boldsymbol{M}_t &= \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\
\boldsymbol{W}_t &= \boldsymbol{W}_{t-1} - \eta_t [\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}]
\end{aligned}
$$

其中：
- $\boldsymbol{G}_t$：当前梯度
- $\boldsymbol{M}_t$：动量矩阵
- $\beta$：动量系数（通常 0.9）
- $\eta_t$：学习率
- $\lambda$：权重衰减系数
- $\text{msign}$：矩阵符号函数

### 3. 动量顺序的关键差异

| 方法 | 顺序 | 效果 |
|------|------|------|
| Orthogonal-SGDM | 正交化 → 动量 | 较差 |
| **Muon** | **动量 → 正交化** | **更好** |

Muon 的关键设计：在正交化**之前**应用动量，这比之前的方法效果更好[^src-muon-optimizer]。

---

## Newton-Schulz 迭代

直接计算 SVD 开销太大，Muon 使用 **Newton-Schulz 迭代**来近似计算 $\text{msign}(\boldsymbol{M})$[^src-muon-optimizer][^src-kexue-muon-analysis]。

### 算法

```python
def newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz 迭代近似矩阵符号函数
    
    参数:
        G: 输入矩阵
        steps: 迭代步数 (通常用 5)
        eps: 数值稳定性参数
    返回:
        近似正交矩阵
    """
    a, b, c = (3.4445, -4.7750, 2.0315)  # 调优系数
    X = G.bfloat16()  # 可以在 bfloat16 下稳定运行
    X /= (X.norm() + eps)  # 归一化，确保奇异值在 [0,1]
    
    if G.size(0) > G.size(1):
        X = X.T  # 转置确保矩阵形状适合迭代
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    
    return X
```

### 系数的选择

系数 (3.4445, -4.7750, 2.0315) 是通过优化得到的目标：让迭代对任意初始奇异值都**尽可能快地收敛到 1**[^src-kexue-muon-analysis]。

详细选择过程：
- 重新参数化：$g(x) = x + \kappa x(x^2 - x_1^2)(x^2 - x_2^2)$
- 目标：无论奇异值初始大于 1 还是小于 1，迭代后都趋近于 1
- 损失函数：平方误差 $((x - 1)^2).mean()$
- 优化方法：梯度下降

### 为什么 Newton-Schulz 比 SVD 快？

| 方法 | 精度要求 | 计算复杂度 | 稳定性 |
|------|----------|------------|--------|
| SVD | float32 | $O(n^3)$ | 高 |
| 耦合 Newton 迭代 | float32 | $O(n^2)$ | 中 |
| **Newton-Schulz** | **bfloat16** | **$O(nm^2)$** | **高** |

Newton-Schulz 可以在 **bfloat16** 下稳定运行，这让它在现代 GPU 上非常高效[^src-muon-optimizer]。

---

## 谱范数视角：为什么 Muon 有效？

这是科学空间文章的核心贡献[^src-kexue-muon-analysis]。

### 从最速下降说起

梯度下降可以理解为在某种范数约束下的最速下降：

$$
\Delta\boldsymbol{W} = \mathop{\text{argmin}}_{\Delta\boldsymbol{W}} \frac{\Vert\Delta\boldsymbol{W}\Vert^2}{2\eta} + \text{Tr}(\boldsymbol{G}^{\top}\Delta\boldsymbol{W})
$$

### 向量 vs 矩阵

- **向量**：常用 $p$ 范数 $\Vert\boldsymbol{v}\Vert_p$
  - $p=2$：欧氏范数 → SGD
  - $p\to\infty$：无穷范数 → SignSGD
- **矩阵**：常用两种范数
  - **Frobenius 范数**：将矩阵展平为向量的欧氏范数 → 等价于 SGD
  - **谱范数（2-范数）**：$\Vert\boldsymbol{\Phi}\Vert_2 = \max_{\Vert\boldsymbol{x}\Vert_2=1} \Vert\boldsymbol{\Phi}\boldsymbol{x}\Vert_2$

### 关键结论

> **Muon = 谱范数约束下的梯度下降**

推导过程：
1. 使用谱范数作为约束
2. 通过 SVD 和对偶范数分析
3. 得到最优更新方向：$\text{msign}(\boldsymbol{G})$

这解释了为什么 Muon 相比 AdamW 更有效：**谱范数更好地度量了矩阵之间的本质差异**[^src-kexue-muon-analysis]。

---

## 与其他优化器的关系

### 1. SignSGD / Tiger

当矩阵 $\boldsymbol{M}$ 是**对角阵**时：
- $\text{msign}(\boldsymbol{M}) = \text{sign}(\boldsymbol{M})$（逐元素取符号）
- **Muon 退化为 SignSGD 或 Tiger**[^src-kexue-muon-analysis]

这说明 Muon 是 SignSGD 的矩阵推广。

### 2. Shampoo (Gupta et al., 2018)

Shampoo 的更新规则：
$$
\boldsymbol{W}_{t+1} = \boldsymbol{W}_t - \eta_t \boldsymbol{L}_t^{-1/4}\boldsymbol{G}_t\boldsymbol{R}_t^{-1/4}
$$

其中 $\boldsymbol{L}_t = \beta\boldsymbol{L}_{t-1} + \boldsymbol{G}_t\boldsymbol{G}_t^{\top}$，$\boldsymbol{R}_t = \beta\boldsymbol{R}_{t-1} + \boldsymbol{G}_t^{\top}\boldsymbol{G}_t$

**关键发现**：当 $\beta = 0$ 时：
$$
\boldsymbol{L}_0^{-1/4}\boldsymbol{G}_0\boldsymbol{R}_0^{-1/4} = \text{msign}(\boldsymbol{G}_0)
$$

因此：**$\beta=0$ 时，Shampoo = Muon**[^src-kexue-muon-analysis]

可以认为 Muon 是 **无累积的"即时"Shampoo**：
- Shampoo：缓存 $\boldsymbol{G}\boldsymbol{G}^{\top}$ 长期累积
- Muon：每次直接计算 msign，无需累积

### 3. Stochastic Spectral Descent (2015)

Muon 与 2015 年的论��� "Stochastic Spectral Descent for Restricted Boltzmann Machines" 大致相同[^src-muon-optimizer][^src-kexue-muon-analysis]。

---

## 实验结果

| 任务 | 指标 | 提升 |
|------|------|------|
| CIFAR-10 (94% 准确率) | A100-秒 | 3.3 → 2.6 (**21%** 提升) |
| NanoGPT 速度训练 | val loss 3.28 | **1.35x** 提升 |
| 1.5B 模型训练 | 8xH100 小时 | 10h vs 13.3h AdamW |

**扩展性**：成功扩展至 774M 和 1.5B 参数模型[^src-muon-optimizer]

---

## 使用注意事项

### 适用场景

- ✅ **隐藏层的 2D 线性参数**（Q、K、V、W 等）
- ✅ **卷积参数**（展平最后三个维度后）

### 不适用场景

- ❌ **标量和向量参数**（使用 AdamW）
- ❌ **Embedding 层**（使用 AdamW）
- ❌ **输出分类头**（使用 AdamW）

### 实践技巧

1. **QKV 分开优化**：对 Q、K、V 参数分别应用 Muon，效果更好[^src-muon-optimizer]
2. **使用 Nesterov 动量**：比普通 SGD-动量效果更好[^src-muon-optimizer]
3. **权重衰减**：同样应用权重衰减

### 计算开销

- **时间**：比 Adam 增加约 5%（作者声称 2%）
- **显存**：比 Adam 少一组缓存变量（因为不需要 Adam 的二阶矩估计）[^src-kexue-muon-analysis]

---

## FLOP 开销分析

对于形状为 $(n, m)$ 的参数：

- 每步 NS 迭代需要 $O(nm^2)$ FLOPs
- 典型 LM 训练场景：FLOP 开销 < **1%**[^src-muon-optimizer]

计算示例（NanoGPT 速度训练）：
- 模型维度 $d = 768$
- Batch token 数 = 512
- 开销 = $5 \times 768 / 512 \approx 0.8\%$

---

## 开源实现

- **PyTorch 实现**：https://github.com/KellerJordan/Muon
- **NanoGPT 速度训练**：https://github.com/KellerJordan/modded-nanogpt

---

## 总结

| 特性 | 说明 |
|------|------|
| **全称** | MomentUm Orthogonalized by Newton-Schulz |
| **核心思想** | 对动量更新矩阵进行正交化 |
| **理论基础** | 谱范数约束下的最速下降 |
| **与 AdamW 关系** | 对隐藏层 2D 参数更有效 |
| **计算开销** | < 1% FLOPs |
| **显存开销** | 比 AdamW 更低 |

---

## 参考文献

[^src-muon-optimizer]: [[source-muon-optimizer]]
[^src-kellerjordan-muon-blog]: [[source-kellerjordan-muon-blog]]
[^src-kexue-muon-analysis]: [[source-kexue-muon-analysis]]