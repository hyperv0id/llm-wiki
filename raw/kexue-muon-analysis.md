---
created: 2026-04-30T10:16:14 (UTC +08:00)
tags: [矩阵, 梯度, 优化器, 谱范数, muon]
source: https://kexue.fm/archives/10592
author: 苏剑林
type: url
---

# Muon优化器赏析：从向量到矩阵的本质跨越 - 科学空间|Scientific Spaces

> ## Excerpt
> 随着LLM时代的到来，学术界对于优化器的研究热情似乎有所减退。这主要是因为目前主流的AdamW已经能够满足大多数需求，而如果对优化器"大动干戈"，那么需要巨大的验证成本。因此，当前优化器的变化，多数都只是工业界根据自己的训练经验来对AdamW打的一些小补丁。
> 
> 不过，最近推特上一个名为"Muon"的优化器颇为热闹，它声称比AdamW更为高效，且并不只是在Adam基础上的"小打小闹"，而是体现了关于向量与矩阵差异的一些值得深思的原理。本文让我们一起赏析一番。

---
随着LLM时代的到来，学术界对于优化器的研究热情似乎有所减退。这主要是因为目前主流的AdamW已经能够满足大多数需求，而如果对优化器"大动干戈"，那么需要巨大的验证成本。因此，当前优化器的变化，多数都只是工业界根据自己的训练经验来对AdamW打的一些小补丁。

不过，最近推特上一个名为"[Muon](https://github.com/KellerJordan/Muon)"的优化器颇为热闹，它声称比AdamW更为高效，且并不只是在Adam基础上的"小打小闹"，而是体现了关于向量与矩阵差异的一些值得深思的原理。本文让我们一起赏析一番。

## 算法初探

Muon全称是"MomentUm Orthogonalized by Newton-schulz"，它适用于矩阵参数$\boldsymbol{W}\in\mathbb{R}^{n\times m}$，其更新规则是  

$$
\begin{equation}\begin{aligned} 
\boldsymbol{M}_t =&\, \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\[5pt] 
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t [\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}] \\ 
\end{aligned}
\end{equation}
$$

这里$\text{msign}$是矩阵符号函数，它跟SVD的关系是：  

$$
\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}^{\top} = \text{SVD}(\boldsymbol{M}) \quad\Rightarrow\quad \text{msign}(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}
$$

## 关键理论洞见

1. **自适应学习率特性**：Muon通过将奇异值置一，实现了两个效果：损失函数的常数缩放不影响优化轨迹；每个参数分量的更新幅度尽可能一致。

2. **与SignSGD的关系**：当矩阵是对角阵时，Muon退化为SignSGD或Tiger。

3. **正交近似**：$\text{msign}(\boldsymbol{M})$ 等价于在Frobenius范数下找到最接近的正交矩阵。

4. **谱范数视角**：Muon相当于谱范数（2-范数）约束下的梯度下降，这更好地度量了矩阵之间的本质差异。

5. **与Shampoo的关系**：当Shampoo的$\beta=0$时，两者等价。

## 范数视角

从最速下降的角度看，Muon体现了向量与矩阵的本质差异：
- 向量优化使用Element-wise更新（如SGD、Adam）
- 矩阵优化使用谱范数约束，捕捉矩阵的非对角线特性

矩阵的谱范数：$\Vert \boldsymbol{\Phi}\Vert_2 = \max_{\Vert \boldsymbol{x}\Vert_2 = 1} \Vert \boldsymbol{\Phi}\boldsymbol{x}\Vert_2$

在谱范数约束下，可以推导出$\text{msign}(\boldsymbol{G})$正是最优更新方向。

## 与Shampoo的关系

Shampoo (Gupta et al., 2018) 缓存梯度外积$\boldsymbol{G}\boldsymbol{G}^{\top}$和$\boldsymbol{G}^{\top}\boldsymbol{G}$，而当$\beta=0$时：

$$
(\boldsymbol{G}\boldsymbol{G}^{\top})^{-1/4}\boldsymbol{G}(\boldsymbol{G}^{\top}\boldsymbol{G})^{-1/4} = \text{msign}(\boldsymbol{G})
$$

这表明Muon与Shampoo在$\beta=0$时等价。

## 参考文献

- Muon: https://github.com/KellerJordan/Muon
- Stochastic Spectral Descent: https://proceedings.mlr.press/v38/carlson15.html
- Shampoo: https://arxiv.org/abs/1802.09568
- Old Optimizer, New Norm: https://arxiv.org/abs/2409.20325
- BERT-whitening: https://kexue.fm/archives/8069