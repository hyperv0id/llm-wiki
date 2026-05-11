---
title: "Fourier Imputation Loss"
type: technique
tags:
  - loss-function
  - fourier-transform
  - low-rank
  - spatio-temporal
created: 2026-05-11
last_updated: 2026-05-11
source_count: 1
confidence: high
status: active
---

# Fourier Imputation Loss (FIL)

**Fourier Imputation Loss (FIL)** 是 ImputeFormer 提出的一种基于傅里叶变换的无监督频谱稀疏正则化损失函数，通过对填补结果的频域施加 ℓ1 稀疏约束来促进低秩结构[^src-2312-01728]。

## 动机

时空填补模型通常使用自监督掩蔽学习（在观测点上随机遮掩部分值并重建），但直接监督损失仅作用于被遮掩的观测点，对未观测（真正缺失）点缺乏约束。频域正则化通过利用数据的低秩先验间接作用于所有位置。

## 理论基础

### 卷积核范数与傅里叶 ℓ1 范数的等价性

对于光滑或周期时间序列 x ∈ R^T，其循环矩阵 C(x) ∈ R^(T×T) 的核范数可以通过离散傅里叶变换高效计算：

```
DFT(x) = Ux
‖DFT(x)‖_0 = rank(C(x))     — ℓ0 范数等于矩阵秩
‖DFT(x)‖_1 = ‖C(x)‖_*       — ℓ1 范数等于核范数（凸松弛）
```

其中 U ∈ C^(T×T) 为 DFT 矩阵，核范数 ‖C(x)‖_* 是矩阵秩的凸代理。

### 定义

给定填补结果 X̂ ∈ R^(N×T)、缺失掩码 M_missing、真实观测 Y：

```
X̄ = M_missing ⊙ X̂ + (1 − M_missing) ⊙ Y   // 完整估计（观测保留+填补补齐）
```

```
L_FIL = (1/NT) · ‖Flatten(FFT(X̄, dim=[0,1]))‖_1
```

其中 FFT 在空间和时间两个轴向上进行，Flatten 将频谱张量展开为向量后取 ℓ1 范数。

### 总损失

```
L = L_recon + λ · L_FIL
```

- L_recon = (1/NT) · ‖M_whiten ⊙ (X̂ − Y)‖_1：在自监督掩蔽点上的 ℓ1 重建损失
- L_FIL：对完整估计频谱的 ℓ1 稀疏正则化
- λ 为权重超参数（推荐值 0.005~0.05）

## 优势

1. **计算高效**：FFT 复杂度 O(NT log(NT))，远低于 SVD 的 O(min{N²T, NT²})
2. **全局约束**：L_recon 仅约束掩蔽观测点，L_FIL 通过频谱正则化间接约束所有位置（包括真正缺失点）
3. **信号-噪声平衡**：低秩模型过度平滑（截断过多能量），深度模型保留噪声（高频过多），FIL 在频域施加的 ℓ1 约束促使填补结果在奇异值频谱上达到平衡
4. **无需层次化损失**：替代了 GRIN/SPIN 等使用的层次化损失（每层都计算损失），避免过拟合的同时保持简单性

## 消融实验

- 移除 FIL 后，PEMS08 Point missing MAE 从 11.01 升至 11.35
- 移除 FIL 后，PEMS08 Block missing MAE 从 12.50 升至 13.07
- 过大 λ 值会导致过度平滑重建
- 最佳 λ：约 0.005~0.01（基于 PEMS08 实验）

## 关联页面

- [[imputeformer]] — ImputeFormer 模型
- [[projected-attention]] — 时间投影注意力
- [[embedded-attention]] — 空间嵌入注意力

[^src-2312-01728]: [[source-2312-01728]]
