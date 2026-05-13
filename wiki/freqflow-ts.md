---
title: "FrèqFlow (SpectFlow) — Lightweight Frequency-Domain Flow Matching for Time Series"
type: entity
tags:
  - flow-matching
  - frequency-domain
  - time-series-forecasting
  - traffic-forecasting
  - lightweight-model
  - neurips-2025
  - complex-valued
created: 2026-05-13
last_updated: 2026-05-13
source_count: 1
confidence: high
status: active
---

# FrèqFlow (SpectFlow)

**FrèqFlow**（别名 **SpectFlow**，Frequency-aware/Spectral Flow Matching）是由 Seyed Mohamad Moghadas 等人（Vrije Universiteit Brussel & imec）提出的多元时间序列确定性预测框架，发表于 NeurIPS 2025[^src-2511-16426]。其核心创新是将条件流匹配引入频域，通过单一复值线性层实现极轻量级（89k 参数）的 SOTA 预测。

## 架构设计

### 整体流程

1. **RIN 归一化** — 可逆实例归一化，消除非零均值导致的零频分量主导
2. **MHA 跨序列建模** — Multi-Head Attention 捕获变量间相关性
3. **rFFT 频域变换** — 将时域信号转换到复频域，得到 N/2 个复值系数
4. **低通滤波 (LPF)** — 保留 6 次谐波内的低频结构，丢弃高频噪声
5. **复值线性插值层** — 单一复值线性层将输入频谱映射到输出频谱（插值率 $\eta = L_o/L_i$），通过复乘法自然建模幅度缩放和相位平移
6. **零填充 + irFFT** — 补回 DC 分量，逆变换回时域
7. **流匹配头** — 在频域学习 velocity field $u_\theta(x_t, t)$，用于残差分量的精确估计

### 关键机制

- **频域流匹配**：将 flow matching 的 velocity field 学习从时域迁移到频域，使模型直接学习频谱分量的变换规律。线性插值路径 $x_t = (1-t)x_0 + tx_1$ 的目标速度 $u_t = x_1 - x_0$ 在频域中对应频谱的线性变化[^src-2511-16426]。
- **复值线性插值**：频域中的复乘法等价于时域中的幅度缩放和时间平移，使单一线性层即可建模复杂时序动力学[^src-2511-16426]。
- **残差学习**：受 Li et al. (2025) 启发，流匹配头专门设计用于残差分量的建模，频率插值头提供趋势和季节性[^src-2511-16426]。
- **Backcast 监督**：同时监督预测（forecast）和输入重建（backcast），消融实验证明可提升精度[^src-2511-16426]。

## 模型变体

提供两种配置，区别在于流匹配组件的深度[^src-2511-16426]：

| 变体 | 流匹配深度 | 参数量 |
|------|-----------|--------|
| FrèqFlow_S (Shallow) | 浅层 | 89k |
| FrèqFlow_D (Deep) | 深层 | 89k+ |

## 实验结果

在 Brussels、PEMS08、PEMS04 三个真实交通数据集上，预测时域 2/4/8 小时平均结果：

| 数据集 | 指标 | FrèqFlow_S | FrèqFlow_D | 最佳基线 | 提升 |
|--------|------|-------------|-------------|----------|------|
| Brussels | RMSE | 11.42 | 11.09 | 12.27 (Moirai-MoE) | 6.9-9.6% |
| Brussels | MAE | 6.78 | 6.18 | 7.05 (Moirai-MoE) | 3.8-12.3% |
| PEMS08 | RMSE | 24.50 | 24.19 | 25.16 (Moirai-MoE) | 2.6-3.9% |
| PEMS08 | MAE | 16.08 | 15.98 | 17.01 (Moirai-MoE) | 5.5-6.1% |
| PEMS04 | RMSE | 31.71 | 31.34 | 32.16 (Moirai-MoE) | 1.4-2.5% |
| PEMS04 | MAE | 21.11 | 20.93 | 22.28 (Moirai-MoE) | 5.3-6.1% |

超越的基线包括 GCRDD、DiffSTG、PriSTI、SpecSTG（扩散方法）以及 Moirai-MoE（基础模型）[^src-2511-16426]。

## 与相关工作的关系

### vs [[freqflow|FreqFlow（图像生成）]]
两者同名但完全不同：
- **FreqFlow (JHU/ByteDance, 2026)**：面向图像生成，双分支架构（频域分支+空域分支），507M-1.08B 参数
- **FrèqFlow/SpectFlow (VUB/imec, NeurIPS 2025)**：面向时间序列预测，单分支频域架构，89k 参数

### vs [[specstg|SpecSTG]]
- SpecSTG 在图谱域执行扩散过程，FrèqFlow 在傅里叶频域执行流匹配
- SpecSTG 使用图傅里叶变换 + U-Net，FrèqFlow 使用 rFFT + 复值线性层
- FrèqFlow 参数量远小于 SpecSTG

### vs [[flow-matching|Flow Matching]]
- FrèqFlow 将标准时域流匹配迁移到频域
- 核心数学框架（CFM、OT 路径、ODE 积分）保持不变
- 创新在于将 velocity field 学习从时域信号迁移到频谱分量

### vs [[generative-time-series-forecasting|生成式时间序列预测]]
- 现有生成式方法（DiffSTG、GCRDD、PriSTI）基于扩散模型，迭代采样开销大
- FrèqFlow 基于流匹配的 ODE 单次确定性采样，推理速度显著更快
- 89k 参数 vs 扩散模型的数百万参数

## 引用

[^src-2511-16426]: [[source-2511-16426]]