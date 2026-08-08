---
title: "Variant-Frequency Positional Encoding"
type: technique
tags:
  - position-encoding
  - sinusoidal
  - physics-informed
  - frequency
created: 2025-07-14
last_updated: 2026-08-08
source_count: 1
confidence: medium
status: active
---

# Variant-Frequency Positional Encoding

**Variant-Frequency Positional Encoding（变频率位置编码）**是 PIPE 的第二核心组件，通过为不同物理变量分配不同波长的正弦函数，将物理变量的频率特性编码到位置嵌入空间中[^src-pipe]。

## 动机

标准正弦位置编码中，所有维度共享相同的频率进程（几何级数从 $1$ 到 $10000$），只编码序列顺序。但不同物理变量有天然不同的周期性特征：一天 24 小时、一年 366 天、纬度 180°、经度 360°——这些周期应当通过不同的波长反映在位置嵌入中[^src-pipe]。

## 公式

变频率正弦函数修改标准正弦编码的波长参数：

$$PE(pos, 2i) = \sin\left(\frac{2\pi \cdot pos}{p \cdot 10000^{2i/d_{\text{model}}}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{2\pi \cdot pos}{p \cdot 10000^{2i/d_{\text{model}}}}\right)$$

其中波长参数 $p$ 取决于物理变量：

| 物理变量 | 范围 | 波长 $p$ |
|---------|------|----------|
| $t_{\text{day}}$ (年日) | $[0, 365]$ | 366 |
| $t_{\text{hour}}$ (小时) | $[0, 23]$ | 24 |
| $\text{lat}$ (纬度) | $[0, 180]$ | 180 |
| $\text{lng}$ (经度) | $[0, 360]$ | 360 |

波长形成几何级数 $p \to p \cdot 10000/2$，使不同物理变量的频率域相互分离[^src-pipe]。

## 维度分配

在 PIPE-3B 中，$d_{\text{model}}$ 的维度被划分为两半：前半分配给时间维度（t_day + t_hour），后半分配给空间维度（latitude + longitude）。每 4 个维度为一组，依次编码 t_day、t_hour、lat、lng 的 sin/cos 分量（论文 Figure 4 以 $d_{\text{model}}=128$ 为例可视化）[^src-pipe]。

## 与标准正弦编码的对比

| 属性 | 标准正弦编码 | 变频率正弦编码 |
|------|------------|--------------|
| 编码对象 | 序列位置 | 物理量 |
| 波长起点 | $10000^{0/d}$ | $p \cdot 10000^{0/d}$ |
| 几何级数 | $1 \to 10000$ | $p \to p \cdot 10000/2$ |
| 多变量区分 | 不支持 | 通过不同 $p$ 区分 |
| 物理可解释性 | ❌ | ✅ |

## 消融结果

在 PIPE 的消融实验中，移除整个正弦函数导致强度 MAE 从 1.515 升至 1.545；仅移除变频率变体（保留标准正弦）导致升至 1.639。变频率设计相对完整 PIPE 贡献约 7.6%（1.639→1.515）；论文报告物理知情索引贡献约 6%（1.617→1.515）[^src-pipe]。

## 相关页面

- [[physics-informed-position-encoding]] — 物理知情位置编码的完整框架
- [[pipe]] — PIPE 模型
- [[source-pipe]] — 论文摘要
- [[rope]] — RoPE（PIPE 中与变频率编码配合使用）

[^src-pipe]: [[source-pipe]]
