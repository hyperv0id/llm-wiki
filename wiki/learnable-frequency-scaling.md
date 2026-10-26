---
title: "Learnable Frequency Scaling"
type: technique
tags:
  - positional-encoding
  - rotary-position-embedding
  - frequency
  - learnable
created: 2026-04-29
last_updated: 2026-10-26
source_count: 2
confidence: medium
status: active
---

# Learnable Frequency Scaling

**可学习频率缩放** 是 SIREN-RoPE 的一项技术创新，用可学习的每维频率缩放 $\omega_j^s$ 替代标准 RoPE[^src-roformer] 中的固定逆频率常数[^src-siren-rope]。

## 背景：标准 RoPE 频率

标准 RoPE 使用固定的逆频率 schedule：
$$\theta_j = base^{-2j/d_k}, \quad j = 0, 1, \dots, d_k/2 - 1$$

- 原始设定 $base = 10^4$[^src-roformer]
- 频率沿维度按几何级数递减：$j$ 小的一对维度每走一个位置就转过接近 1 弧度，只分辨相邻 token；$j$ 大的一对旋转慢、波长长，跨越大范围位置才转过一圈。位置分辨率由低维承担，长程相位由高维承担[^src-roformer]

> [!note] 更正
> 本页旧版写「高维编码细粒度位置信息[^src-siren-rope]」，方向与 $\theta_j=base^{-2j/d_k}$ 的定义相反：$j$ 大则 $\theta_j$ 小、旋转慢、波长大，承担的是长程相位；短程分辨率由 $j$ 小的维度承担[^src-roformer]。原句所引的 SIREN-RoPE 源只给出 schedule，未讨论这一方向。

## SIREN-RoPE 的改进

$$\Theta_j(T_i) = f_\phi(T_i)_j \cdot \omega_j^s + p_i \cdot \theta_j \cdot \lambda$$

引入可学习的每维频率缩放 $\omega_j^s$：
- 初始化为 $\pi$
- 通过梯度下降端到端学习
- 允许模型自适应调整不同维度的时间敏感度[^src-siren-rope]

## 学习到的模式

实验发现（通过 FFT 分析）：
- 模型学习到日周期频率 (1 cycle/day)
- 模型学习到周周期频率 (1/7 cycle/day)
- 次谐波峰值出现在 1/2、1/3 等位置[^src-siren-rope]

## 与其他频率自适应方法的对比

| 方法 | 频率策略 | 可学习 |
|------|----------|--------|
| 标准 RoPE | 固定逆频率 $\theta_j = base^{-2j/d_k}$ | ❌ |
| Position Interpolation | 等比例拉伸所有维度 | ❌ |
| NTK-aware | 改变 base 参数 | ❌ |
| **可学习频率缩放** | 每维独立 $\omega_j^s$ | ✅ |

## 组合效果

可学习频率缩放 + SIREN 周期性分支 + 可学习门控 $\lambda$：
- **周期性分支**：捕获离散周期（日/周）
- **频率缩放**：调整各维度对时间信号的响应强度
- **门控 $\lambda$**：控制序数分量的相对贡献[^src-siren-rope]

## 相关页面

- [[rope]] — 标准 RoPE 技术详解
- [[siren-rope]] — 主 entity 页面
- [[ordinal-temporal-fusion]] — 融合公式
- [[dual-branch-siren]] — 架构详情
- [[ntk-aware-interpolation]] — 另一种 RoPE 频率调整方法
- [[yarn]] — 上下文扩展方法

[^src-roformer]: [[source-roformer]]
[^src-siren-rope]: [[source-siren-rope]]