---
title: "Adaptive Positional Encoding (APE)"
type: entity
tags:
  - position-encoding
  - extrapolation
  - llm
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

**Adaptive Positional Encoding (APE)** 是 Vetcha 2026 论文提出的位置编码方法，旨在实现无限长度上下文外推[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 核心公式

$$A_{APE}(n) = \underbrace{temp(n) \cdot q_i^\top R_{\alpha(n)}(n) k_j}_{\text{自适应乘法}} + \underbrace{b(n)}_{\text{自适应加性}}$$

其中[^src-vetcha-2026-towards-infinite-length-extrapolation]：

1. **频率适应**：旋转角度 $\theta(n) = n/\alpha(n)$，其中 $\alpha(n) \propto attention\_entropy(n)$——注意力分布越平坦，$\alpha(n)$ 越大

2. **温度调度**：$temp(n) = 1/(1 + \lambda|n|)$，$\lambda > 0$，对远距离注意力分数进行阻尼

3. **可学习加性偏置**：$b(n) = -\delta|n| - \beta\log(1+|n|) - \gamma\sqrt{|n|}$，结合线性、对数和平方根三项惩罚

## 理论性质

APE 满足无限上下文外推的四个关键性质[^src-vetcha-2026-towards-infinite-length-extrapolation]：

- **收敛归一化**：由于 $\delta > 0$ 和 $\gamma > 0$，$e^{A(n)}$ 随 $|n|$ 指数衰减
- **熵有界性**：由定理 3.5，收敛归一化 ⇒ 熵有界
- **梯度位置敏感性**：旋转矩阵和温度调度都显式依赖位置 $n$
- **局部 LDCP**：由于对数项和平方根项的次线性衰减，APE 的局部 LDCP 范围大于 ALiBi

## 与现有方法对比

| 方法 | 收敛归一化 | 熵有界 | LDCP | GPS |
|------|-----------|--------|------|-----|
| RoPE | ❌ | ❌ | ✅ | ✅ |
| ALiBi | ✅ | ✅ | ❌ | ❌ |
| APE | ✅ | ✅ | 局部 | ✅ |

## 实验表现

APE 在 TinyStories 和 LongTinyStories 数据集上表现优异：使用训练上下文窗口 64，可外推到 16,384（256 倍）仍保持低困惑度，优于 RoPE 和 ALiBi[^src-vetcha-2026-towards-infinite-length-extrapolation]。

---

[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]