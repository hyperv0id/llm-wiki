---
title: "Attention Temperature Scaling"
type: technique
tags:
  - attention
  - rope
  - context-extension
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# Attention Temperature Scaling

注意力温度缩放是 YaRN 方法的关键创新，通过在 softmax 前对注意力 logits 除以温度 $t$ 来控制长距离注意力的熵[^src-yarn]。

## 数学定义

标准注意力计算[^src-yarn]：

$$\text{softmax}\left(\frac{q_m^T k_n}{\sqrt{|D|}}\right)$$

YaRN 修改为[^src-yarn]：

$$\text{softmax}\left(\frac{q_m^T k_n}{t\sqrt{|D|}}\right)$$

## 温度公式

对于缩放因子 $s$，推荐温度由经验公式给出[^src-yarn]：

$$\sqrt{\frac{1}{t}} = 0.1\ln(s) + 1$$

示例：
- $s=2$：$\sqrt{1/t} \approx 1.07$，$t \approx 0.87$
- $s=8$：$\sqrt{1/t} \approx 1.21$，$t \approx 0.68$
- $s=16$：$\sqrt{1/t} \approx 1.28$，$t \approx 0.61$
- $s=32$：$\sqrt{1/t} \approx 1.35$，$t \approx 0.55$

## 工作原理

1. **$t < 1$ 时**：温度降低 → softmax 更尖锐 → 注意力更集中[^src-yarn]
2. **长上下文效应**：在扩展后的长上下文中，降低温度有助于模型聚焦于真正相关的 token[^src-yarn]
3. **普适性**：同一温度公式在 LLaMA 7B-65B 和 Llama 2 7B-70B 上均有效[^src-yarn]

## 实现技巧

通过"长度缩放"技巧实现零开销[^src-yarn]：

- 将 $q_m$ 和 $k_n$ 同时乘以 $\sqrt{1/t}$
- 等价于缩放 RoPE 的复数旋转嵌入
- 无需修改注意力代码
- 训练和推理均无额外开销

## 实验验证

在 896 个 16k-token 文档上的实验表明[^src-yarn]：

- 合适的 $t$ 值在整个扩展上下文窗口内一致提升 perplexity
- 最优 $t$ 在不同样本和不同位置间高度一致
- 温度缩放的效果独立于数据样本和 token 位置

## 与 NTK-by-parts 的关系

温度缩放与 [[ntk-by-parts-interpolation|NTK-by-parts]] 插值互补[^src-yarn]：
- NTK-by-parts 处理位置编码的频率分布
- 温度缩放处理注意力的熵/集中度
- 两者结合 = [[yarn|YaRN]]

## 引用

[^src-yarn]: [[source-yarn]]