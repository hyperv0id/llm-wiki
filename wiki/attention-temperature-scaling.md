---
title: "Attention Temperature Scaling"
type: technique
tags:
  - attention
  - rope
  - context-extension
created: 2026-04-28
last_updated: 2026-10-26
source_count: 3
confidence: medium
status: active
---

# Attention Temperature Scaling

注意力温度缩放是 YaRN 方法的关键创新，通过在 softmax 前对注意力 logits 除以温度 $t$ 来控制长距离注意力的熵[^src-yarn]。

## 数学定义

标准注意力计算[^src-yarn]：

$$\text{softmax}\left(\frac{q_m^T k_n}{\sqrt{|D|}}\right)$$

YaRN 修改为[^src-yarn]：

$$\text{softmax}\left(\frac{q_m^T k_n}{t\sqrt{|D|}}\right)=\text{softmax}\left(\frac{c\,q_m^T c\,k_n}{\sqrt{|D|}}\right),\quad c=\sqrt{1/t}$$

温度与原有的 $1/\sqrt{|D|}$ 合成一个分母 $t\sqrt{|D|}$：$q_m$ 与 $k_n$ 各乘 $c$，内积放大 $c^{2}=1/t$，因此接入点只有一处缩放常数[^src-yarn]（§3.3, Eq. 14）。

## 温度公式

对于缩放因子 $s$，推荐温度由拟合给出：在 LLaMA 7b/13b/33b/65b 上不微调、扫不同 $s$ 并取 perplexity 最低的 $\sqrt{1/t}$，再对该结果拟合[^src-yarn]：

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

温度缩放与 [[ntk-by-parts-interpolation|NTK-by-parts]] 插值互补[^src-yarn]（两者在 [[yarn#数据流与维度变换|YaRN 数据流]] 的 Stage 3 与 Stage 4 依次生效）：
- NTK-by-parts 处理位置编码的频率分布
- 温度缩放处理注意力的熵/集中度
- 两者结合 = [[yarn|YaRN]]

温度缩放与 [[scaling-factor-sqrt-dk|缩放因子 $1/\sqrt{d_k}$]] 在数学上同构——两者都是在 Softmax 前对 logits 施加线性缩放。$1/\sqrt{d_k}$ 将点积方差归一化至 1 以预防饱和[^src-bluuuuue-scaling-factor-intuition]；温度 $t$ 则根据上下文扩展比例 $s$ 动态调节注意力锐度[^src-yarn]。温度缩放可视为缩放因子在位置编码上下文扩展场景下的推广。同一位置还有第三种取值：[[qk-norm|QK-Norm]] 先把打分归一成 $[-1,1]$ 的余弦相似度，再乘一个可学习标量 $g$（初值 $\log_2(L^2-L)$），$g$ 相当于训练出来的温度，而非按扩展比例算出的温度[^src-henry-2020-qk-norm]。

## 引用

[^src-yarn]: [[source-yarn]]
[^src-bluuuuue-scaling-factor-intuition]: [[source-bluuuuue-scaling-factor-intuition]]
[^src-henry-2020-qk-norm]: [[source-henry-2020-qk-norm]]