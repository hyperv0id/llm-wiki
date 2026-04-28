---
title: "Dynamic Scaling"
type: technique
tags:
  - rope
  - inference
  - context-extension
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Dynamic Scaling

Dynamic Scaling 是一种推理时技术，在每次前向传播中根据当前序列长度动态调整 RoPE 插值的缩放因子 $s$[^src-yarn]。

## 动机

固定缩放因子的问题[^src-yarn]：

1. **短序列性能下降**：当 $l' < L$ 时，固定 $s = L'/L$ 导致过度插值
2. **边界突变**：序列长度超过 $L'$ 时性能突然崩溃

Dynamic Scaling 通过动态调整 $s$ 解决这两个问题[^src-yarn]。

## 定义

在每次前向传播中[^src-yarn]：

$$s = \max\left(1, \frac{l'}{L}\right)$$

其中：
- $l'$：当前序列的实际长度
- $L$：预训练时的上下文长度

## 两种使用模式

| 模式 | $s$ 的取值 | 适用场景 |
|------|-----------|---------|
| 固定缩放 | $s = L'/L$（常数） | 已知最大长度，微调后使用 |
| **Dynamic Scaling** | $s = \max(1, l'/L)$ | 自回归生成，零微调扩展 |

## Dynamic-YaRN

Dynamic Scaling 与 YaRN 结合 = Dynamic-YaRN[^src-yarn]：

- 零微调即可扩展 2x 以上上下文
- 在 Llama 2 上，Dynamic-YaRN 的 long-range perplexity 优于 Dynamic-PI
- 有效防止超出预训练上下文窗口后的 perplexity 爆炸

## KV-Cache 兼容性

使用 KV-cache 时需注意[^src-yarn]：

- **正确做法**：在应用 RoPE 之前缓存 kv-embeddings
- **原因**：$s$ 变化时，每个 token 的 RoPE 都会改变
- **错误做法**：缓存已应用 RoPE 的向量 → Dynamic Scaling 失效

## 与其他方法的关系

- **Dynamic NTK**：Dynamic Scaling + NTK-aware 插值[^src-yarn]
- **Dynamic-YaRN**：Dynamic Scaling + YaRN[^src-yarn]
- 已被 Qwen 7B 等开源模型采用[^src-yarn]

## 引用

[^src-yarn]: [[source-yarn]]