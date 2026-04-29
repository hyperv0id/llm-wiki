---
title: "YaRN"
type: entity
tags:
  - transformer
  - rope
  - context-extension
  - interpolation
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

# YaRN (Yet another RoPE extensioN method)

YaRN 是由 Nous Research 团队（Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole）提出的一种高效扩展 RoPE 位置编码模型上下文窗口的方法[^src-yarn]。

## 核心思想

YaRN 是三项技术的组合[^src-yarn]：

1. **NTK-by-parts 插值**：根据 RoPE 各维度的波长与上下文长度的关系，对不同维度采用不同插值策略
2. **注意力温度缩放**：在 softmax 前对注意力 logits 除以温度 $t$，控制长距离注意力的熵
3. **Dynamic Scaling**（可选）：推理时动态调整缩放因子

## 方法演进谱系

```
Position Interpolation (PI)
    │  等比例拉伸所有维度 → 丢失高频信息
    │
    ▼
NTK-aware 插值
    │  改变 base 参数 → 高频少缩放、低频多缩放
    │
    ▼
NTK-by-parts 插值
    │  按波长分段处理 → 精确控制各维度行为
    │
    ▼
YaRN = NTK-by-parts + 温度缩放
    │  控制注意力熵 → 最优长上下文性能
    │
    ▼
Dynamic-YaRN = YaRN + Dynamic Scaling
    推理时动态调整 → 零微调扩展 2x+
```

## 关键参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| $s$ | 缩放因子 $L'/L$ | 16 或 32 |
| $\alpha$ | 插值阈值下界 | 1 |
| $\beta$ | 插值阈值上界 | 32 |
| $t$ | 注意力温度 | $\sqrt{1/t} = 0.1\ln(s) + 1$ |

## 性能对比

### 长序列 perplexity（Proof-pile, LLaMA 7B, s=16）

| 方法 | 2k | 8k | 16k | 32k |
|------|-----|-----|------|------|
| PI (微调) | 5.70 | 4.64 | 3.97 | 3.57 |
| NTK-aware (微调) | 4.39 | 3.73 | 3.21 | 8.49 |
| NTK-by-parts (微调) | 4.14 | 3.62 | 3.12 | 2.81 |
| **YaRN (微调)** | **4.19** | **3.30** | **3.09** | **2.77** |

### 标准基准（Llama 2 7B, s=16）

| 基准 | 原始 Llama 2 | YaRN | 变化 |
|------|-------------|------|------|
| ARC-c | 53.1 | 52.3 | -0.8 |
| HellaSwag | 77.8 | 78.8 | +1.0 |
| MMLU | 43.8 | 42.5 | -1.3 |
| TruthfulQA | 39.0 | 38.2 | -0.8 |

## 与 ALiBi 的关系

YaRN 和 [[alibi|ALiBi]] 都解决 Transformer 的上下文外推问题，但路径不同[^src-yarn]：

| 方面 | ALiBi | YaRN |
|------|-------|------|
| 适用模型 | 从头训练 | 基于 RoPE 的预训练模型 |
| 位置编码 | 线性偏置（替代位置嵌入） | 修改 RoPE 频率 |
| 是否需要微调 | 不需要 | 推荐微调（少量数据） |
| 最大扩展 | ~3L | 32L（4k→128k） |

## 应用场景

- **长文档处理**：处理超过预训练长度的文档[^src-yarn]
- **Few-shot prompting**：在上下文中放入更多示例[^src-yarn]
- **长文本生成**：生成超长连贯文本[^src-yarn]
- **检索增强**：在更长上下文中检索信息[^src-yarn]

## 引用

[^src-yarn]: [[source-yarn]]