---
title: "Context Window Extension"
type: concept
tags:
  - transformer
  - context-length
  - extrapolation
created: 2026-04-28
last_updated: 2026-04-28
source_count: 3
confidence: high
status: active
---

# Context Window Extension

上下文窗口扩展是指将预训练语言模型的推理上下文长度扩展到超过其训练时长度的一系列技术[^src-yarn][^src-alibi]。

## 问题定义

- **预训练上下文长度 $L$**：模型训练时使用的最大序列长度[^src-yarn]
- **目标上下文长度 $L'$**：推理时需要支持的最大序列长度[^src-yarn]
- **缩放因子 $s = L'/L$**：扩展倍数[^src-yarn]

## 为什么需要扩展

1. **训练成本**：长序列训练需要大量 GPU 内存和计算（注意力 $O(L^2)$）[^src-yarn]
2. **实际需求**：推理时往往需要比训练时更长的上下文（长文档、多轮对话）[^src-yarn]
3. **灵活性**：同一模型可能需要处理不同长度的输入[^src-yarn]

## 主要方法分类

### 从头训练方法

修改模型架构，使模型天然支持外推：

| 方法 | 机制 | 最大扩展 |
|------|------|---------|
| [[alibi|ALiBi]] | 线性注意力偏置 | ~3L |
| [[adaptive-positional-encoding|APE]] | 自适应频率调制 + 次线性衰减偏置 | >256L |
| xPos (Sun et al., 2022) | 旋转位置编码 + 衰减 | ~2L |

### 插值方法（基于 RoPE）

修改已有 RoPE 模型的频率，通过少量微调恢复性能：

| 方法 | 机制 | 最大扩展 |
|------|------|---------|
| Position Interpolation (PI) | 等比例拉伸频率 | ~8x |
| [[ntk-aware-interpolation|NTK-aware]] | 改变 base 参数 | ~16x |
| [[ntk-by-parts-interpolation|NTK-by-parts]] | 按波长分段插值 | ~32x |
| [[yarn|YaRN]] | NTK-by-parts + 温度缩放 | ~32x |

### 推理时方法（零微调）

| 方法 | 机制 | 最大扩展 |
|------|------|---------|
| [[dynamic-scaling|Dynamic Scaling]] | 动态调整缩放因子 | ~2x |
| Dynamic-YaRN | Dynamic Scaling + YaRN | ~2x+ |

## 关键挑战

1. **高频信息丢失**：等比例拉伸会移除位置编码的高频分量[^src-yarn]
2. **绝对 vs 相对位置**：长波长维度编码绝对位置，短波长编码相对位置[^src-yarn]
3. **注意力熵失控**：长上下文中注意力分布过于平坦[^src-yarn]
4. **短序列性能退化**：固定缩放因子在短序列上过度插值[^src-yarn]
5. **LDCP vs 收敛归一化矛盾**：远距离相关性保持与归一化收敛不可兼得[^src-vetcha-2026-towards-infinite-length-extrapolation]

## 评估指标

- **Perplexity（滑动窗口）**：衡量长序列语言建模质量[^src-yarn]
- **Passkey 检索准确率**：衡量模型是否真正关注了长上下文中的所有位置[^src-yarn]
- **标准基准退化**：确保扩展后短上下文性能不显著下降[^src-yarn]

## 引用

[^src-yarn]: [[source-yarn]]
[^src-alibi]: [[source-alibi]]
[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]