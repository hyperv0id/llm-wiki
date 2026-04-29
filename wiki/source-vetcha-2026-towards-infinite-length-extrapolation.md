---
title: "Vetcha 2026: Towards Infinite Length Extrapolation - A Unified Approach"
type: source-summary
tags:
  - position-encoding
  - extrapolation
  - llm
  - transformer
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

本文提出 **Adaptive Positional Encoding (APE)** 方法，通过统一框架分析现有位置编码方法的局限性，并从理论上证明无限上下文外推的可行性。

## 核心贡献

1. **统一理论框架 (GPE)**：将位置编码对注意力分数的修改分解为乘法变换和加性偏置两部分，涵盖 RoPE、ALiBi 等主流方法[^src-vetcha-2026-towards-infinite-length-extrapolation]。

2. **无限上下文外推的数学条件**：提出四个关键性质——收敛归一化、熵有界性、远距离相关性保持 (LDCP)、梯度位置敏感性 (GPS)[^src-vetcha-2026-towards-infinite-length-extrapolation]。

3. **APE 方法**：结合自适应频率调制和精心设计的衰减偏置（线性 + 对数 + 平方根项），在理论上满足所有外推条件[^src-vetcha-2026-towards-infinite-length-extrapolation]。

4. **LongTinyStories 数据集**：新构建的合成数据集，包含 500-32,000 词的故事，用于评估长上下文处理能力[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 关键发现

- **RoPE** 违反收敛归一化和熵有界性——注意力分数不随距离衰减，导致归一化常数发散[^src-vetcha-2026-towards-infinite-length-extrapolation]
- **ALiBi** 无法满足 LDCP——远距离 token 的注意力分数被线性抑制到接近零[^src-vetcha-2026-towards-infinite-length-extrapolation]
- **APE** 在局部 LDCP 范围内优于 ALiBi，且通过次线性衰减（log + √|n|）保留更多长程依赖[^src-vetcha-2026-towards-infinite-length-extrapolation]

## 实验结果

在 TinyStories 和 LongTinyStories 数据集上，APE 在训练上下文窗口为 64 时，可外推到 16,384（256 倍）仍保持合理困惑度，优于 RoPE 和 ALiBi[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 局限性

- 仅关注位置编码，未涉及其他外推技术（如稀疏注意力、记忆模块）
- 实验在 30M 参数模型上进行，未验证大规模模型
- 内存占用略高于 ALiBi，推理速度略慢[^src-vetcha-2026-towards-infinite-length-extrapolation]

---

[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]