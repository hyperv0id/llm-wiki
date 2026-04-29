---
title: "LongTinyStories Dataset"
type: entity
tags:
  - dataset
  - evaluation
  - extrapolation
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: medium
status: active
---

**LongTinyStories (LTS)** 是 Vetcha 2026 论文新构建的合成数据集，用于评估小型语言模型的长上下文处理能力[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 构建方法

与 Microsoft TinyStories 数据集生成方式相似，唯一区别在于提示词[^src-vetcha-2026-towards-infinite-length-extrapolation]：

- **TinyStories**: "Write a short story (3-5 paragraphs)"
- **LongTinyStories**: "Write a very long story (5-7 chapters)"

## 数据规模

- 故事长度范围：500 - 32,000 词
- TinyStories 验证集平均长度：约 180 词

## 可读性指标

| 数据集 | FRE | Gunning Fog | ARI |
|--------|-----|-------------|-----|
| LongTinyStories | 93.06 | 3.63 | 2.64 |
| TinyStories | 105.19 | 4.83 | 0.85 |

注：较低的 Flesch Reading Ease (FRE) 和较高的 Gunning Fog、ARI 表明 LongTinyStories 更复杂[^src-vetcha-2026-towards-infinite-length-extrapolation]。

## 用途

1. **短程依赖评估**：TinyStories 验证集
2. **中程依赖评估**：LTS 0-5k 词区间
3. **长程依赖评估**：LTS 5k-10k 词区间

## 意义

现有语言建模基准（如 WikiText、Project Gutenberg-19）复杂度远高于 TinyStories，需要超大模型才能评估。LongTinyStories 的引入使得小型模型（~30M 参数）也能有效测试长度外推能力[^src-vetcha-2026-towards-infinite-length-extrapolation]。

---

[^src-vetcha-2026-towards-infinite-length-extrapolation]: [[source-vetcha-2026-towards-infinite-length-extrapolation]]