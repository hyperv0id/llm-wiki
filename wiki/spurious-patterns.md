---
title: "Spurious Patterns in Attention"
type: technique
tags:
  - transformer
  - attention
  - training-dynamics
  - robustness
created: 2026-04-28
last_updated: 2026-04-28
source_count: 1
confidence: high
status: active
---

# Spurious Patterns in Attention

虚假模式（Spurious Patterns）是指数据中存在的非语义相关性，模型容易学习这些"捷径"而非真正的语义特征，导致注意力机制失效[^src-quest]。

## 问题定义

### 什么是虚假模式
- 数据中存在的非因果相关性
- 可以帮助解决部分样本，但非普遍真理
- 模型倾向于学习这些"捷径"而非真正特征[^src-quest]

### 例子
- 图像背景与标签的虚假相关性
- 位置信息与答案的虚假关联
- 特定纹理/颜色模式与类别的关联[^src-quest]

## 对注意力的影响

### 1. 注意力集中于虚假特征
- 模型学习关注携带虚假信号的 token
- 忽略真正语义相关的区域
- 注意力分布失去语义意义[^src-quest]

### 2. 陷入次优解
- 一旦学习到虚假模式，难以"解学习"
- 虚假模式通常是"局部最优"
- 需要更大的训练信号才能转向[^src-quest]

### 3. 泛化性能下降
- 训练集上表现好
- 测试集（无虚假模式）上表现差
- 虚假模式在测试分布中不存在[^src-quest]

## QUEST 论文中的玩具实验

### 实验设计
- 序列包含 N 个 token
- 随机位置 L 包含"答案"
- 50% 样本有偏差（biased）：答案 token 有额外偏差向量 b
- 50% 样本无偏差（unbiased）：答案 token 无偏差[^src-quest]

### 结果
| 方法 | 学习正确解的成功率 |
|------|-------------------|
| 标准注意力 | 25% |
| QNorm | 49% |
| **QUEST** | **58%** |

### 关键发现
- 标准注意力：biased 样本的键范数增长，模型依赖偏差向量
- QNorm：完全丢弃键范数信息，无法区分 biased/unbiased
- QUEST：保持键范数信息用于区分，但不 allow "窃取"注意力[^src-quest]

## 解决方案

### 1. 键归一化（QUEST）
- 消除键范数的"窃取"效应
- 允许模型使用键范数信息（区分不同 token）
- 但不允许高范数键主导全局注意力[^src-quest]

### 2. 数据增强
- 减少虚假模式的影响
- 增加正样本多样性
- 对抗训练[^src-quest]

### 3. 正则化
- 注意力熵正则化
- 鼓励更均匀的注意力分布
- 防止过度聚焦少数 token[^src-quest]

## 与鲁棒性的关系

### 虚假模式 → 脆弱性
- 模型依赖虚假特征
- 当虚假特征被破坏时性能下降
- 对抗攻击正是利用这一点[^src-quest]

### QUEST 的改进
- 更均匀的注意力分布
- 关注更多相关区域
- 对输入扰动更鲁棒
- 实验验证：对抗攻击、数据损坏下性能更好[^src-quest]

## 引用

[^src-quest]: [[source-quest]]

## 相关页面

- [[cbsa]] — CBSA 通过子空间压缩和可解释架构减少虚假模式学习的机会