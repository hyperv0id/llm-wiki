---
title: "Muon: An optimizer for hidden layers in neural networks"
type: source-summary
tags:
  - optimizer
  - neural-network-optimization
  - newton-schulz
created: 2026-04-30
last_updated: 2026-04-30
source_count: 2
confidence: high
status: active
---

# Source Summary: Muon Optimizer

## 核心论点

Muon 是一种针对神经网络隐藏层 2D 参数的新型优化器，通过在 SGD-动量更新后应用 Newton-Schulz 迭代进行正交化来处理更新矩阵[^src-muon-optimizer][^src-kellerjordan-muon-blog]。

## 主要贡献

1. **提出 Muon 优化器**：通过正交化更新矩阵来改进隐藏层参数优化
2. **Newton-Schulz 迭代**：使用高效的矩阵正交化方法，可在 bfloat16 下稳定运行
3. **实验验证**：在 NanoGPT 和 CIFAR-10 速度训练中取得当前最佳成绩
4. **竞争性任务框架**：提出用竞争性训练任务来解决优化器研究中的基线调优不足问题

## 实验结果

| 任务 | 指标 | 提升 |
|------|------|------|
| CIFAR-10 (94% 准确率) | A100-秒 | 3.3 → 2.6 (21% 提升) |
| NanoGPT 速度训练 | val loss 3.28 | 1.35x 提升 |
| 1.5B 模型训练 | 8xH100 小时 | 10h vs 13.3h (AdamW) |

## 设计细节

- 调优后的系数：a=3.4445, b=-4.7750, c=2.0315
- NS 迭代步数：5 步
- FLOP 开销：典型 LM 训练场景 < 1%

## 与先前工作的关系

- 与 Shampoo (Gupta et al. 2018) 相关，可视为无累积的即时版本
- 与 Orthogonal-SGDM (Tuddenham et al. 2022) 类似，但移动动量到正交化之前
- 继承自 Carlson et al. (2015) 的随机谱下降思想，但使用 NS 迭代替代 SVD

## 局限性

- 仅适用于 2D 参数和展平后的卷积参数
- 嵌入层和输出层建议使用 AdamW
- 尚未在大规模训练（20B+ 参数）中验证
- 分布式实现面临挑战

## 参考文献

[^src-muon-optimizer]: [[source-muon-optimizer]]
[^src-kellerjordan-muon-blog]: [[source-kellerjordan-muon-blog]]