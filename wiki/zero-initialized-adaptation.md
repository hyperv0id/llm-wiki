---
title: "零初始化适配"
type: concept
tags:
  - zero-initialization
  - model-adaptation
  - tsfm-adaptation
created: 2026-07-04
last_updated: 2026-07-28
source_count: 5
confidence: high
status: active
---

# 零初始化适配

**零初始化适配**（Zero-Initialized Adaptation）是一种模型适配设计原则：所有为下游任务新增的参数在训练开始时初始化为零，确保适配后的模型在起点处与预训练模型**完全等价**，随后渐进式地融合新信息。这一原则的核心价值在于消除适配的"冷启动劣势"——避免适配后模型的反而不如零样本推理。

## 跨领域实例

零初始化适配在三个不同领域得到验证，形成了连贯的设计谱系：

### 1. LoRA：低秩适配中的零初始化（2021）

LoRA（Low-Rank Adaptation）将预训练权重的更新分解为低秩矩阵乘积 $\Delta W = BA$，其中 A 用高斯初始化，但 **B 矩阵被零初始化**，确保 $\Delta W = 0$ 在训练起点[^src-cora]。这意味着适配始于原始预训练模型，低秩更新渐进注入。

### 2. DiT adaLN-Zero：扩散 Transformer 的恒等初始化（2023）

[[dit|DiT]] 将 adaLN 模块中控制残差路径的缩放参数 $\alpha_1, \alpha_2$ 零初始化（其他参数标准初始化），使每个 Transformer block 在训练起点等价于恒等映射[^src-dit]。实验证明这不仅是稳定训练的技巧——adaLN-Zero 在所有条件注入方案中取得了最佳 FID（19.47 vs cross-attention 26.14），且 12 个模型变体均无 loss spike[^src-dit]。

### 3. CoRA：TSFM 协变量适配的全部零初始化（2025/ICLR 2026）

[[cora-tsfm|CoRA]] 将零初始化推向极致：**所有**新增参数——包括跨模态投影矩阵 $W_{mi}$、偏置 $b_{mi}$、adaLN MLP——全部零初始化[^src-cora]。这意味着适配起点与预训练 TSFM 的零样本推理完全一致。消融实验：替换零初始化为 Xavier 初始化导致性能退化 4.3% MSE[^src-cora]。

## 为什么零初始化有效

三个相互关联的机制：

1. **灾难性遗忘预防**：零初始化确保预训练知识在学习新信息之前不被覆盖。非零初始化从第一步就扰动了预训练表示空间，可能导致不可逆的知识丢失。

2. **渐进式融合**：参数从零开始增长，协变量信息被逐步"掺入"预测过程。消融实验佐证：移除 adaLN（直接将条件加到输入）导致 12.9% MSE 退化——说明粗暴的信息注入比渐进融合差得多[^src-cora]。

3. **训练稳定性**：零初始化避免了 Transformer 训练早期常见的 loss spike。DiT 的 12 个变体无需 lr warmup、dropout 或 weight decay 即实现稳定训练[^src-dit]。

## 缺乏零初始化的后果

截至 2026 年，多个 TSFM 协变量适配方法仍未采用零初始化：

| 方法 | 零初始化 | 已知后果 |
|------|---------|---------|
| [[cora-tsfm|CoRA]] | ✅ 全部新增参数 | 最佳性能 |
| [[unica|UniCA]] | ❌ | 前置注入扰乱预训练嵌入空间；CoRA 在相同 backbone 下超出 23.5% |
| [[chronosx|ChronosX]] | ❌ | IIB 前置改嵌入 + 无零 init；CoRA 在 Sundial 协议超出 97.1%（非唯一原因）[^src-chronosx] |
| AdaPTS | ❌ | CoRA 超出 23.5% |

需要指出：ChronosX 和 AdaPTS 的差距不能全部归因于零初始化——它们还在编码器前注入协变量（双重劣势）。UniCA 作为最接近的对比，其前置注入 + 无零初始化的组合劣势最为清晰[^src-cora][^src-unica]。

## 开放问题

零初始化的有效性在经验层面已被多次验证，但缺乏严格的理论分析。为什么"零 → 增长"优于"小随机值 → 调整"？可能的解释包括：零初始化保证了参数更新的梯度方向纯粹由任务损失驱动，而非初始随机扰动的遗留效应；零初始化也使优化器在参数空间的起点位于一个对称性更高的区域。

## 相关页面

- [[cora-tsfm|CoRA]] — 零初始化 + 因果嵌入的 TSFM 协变量适配框架
- [[dit|DiT]] — adaLN-Zero 的原始出处
- [[tsfm-covariate-adaptation-comparison]] — 六种 TSFM 适配方法的系统对比
- [[unica|UniCA]] — 缺乏零初始化的对比方法
- [[chronosx|ChronosX]] — 早期 IIB+OIB 适配，无零初始化

[^src-cora]: [[source-cora]]
[^src-dit]: [[source-dit]]
[^src-unica]: [[source-unica]]
[^src-chronosx]: [[source-chronosx]]
[^src-sundial]: [[source-sundial]]