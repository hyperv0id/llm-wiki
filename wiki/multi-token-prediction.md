---
title: "Multi-token Prediction"
type: technique
tags:
  - llm
  - pretraining
  - training-objective
  - sample-efficiency
  - code-generation
created: 2026-07-27
last_updated: 2026-07-27
source_count: 1
confidence: medium
status: active
---

# Multi-token Prediction

**Multi-token prediction（多 token 预测）** 是一种语言模型预训练目标：在每个位置同时预测后续 **n** 个 token，而不是只预测 next token。Gloeckle et al. (ICML 2024) 给出可扩展到 13B、训练时间与峰值显存几乎无额外开销的实现，并在代码生成与部分自然语言生成任务上证明其样本效率与推理加速收益[^src-gloeckle-2024-multi-token-prediction]。

## 动机

标准 next-token + teacher forcing 容易：

1. 贴合局部、易预测模式，忽视决定后续语义轨迹的 **choice points**；
2. 训练（真值前缀）与推理（自回归前缀）分布错配（exposure bias 的 LM 版本）[^src-gloeckle-2024-multi-token-prediction]。

多 token 预测把“看更远”写进主损失，作为辅助任务强化长期依赖与规划。

## 方法

**共享 trunk + n 个独立 head + 共享 unembedding**：

$$P_\theta(x_{t+i}\mid x_{t:1}) = \mathrm{softmax}\big(f_u(f_{h_i}(f_s(x_{t:1})))\big),\quad i=1,\ldots,n$$

- 训练：对 $i=1..n$ 的交叉熵求和（条件独立分解）。
- 参数公平：新增 head 层数从 trunk 层数中扣除，总参数与 next-token 基线相同。
- 显存：逐 head 串行反传，避免同时物化 $n$ 份 vocab logits 梯度，峰值显存 $O(V+d)$ 而非 $O(nV+d)$[^src-gloeckle-2024-multi-token-prediction]。
- 推理：默认只用 $i=1$ 头；其余头支撑 [[self-speculative-decoding|self-speculative decoding]]。

备选 head 结构（线性 / causal 堆叠 / anticausal）在附录中探索，**并行独立 transformer head** 最稳[^src-gloeckle-2024-multi-token-prediction]。

## 何时有效

| 设定 | 现象 |
|------|------|
| 模型规模 | 小模型常弱于 baseline；约 ≥3B 起代码任务持续反超，13B 增益最大 |
| 最优 n | 32k vocab 代码上 n=4 较稳；字节级 n=8 更一致；APPS 等任务可偏 n=6 |
| 多 epoch | 1T tokens 仍有增益，幅度缩小 |
| 微调 | 多 token **预训练** 表征更强；接 next-token 微调往往最好 |
| 任务类型 | 代码 pass@k、摘要 ROUGE、算法合成任务↑；多数 multiple-choice NL 基准≈或略↓ |
| 字节模型 | 显著缓解 next-byte 的局部过拟合，使 byte-level 预训练更可行 |

## 机制解释（推测）

1. **Choice-point 重加权**：关键转移及其后果在 n-step loss 中获得约 $n(n+1)/2$ 项，无关紧要转移约 n 项，相对权重约 $(n+1)/2$[^src-gloeckle-2024-multi-token-prediction]。
2. **互信息放大**：2-token 目标相对提高 $I(X;Y)$ 权重，迫使表示编码约束后续文本的决策。
3. **Induction / 算法推理**：小模型上显著促进 induction heads；多项式算术 OOD 精度提升可超过单纯放大参数量。

与 [[sparse-teacher-forcing|Sparse Teacher Forcing]] / DCRNN scheduled sampling 同属“缓解教师强制与自回归错配”家族，但 multi-token 不改输入为模型采样 token，因而更适合离散文本的并行 Transformer 训练[^src-gloeckle-2024-multi-token-prediction]。

## 局限

- 不保证选择题/似然基准提升；n 与数据/词表需调。
- 对已 next-token 收敛的模型（如 Llama-2）直接多 token 微调收益有限。
- 增益随数据量增大可能收窄（摘要、GSM8K 均有此趋势）。

## 相关

- [[source-gloeckle-2024-multi-token-prediction]] — 源论文摘要
- [[self-speculative-decoding]] — 多头加速推理
- [[sparse-teacher-forcing]] — DSR 中的间歇真值重校准
- [[moirai-moe]] — 时序基础模型中的 next-token（patch）预训练对照

## 引用

[^src-gloeckle-2024-multi-token-prediction]: [[source-gloeckle-2024-multi-token-prediction]]
