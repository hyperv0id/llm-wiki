---
title: "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
type: source-summary
tags:
  - source
  - attention
  - gating
  - llm
  - attention-sink
  - long-context
created: 2026-10-26
last_updated: 2026-10-26
source_count: 1
confidence: medium
status: active
---

# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free

## 论文信息

- **标题**: Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
- **作者**: Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin（Qwen Team, Alibaba Group；University of Edinburgh / Stanford / MIT / Tsinghua）
- **arXiv**: 2505.06708v1（2025-05-10；PDF 编译日期 2025-06-11）
- **raw**: `raw/gated-attention-for-large-language-models-2505-06708.pdf`（正文 §1–§6 + Limitations + 附录 A.1–A.5，表 1–6、图 1–5）

## 问题

门控在 LSTM、Highway、SSM、线性注意力里已是默认组件，但少有工作受控地拆解它在 softmax attention 里的作用；Switch Heads、NSA、MoSA 把门控与专家路由、稀疏设计捆绑在一起，无法单独归因[^src-gated-attention-for-large-language-models-2505-06708]。论文用同一套训练配方扫 30 个变体，把门控的位置、粒度、head 独立性与激活函数逐项分离。

## 机制

门控写作 $Y'=Y\odot\sigma(XW_\theta)$，在 $G_1$–$G_5$ 五个位置、二档粒度（headwise / elementwise）、head 独立或共享、乘性或加性、sigmoid 或 SiLU 上组合。最优配置是在 SDPA 输出之后、输出投影 $W_O$ 之前，用于生成门控的 $X$ 取 pre-norm 之后的层输入，故门控随 query token 变化、按 head 独立[^src-gated-attention-for-large-language-models-2505-06708]。机制归到两点：一是 $W_V$ 与 $W_O$ 两个线性之间插入逐元素非线性，使二者不能再合并成单个 rank-$d_k$ 映射；二是 sigmoid 门控分数稀疏（均值 0.116），给 SDPA 输出带来输入相关的稀疏性[^src-gated-attention-for-large-language-models-2505-06708]。详见 [[sdpa-output-gating]]。

## 证据

15A2B MoE（128 专家 top-8，GQA，q=32、k=4、$d_k$=128）与 1.7B dense，最高 3.5T token。400B token 上 $G_1$ 逐元素门控把 Avg PPL 从 6.026 压到 5.761、MMLU 58.79→60.82，且优过同参数量级的扩张基线（k=8 加 50M 得 5.979，q=48 加 201M 得 5.953，+4 专家加 400M 得 5.964；表 1）[^src-gated-attention-for-large-language-models-2505-06708]。dense 侧改善训练稳定性：48 层 1.7B、lr $8\times10^{-3}$ 时 baseline 发散或退化到 9.195，带门控的 1T token 模型仍得 7.078（表 2）[^src-gated-attention-for-large-language-models-2505-06708]。attention sink 侧，首 token 注意力占比 0.467→0.048、层内最大激活 1053→94（表 4）；32k 训练长度内 RULER 79.50→79.77，YaRN 外推到 128k 后 31.65→58.82（表 5）[^src-gated-attention-for-large-language-models-2505-06708]。

## 边界

全部对照来自作者自有基线、同数据同 token 预算，未与其他架构改进做跨家族比较；30 个变体集中在 15B/1.7B 两档规模[^src-gated-attention-for-large-language-models-2505-06708]。长上下文增益主要在 YaRN 训练无关外推之后，训练长度内差距仅 0.27 分。sink 消除到长上下文泛化之间只有假设，无解析结论（Limitations）；附录 A.5 的截断实验（clip 到 300 或 100 仍在 lr $8\times10^{-3}$ 发散）表明大激活不是训练不稳定的唯一来源[^src-gated-attention-for-large-language-models-2505-06708]。

[^src-gated-attention-for-large-language-models-2505-06708]: [[source-gated-attention-for-large-language-models-2505-06708]]
