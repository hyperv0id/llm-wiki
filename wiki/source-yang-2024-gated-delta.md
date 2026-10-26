---
title: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
type: source-summary
tags:
  - linear-rnn
  - delta-rule
  - mamba
  - gating
  - iclr-2025
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Gated Delta Networks: Improving Mamba2 with Delta Rule

Songlin Yang（MIT CSAIL，主要贡献于 NVIDIA 实习期间完成）、Jan Kautz、Ali Hatamizadeh（NVIDIA），ICLR 2025，arXiv:2412.06464。通读版本为 v3（2025-03-06，页眉 "Published as a conference paper at ICLR 2025"，22 页含附录 A–B），经比对 v1/v2 确认实验范围一致；PDF `raw/gated-delta-networks-2412-06464.pdf`。机制页见 [[gated-delta-rule]]。

## 核心论点

线性 Transformer 的外积关联记忆可存储的正交 key-value 对受维度上界约束，超限后的记忆碰撞使精确检索失效。既有两条修补路线各缺一半：Mamba2 的门控更新 $S_t=\alpha_tS_{t-1}+v_tk_t^\top$ 按同一比例衰减全部关联，能整体遗忘但无法定点修改；DeltaNet 的 delta rule 能软替换单个 key 的关联值，却无法快速清空。论文提出 gated delta rule $S_t=S_{t-1}(\alpha_t(I-\beta_tk_tk_t^\top))+\beta_tv_tk_t^\top$（Eq. 10）统一两者，并将 Yang et al. (2024b) 基于 WY 表示的 DeltaNet 分块并行算法推广至含门控项，实现张量核友好的 chunkwise 训练（§3.3）[^src-yang-2024-gated-delta]。

## 证据

1.3B/100B token（FineWeb-Edu）统一训练下：语言建模 Wiki ppl 16.42、LMB ppl 12.17，八项常识推理平均 55.32，均优于 Mamba2（16.56/12.56/54.89）与 DeltaNet（52.14）（Table 3）；S-NIAH 三任务 11 个长度点多数最优，如 S-NIAH-1 8K 处 91.8 对 Mamba2 的 30.4、S-NIAH-2 2K 处 99.8 对 DeltaNet 的 45.6（Table 2）；真实检索六任务平均 30.6 高于 Mamba2（29.8），混合 H1/H2 达 39.0/40.1 高于 Transformer++（37.0）（Table 4）；LongBench 平均 16.6 为纯 RNN 最高（Table 5）；六项长度外推基准在 RNN 中整体困惑度最低（Fig 2）；吞吐与 DeltaNet 持平（Fig 3）[^src-yang-2024-gated-delta]。

## 边界

长度外推结果混合，仅相对更稳；真实检索优势小于合成任务，归因于未对齐指令小模型的重复错误；吞吐比 Mamba2 慢 2–3K tokens/sec；正文称 Table 3 覆盖 400M 与 1.3B，表中只有 1.3B 行。全部实验在 0.4B–1.3B 规模、100B token 内，无更大规模结果[^src-yang-2024-gated-delta]。

[^src-yang-2024-gated-delta]: [[source-yang-2024-gated-delta]]
