---
title: "Hyper-Connections (Zhu et al., ByteDance Seed, ICLR 2025)"
type: source-summary
tags:
  - residual-connections
  - macro-design
  - large-language-models
  - vision
created: 2026-10-27
last_updated: 2026-10-27
source_count: 1
confidence: medium
status: active
---

# Hyper-Connections (Zhu et al., ByteDance Seed, ICLR 2025)

> Zhu, D., Huang, H., Huang, Z., Zeng, Y., Mao, Y., Wu, B., Min, Q., Zhou, X. (2025). Hyper-Connections. *ICLR 2025*. arXiv:2409.19606v3. Seed-Foundation-Model Team, ByteDance.（PDF raw/hyper-connections-2409-19606.pdf）

## 核心论点

残差连接的 Pre-Norm 与 Post-Norm 变体在梯度消失与表示坍缩间构成跷跷板，根源在于层输入输出间的连接强度被预先固定[^src-hyper-connections]。Hyper-Connections 将隐向量复制为 $n$ 份（expansion rate），每层用 $(n+1)\times(n+1)$ 连接矩阵（$\mathbf{A_m}$ 读流、$\mathbf{B}$ 写流、$\mathbf{A_r}$ 流间混合）替代残差连接；矩阵可静态学习或由输入动态生成，Pre/Post-Norm 则是其 $n=1$ 的非训练特例[^src-hyper-connections]。

## 贡献

连接矩阵形式化分解为 depth-connections 与 width-connections，使残差连接成为 $n=1$ 特例[^src-hyper-connections]。动态分量管线为归一化 → 线性 → tanh → 可学习缩放（初始 0.01），初始化严格等价 Pre-Norm 残差[^src-hyper-connections]。OLMo-1B-DHC×4 参数 +0.033%、FLOPs +0.200%，重计算可压缩激活显存[^src-hyper-connections]。

## 证据

在 500B tokens 的 OLMo/OLMoE 复现设置中，1B DHC×4 V2 loss 2.811→2.781（Table 1），7B 下游 70.1→71.0（Table 5）；OLMoE-1B-7B ARC-C 41.8→47.8、训练损失 −0.027、收敛快 1.8×（Table 6、Fig. 1）；视觉任务 DiT 生成 FID 2.36→2.18、ViT-Large 分类 77.25→79.94（Tables 10–11）[^src-hyper-connections]。

## 局限

$n=1$ 劣于基线（Appendix F），ViT 多 epoch 训练增益递减（Fig. 11）。实验止于 500B tokens 与 ≤7B（激活 1.3B）规模，下游均值剔除波动 >20% 的数据集（Table 13）[^src-hyper-connections]。

[^src-hyper-connections]: [[source-seed-2024-hyper-connections]]
