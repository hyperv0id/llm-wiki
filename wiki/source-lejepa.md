---
title: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
type: source-summary
tags:
  - self-supervised-learning
  - representation-learning
  - joint-embedding
  - statistics
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

**来源**：Randall Balestriero（Brown University、Meta-FAIR）与 Yann LeCun（NYU、Meta-FAIR），二人同等贡献，arXiv:2511.08544v3 [cs.LG]，2025-11-14，preprint。PDF 存于 raw/。

**核心论点**：JEPA 现有的防塌缩手段（stop-gradient、非对称视图、teacher-student、EMA、显式归一化）均为启发式设计，其根源在于缺乏关于「嵌入应当服从何种分布」的理论指导。论文证明：在协方差受标量约束的分布族中，各向同性高斯是唯一使 worst-case 下游预测风险最小的嵌入分布（线性探针由偏差与方差两条引理保证，k-NN 与 kernel 探针由积分平方偏差定理保证）；据此提出可微正则项 SIGReg——将嵌入投影到随机 1D 方向上，采用 Epps-Pulley 特征函数检验度量其与标准高斯的偏离，时间与内存复杂度关于样本量和维度均为线性。

**贡献**：(1) 证明各向同性高斯最优性定理；(2) 提出 SIGReg 分布匹配目标（包含方向检验充分性定理、Epps-Pulley 有界损失/梯度/曲率定理及梯形积分实现）；(3) 提出 LeJEPA 训练目标——由预测损失与 SIGReg 凸组合构成，仅含单一权衡超参 $\lambda$（推荐 0.05），无 stop-gradient、无 teacher-student、无超参调度器，核心实现约 50 行代码；(4) 实验覆盖 10+ 数据集与 60+ 架构：超参扫描未发生塌缩；训练损失与下游准确率的 Spearman 相关达 85%，经 $\lambda$ 缩放律修正后达 94–99%，支持无标签模型选择；在 Galaxy10 上，小架构域内预训练表现超过 DINOv2/v3 迁移；在 IN-1K 上，ViT-L 在线线性探针达 77.1%、ConvNeXtV2-H 达 78.5%，1.8B ViT-g 训练损失平稳。

**局限**：SIGReg 损失与梯度带有显式 $O(1/N)$ 偏差（小 batch 实测无碍）；少样本迁移对比（Table 2）中，LeJEPA ViT-L 在 1-shot 平均表现低于 I-JEPA ViT-H，且低于 I-JEPA+STOP 与 IN-22K 900 epoch 变体；在自然图像小数据集上，域内小架构预训练仍低于 I-JEPA ViT-H/22K 迁移（Table 5，Galaxy10 除外）。

## Related Pages

- [[lejepa]] — 训练目标与全部实验
- [[sigreg]] — SIGReg 正则的统计检验构造
- [[isotropic-gaussian-embeddings]] — 各向同性高斯最优性定理

[^src-lejepa]: [[source-lejepa]]
