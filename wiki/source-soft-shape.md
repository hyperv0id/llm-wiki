---
title: "Learning Soft Sparse Shapes for Efficient Time-Series Classification (Liu et al., ICML 2025)"
type: source-summary
tags:
  - time-series-classification
  - mixture-of-experts
  - shapelets
  - soft-sparsification
  - load-balancing
  - icml-2025
created: 2026-09-17
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# Learning Soft Sparse Shapes for Efficient Time-Series Classification

**Authors**: Zhen Liu, Yicheng Luo, Boyuan Li, Emadeldeen Eldele, Min Wu, Qianli Ma（华南理工大学 + 新加坡 A*STAR I2R）。[^src-soft-shape]
**Venue**: ICML 2025（Proceedings of the 42nd ICML, Vancouver, PMLR 267），arXiv:2505.06892v2。[^src-soft-shape]
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/Q5RTMRVY/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 问题

现有 shapelet 方法对候选子序列做 hard 稀疏化：只保留判别性 shapelet、直接丢弃其余子序列，既损失可能有用的信息，也忽略不同 shapelet 对分类贡献的差异。同时 patch 化方法把全部 patch 送入模型，长序列下计算低效。[^src-soft-shape]

## 方法

SoftShape 分三步，MoE 插在 shape 稀疏化之后、分类器之前，共堆叠 L=2 个 soft shape learning block：[^src-soft-shape]

1. **软稀疏化（§4.2）**：1D CNN（Eq 2）把长度 m=γT 的滑窗子序列（步长 q=4）嵌入为 shape embedding，加可学习位置嵌入；gated attention head（Eq 3）给出贡献分 α∈(0,1)。得分位于前 η 比例的 shape 按 α 缩放保留（Eq 4），其余按 α 加权融合成单个 shape（Eq 5）。η 默认 50%，且先 warm-up 训练 150 epoch 再启动稀疏化。[^src-soft-shape]
2. **Intra-shape MoE（§4.3.1，Eq 6-9）**：路由 G(Sem)=TOPk(softmax(Wt·Sem))，Wt∈R^{Ĉ×d}；TOPk 在 softmax 之后施加以防门控值几乎处处为零（引 Riquelme et al. 2021）。每个 expert 是轻量 MLP（Eq 8：GeLU 激活），输出按 top-k 门控归一化加权（Eq 9），并按 Zoph et al. 2022 加残差连接。**k 默认 1；expert 总数 Ĉ = 数据集类别数 C**（Table 18 验证 Ĉ=C 最优：avg acc 0.9453，优于 C/2 的 0.9215 与 2C 的 0.9408）。expert 参数跨不同深度的 block 共享。[^src-soft-shape]
3. **Inter-shape 共享 expert（§4.3.2）**：把稀疏化后的 shape 序列 reshape 为 (B,d,Num)，用三核 Inception 1D CNN 作为**单个共享 expert**学习 shape 间时序模式；Num=J×η+1 远小于 T，降低计算量。Table 20：Inception（179.5K 参数，acc 0.9453）优于 Transformer（422.5K，0.8239）与 MLP（157.8K，0.8103）。[^src-soft-shape]

**负载均衡（三篇中唯一）**：沿用 Shazeer et al. 2017，L_imp 惩罚各 expert 门控和的变异系数（Eq 10），L_load 惩罚各 expert 分到的 shape 数量的变异系数（Eq 11）；总损失 L_total = L_ce + λ(L_imp + L_load)（Eq 15），λ 默认 0.001。Table 21 显示 18 数据集上 λ=0.01 的 avg acc（0.9471）实际略高于默认 0.001（0.9453），论文未解释该差异。[^src-soft-shape]

## 实验

- **Table 1（128 UCR）**：SoftShape avg acc 0.9334 / avg rank 2.72 / win 53，优于 TSLANet（0.9205 / 3.68 / 31）与 InceptionTime（0.9181 / 4.05 / 29），Wilcoxon p=1.06E-03（对 TSLANet）。[^src-soft-shape]
- **Table 2 消融（128 UCR）**：w/o Inter 0.9022、w/o Soft Sparse 0.9123、w/o Intra 0.9245、w/o Intra & Inter 0.8696、Linear Shape 换 CNN 0.9164，均显著低于全模型 0.9334。[^src-soft-shape]
- **Table 3（18 UCR）**：稀疏比例 (1−η)≤50% 时精度无显著下降（p>0.05，如 10% 时 0.9469 对不稀疏 0.9461），70%/90% 显著下降（p=9.37E-03 / 4.02E-04）。**Table 4**：k=1 的 avg rank（1.89）最优，k=3 最差（2.78）。[^src-soft-shape]
- **效率（Figure 3）**：ChlorineConcentration 拼接至长 16,600、HouseTwenty 拼接至长 20,000 的训练时间对比中，SoftShape 快于全部深度基线（TSLANet、InceptionTime、Medformer），仅慢于 CPU 上的 MR-H。[^src-soft-shape]

## 局限

论文无独立 Limitations 章节。自述的约束：对全部 128 UCR 数据集做评估耗时过大，分析实验只能选取 18 个子集；运行时间仅略优于 Medformer-50（50 epoch 截断计数）且不及非深度方法 MR-H；注意力分数在训练初期不准确，需 150 epoch warm-up 才能启动稀疏化。[^src-soft-shape]

## 与其它 MoE 方法的差异

论文自述（§2.3）：Wen et al. 2025（即 InterpGN）在 MoE 中用 gated router 把 DNN 与 shapelet transform 特征结合以补足 shapelet transform 单独使用的性能上限；SoftShape 不同，用 MoE router 为每个 shape embedding 激活类别专属 expert 学习 intra-shape 模式，并以共享 expert 学习 inter-shape 模式。动机引自 Chen et al. 2022 / Chowdhury et al. 2023：MoE router 可把同类模式路由到同一 expert 并滤除类别无关特征。论文未与其它时序 MoE 预测方法（如 MoLE、MoU）做实验对比。[^src-soft-shape]

[^src-soft-shape]: [[source-soft-shape]]
