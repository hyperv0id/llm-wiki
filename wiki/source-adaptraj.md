---
title: "AdapTraj: A Multi-Source Domain Generalization Framework for Multi-Agent Trajectory Prediction"
type: source-summary
tags:
  - trajectory-prediction
  - mixture-of-experts
  - domain-generalization
  - multi-source
  - spatio-temporal
created: 2026-09-17
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# AdapTraj: A Multi-Source Domain Generalization Framework for Multi-Agent Trajectory Prediction

**Authors**: Tangwen Qian、Yile Chen、Gao Cong、Yongjun Xu、Fei Wang（中国科学院计算技术研究所；南洋理工大学；中国科学院大学）。
**Venue**: arXiv:2312.14394v1 [cs.AI]（2023-12-22）；缓存未见会议标注。论文自称 plug-and-play 框架，可挂到不同 backbone 上。
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/VAR3CEQQ/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 问题

现有多智能体轨迹预测假定训练与部署分布一致，而部署环境存在分布漂移；单源域泛化到未见域性能下降，多源域共同训练又会遇到 negative transfer——论文用 Table II 与 Table III（Counter 从 ETH&UCY 单源增到三源时 ADE/FDE 由 1.48/3.03 升到 1.77/3.68，CausalMotion 由 1.56/3.28 升到 1.89/3.68，越低越好）量化了这两个问题[^src-adaptraj]。目标是在多源域泛化设定下（训练与评测都接触不到目标域）学到可迁移模型。

## 方法

框架含三部分：**domain-invariant extractor**（源域共享权重）、**domain-specific extractor**、**domain-specific aggregator**；本节只记录与 MoE 相关的部分[^src-adaptraj]。

- **MoE 的借用方式与关键判断**：Sec III.C 原文写 "drawing inspiration from the concept of mixture-of-experts [8], [10], [20], [23]"，引用为 Meta-DMoE（NeurIPS 2022）、Jacobs 等 1991 的 Adaptive mixtures of local experts、Spatial mixture-of-experts（NeurIPS 2022）、Adversarial mixture of experts（ICDE 2021）[^src-adaptraj]。Related Work 把 MoE 泛化路线分为推理时稀疏选择少数 expert（[37]–[39]）与**全量聚合**（full aggregation，[8], [20], [23]）两类，论文将自己归入后者[^src-adaptraj]。
- **expert 数量与类型**：expert 数 = 源域数 $|D^S|$。每个源域各有一套 extractor，逐域单独训练：$H^s_i=M^k_{ind}(h^{t,le}_{ei})$、$H^{Es}_i=M^k_{nei}(P_i)$（Eq 17、18），再由 $M_{fuse}$ 融合为 $H^s$（Eq 19）[^src-adaptraj]。
- **路由/聚合形式**：**没有门控网络、没有 top-k、没有 softmax gating**。路由信号是**域标签**——训练时以概率 $\sigma$（aggregator ratio）把域标签掩蔽 $D^S_k\to D^?_S$，模拟测试阶段的分布漂移；聚合则对所有源域 expert 输出求和后送入 aggregator：$H^s_i=A_{ind}(\sum_{k=1}^{|D^S|}M^k_{ind}(X^t_i))$（Eq 21），邻居侧同理（Eq 22）[^src-adaptraj]。因此这里 expert 指按源域划分的专用子模型，路由是显式的域归属而非可学习的稀疏选择。
- **训练形式**：teacher-student 式三步训练（Alg. 1）：第一步联合训练 backbone 与两个 extractor，$\mathcal{L}_{total}=\mathcal{L}_{base}+\delta\mathcal{L}_{ours}$，$\mathcal{L}_{ours}=\alpha\mathcal{L}_{recon}+\beta\mathcal{L}_{diff}+\gamma\mathcal{L}_{similar}$（Eq 23、24）；第二、三步训练 aggregator，aggregator 用较高学习率 $lr\times f_{high}$、其余模块用 $lr\times f_{low}$，损失改为 $\mathcal{L}_{base}+\delta'\mathcal{L}_{ours}$（Eq 25）[^src-adaptraj]。
- **额外损失**：$\mathcal{L}_{diff}$ 是 specific 特征与 invariant 特征之间的 soft subspace 正交约束（Eq 20），$\mathcal{L}_{similar}$ 为域对抗相似损失（Eq 15），$\mathcal{L}_{recon}$ 为重建损失（Eq 12）。论文**未提及负载均衡损失**[^src-adaptraj]。
- **超参**：$\alpha=0.01$、$\beta=0.075$、$\gamma=0.25$（全部实验固定）；300 epochs、batch size 32；观测 8 步（3.2 s）预测 12 步（4.8 s）[^src-adaptraj]。

## 实验

四个真实多智能体数据集：ETH&UCY、L-CAS、SYI、SDD（统计见 Table I），TrajNet++ 预处理（插值到 0.4 s 间隔）；backbone 用 PECNet 与 LBEBM 两种，baseline 为 vanilla、Counter、CausalMotion[^src-adaptraj]。

- **Table IV（多源域泛化，每数据集轮流作目标域）**：PECNet-AdapTraj 平均 ADE 0.665 / FDE 1.115，对照 vanilla 0.692 / 1.192、Counter 1.144 / 1.884；LBEBM-AdapTraj 0.589 / 1.124，对照 vanilla 0.694 / 1.392、Counter 1.238 / 2.586[^src-adaptraj]。
- **Table V（单源域泛化，各源域单训、在 SDD 评测）**：PECNet-AdapTraj 平均 1.334 / 2.074，LBEBM-AdapTraj 1.129 / 2.260；PECNet-vanilla 为 1.482 / 2.146、LBEBM-vanilla 1.209 / 2.397[^src-adaptraj]。
- **Table VI（源域数量，目标 SDD）**：PECNet-AdapTraj 在 SDD 单源、ETH&UCY、ETH&UCY+L-CAS 三种输入下 ADE 0.585 → 1.121 → 1.072、FDE 1.052 → 1.743 → 1.729；对照 PECNet 为 0.592 → 1.203 → 1.240 与 1.051 → 1.877 → 1.956[^src-adaptraj]。
- **Table VII（消融，目标 SDD、源域 ETH&UCY+L-CAS+SYI）**：PECNet 下 w/o specific 0.942 / 1.799、w/o invariant 0.927 / 1.671、完整 0.911 / 1.670；LBEBM 下 0.842 / 1.728、0.850 / 1.773、0.814 / 1.648[^src-adaptraj]。
- **Table VIII（推理时间，秒）**：PECNet-AdapTraj 0.007 vs PECNet-vanilla 0.003；LBEBM-AdapTraj 0.030 vs LBEBM-vanilla 0.027[^src-adaptraj]。
- **Fig. 4(a)–(f)**：$\delta$、$e_{start}$、$e_{end}$、$\sigma$、$f_{low}$、$f_{high}$ 的敏感度曲线，论文报告 $\delta$ 取中等值最好、$\sigma$ 大于 0.5 后收益转平或变差[^src-adaptraj]。

## 局限

缓存中**无独立 Limitations 章节**：Conclusion 之后即为致谢与参考文献，论文在结论只重述框架与实验结论[^src-adaptraj]。论文未提及 expert（源域）数量继续增加时的上界，也未给出负迁移的定量分解。

## 与其它 MoE 方法的差异

论文只在 Related Work 与 Sec III.C 概括 MoE 的两条路线（稀疏选择 vs 全量聚合），并把自己归入全量聚合一侧；未与任何具体 MoE 方法做实验对比，也未给出门控设计[^src-adaptraj]。按本文证据，AdapTraj 借用的是 MoE 的"多个专家子模型 + 集合并合"思想，用域标签代替门控做显式路由；论文未提及与标准 MoE 在路由形式、稀疏性与负载均衡方面的差异。

[^src-adaptraj]: [[source-adaptraj]]
