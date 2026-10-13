---
title: "LeJEPA: Latent-Euclidean JEPA"
type: technique
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

# LeJEPA: Latent-Euclidean JEPA

LeJEPA（Balestriero & LeCun，arXiv:2511.08544v3，2025-11 preprint）是 JEPA 预测损失与 [[sigreg|SIGReg]] 各向同性高斯正则项的凸组合，二者的权衡由单超参 $\lambda$ 完全确定。既有 JEPA 依赖的防塌缩机制（stop-gradient、非对称视图、teacher-student、EMA、显式归一化/白化）本质上是对表示塌缩症状的事后补偿；LeJEPA 则将嵌入空间的目标分布直接规定为各向同性高斯分布（见 [[isotropic-gaussian-embeddings]]），使表示塌缩在目标构造层面被排除，从而移除了 stop-gradient、teacher-student 网络与超参调度器[^src-lejepa]。

## 问题：既有判据的四点不足

既有自监督学习判据存在四项不足：(i) 欠规定——判据可在嵌入退化的配置下取得最小值；(ii) 面对 batch 大小与嵌入维度具有平方级的时间与内存复杂度；(iii) 对数据分布、超参和模型架构敏感；(iv) 缺乏理论解释与统计保证[^src-lejepa]。表示塌缩（全部输入映射至近乎同一点或低维子空间）即为欠规定的直接体现，[[jepa]] 页记录的固定噪声失效亦属此类退化[^src-lejepa]。

## 损失

预测损失沿用 DINO 的多视图设定：输入包含 $V_g$ 个全局视图与 $V_l$ 个局部视图，由全体视图共同预测全局视图。具体实现中，对每个样本计算全局视图的均值 $\mu_n$，损失定义为各视图嵌入到 $\mu_n$ 的均方距离。总损失对 batch 内每个样本与每个视图分别施加：

$$\mathcal{L}_{\mathrm{LeJEPA}}=\frac{\lambda}{V}\sum_{v=1}^{V}\mathrm{SIGReg}(\{z_{n,v}\}_{n=1}^{B})+\frac{1-\lambda}{B}\sum_{n=1}^{B}\mathcal{L}_{\mathrm{pred}}(\{z_{n,v}\}_{v=1}^{V})$$

其中 $\lambda$ 为唯一权衡超参，推荐默认值为 0.05；SIGReg 施加于各视图各自对应的 batch 嵌入，预测损失与正则项两路计算均不截断梯度[^src-lejepa]。核心实现约 50 行，仅依赖 all_reduce 通信算子即可扩展至多卡分布式训练[^src-lejepa]。

## 证据

以下结果除注明外均为论文报告：IN-1K/100 epochs、ViT-L/14、冻结骨干线性探针（top-1）为默认协议（Section 6.1，Table 1）。

**超参稳定性。** 在 IN-1K 上的 ViT-L/14 实验中：Epps-Pulley 积分区间、切片数 512–2048 与积分点数 5–41 扫描得到 71.82–75.02；局部/全局视图组合（$V_g$ 1–4、$V$ 4–10）得到 53.06–75.08；batch 128–1024 得到 72.20–74.72；嵌入/投影维与切片数组合得到 73.94–75.65。没有任何一组参数配置引发表示塌缩，个别配置略有增益（Table 1）[^src-lejepa]。在 IN-100/ResNet-50 上，$\lambda$ 从 0.001 扫描至 0.5、视图数取 2/4/8 时，准确率保持在 76.67–84.68，且各视图数对应的峰值 $\lambda$ 随视图数增加而右移（Figure 8、Table 7），因此以 $\lambda=0.05$ 为稳健默认[^src-lejepa]。

**训练损失作无标签模型选择。** 在（SIGReg 损失, 预测损失）构成的二维平面上，存在指向左下方的线性 trade-off front，训练总损失与下游准确率呈现高度单调关系；训练损失与下游准确率的 Spearman 相关系数约为 85%，若依据缩放律以 $\mathrm{train\_loss}/\lambda^\alpha$ 进行修正，在 $\alpha\approx0.4$ 时，六组「模型×数据」组合的相关系数提升至 93.8–99.2%（resnet18/flowers102 0.60→0.95、resnet50/inet10 0.81→0.99、ViT-base-8/inet1k 0.88→0.93 等），使得无标签交叉验证成为可能（Figures 10、11、19）[^src-lejepa]。论文对照指出，最新 JEPA 模型的训练损失与下游性能相关性较低[^src-lejepa]。

**域内预训练对比前沿迁移。** 在 Galaxy10（星系形态分类，11008 训练样本、非自然图像）上采用默认超参预训练 400 epochs：冻结骨干线性探针下，LeJEPA ConvNeXt-V2 Nano 达到 76.52、ResNet-34 达到 78.17，而 DINOv2 Small 为 67.62、DINOv3 ViT-S/16 为 71.38；全量微调下，LeJEPA 分别达到 82.72 与 83.28，DINOv2 为 78.34、DINOv3 为 81.60——两组协议下域内预训练结果均高于前沿模型迁移（Table 3）[^src-lejepa]。在 flowers102（1020 样本）上，小架构域内预训练准确率达到 74.57–82.19（Table 5）[^src-lejepa]。

**迁移学习对比 I-JEPA。** 在 8 个数据集上的少样本分类评测中（Table 2）：LeJEPA ViT-L（304M、IN-1K、100 epochs）的 10-shot 平均准确率为 60.95、全量为 79.48，而 I-JEPA ViT-H（632M、IN-1K、300 epochs）对应为 60.51 与 78.50；在细粒度数据集（DTD、flowers102、food101）上 LeJEPA 取得领先。表内数据亦包含例外情形：LeJEPA 的 1-shot 平均值 29.55 低于 I-JEPA 的 30.20，且 I-JEPA+STOP（80.70）与 I-JEPA IN-22K/900 epochs（80.82）的全量平均值高于 LeJEPA[^src-lejepa]。论文正文中「consistently outperform」的表述与表内 1-shot 结果存在出入，两处口径均按原表记录（备注见 ingest 报告）[^src-lejepa]。

**规模。** 在 IN-1K 预训练与在线线性探针协议下：ViT-L（0.3B）达到 77.1%、ConvNeXtV2-H（0.6B）达到 78.5%（Section 6.4）；论文摘要中报告的冻结骨干线性评估口径下，ViT-H/14 达到 79%[^src-lejepa]。在 1.8B 参数的 ViT-g 上，训练损失曲线保持平稳（Figure 1）；实验总计覆盖 10+ 数据集与 60+ 架构[^src-lejepa]。

**涌现结构。** ImageNet 预训练的 LeJEPA ViT-L 最后一层特征经 PCA 投影展现出清晰的前景与背景分离；自注意力图阈值化可在无标注视频中实现目标跟踪（Figures 13、14）[^src-lejepa]。

## 边界

- $O(1/N)$ 偏差与梯形数值积分是误差的近似来源：积分点数与切片数在 Table 1 的参数扫描中对准确率产生约 3 个百分点的影响（71.82–75.02），论文以超参扫描展现的稳定性作为该近似成立的经验依据，未提供理论误差校正项[^src-lejepa]。
- 迁移对比实验仅以 I-JEPA 系列模型为基线；与 DINOv2/v3 的对比仅局限于域内小数据集场景，表内未包含 IN-1K 规模下的对照基线[^src-lejepa]。
- 在自然图像小数据集（flowers102、cifar10/100、food101、inet10）上，I-JEPA ViT-H/22K 迁移表现依然高于小架构的域内预训练，LeJEPA 域内预训练的性能优势主要体现在非自然图像领域（Table 5）[^src-lejepa]。

## Related Pages

- [[sigreg]] — 防塌缩正则的统计构造
- [[isotropic-gaussian-embeddings]] — 目标分布的最优性定理
- [[jepa]] — JEPA 架构与本页所替换的启发式背景
- [[vicreg]] — 矩统计正则及其在本框架下的退化位置
- [[non-contrastive-ssl]] — 非对比路线谱系

[^src-lejepa]: [[source-lejepa]]
