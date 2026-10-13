---
title: "Source: VICReg (Bardes et al., ICLR 2022)"
type: source-summary
tags:
  - self-supervised-learning
  - representation-learning
  - collapse
created: 2026-10-13
last_updated: 2026-10-13
source_count: 2
confidence: medium
status: active
---

# VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning

Bardes, Ponce, LeCun，ICLR 2022。文件：`raw/vicreg-variance-invariance-covariance-regularization.pdf`。

## 问题

联合嵌入自监督通过拉近同图双视图嵌入进行训练，核心困难是防止两分支输出常数或无信息向量的表示塌缩[^src-vicreg]。既有机制各有强约束：对比学习（SimCLR）依赖大 batch 负样本对；聚类方法（SwAV）需在线聚类与多原型；不对称方法（BYOL、SimSiam）依赖动量编码器、stop-gradient 或 predictor，且高度依赖 Batch Normalization；信息最大化方法（Barlow Twins、W-MSE）依赖批次标准化或高开销协方差白化，限制了分支权重、架构与输入模态的自由度[^src-vicreg]。

## 贡献

1. 三项损失解耦：不变项（MSE）+ 方差项（hinge 损失约束各维标准差 $\ge 1$ 防常数塌缩）+ 协方差项（非对角协方差平方和归零防维度冗余），默认权重 $\lambda=25, \mu=25, \nu=1$[^src-vicreg]。正则独立施加于各分支，不要求分支共享权重、同构架构或同模态输入，无需嵌入归一化、stop-gradient 或 predictor[^src-vicreg]。
2. ImageNet 线性评估（ResNet-50，1000 epochs）达 73.2% top-1，半监督 1%/10% 分别达 54.8%/69.5% top-1[^src-vicreg]；MS-COCO 图文检索（图到文 R@1 33.6）优于 Barlow Twins（31.4）与对比基线 VSE++（30.3）[^src-vicreg]。
3. 异构鲁棒性与机制解构：分支不同权重（DW 66.5% vs BT 64.2%）与不同架构（DA R50/ViT-S 66.2% vs BT 63.9%）下显著优于 Barlow Twins，而不对称方法（BYOL）在此配置结构上无法适用[^src-vicreg]；方差正则使 predictor 冗余，加入 BYOL/SimSiam 可稳定训练并缓解表征层缓慢塌缩[^src-vicreg]。

## 局限

协方差项依赖 expander 维度（8192 维饱和，256 维时 top-1 跌至 55.9%）[^src-vicreg]；COCO 目标检测微调（AP 39.4%）略逊于 BYOL（41.6%）[^src-vicreg]；防塌缩仅约束嵌入空间 batch 统计量，不约束表征的具体语义[^src-vicreg]——固定噪声环境下的平凡解见 [[jepa]]（Sobal et al.）[^src-jepa-focus-on-slow-features]。

## Related Pages

- [[vicreg]] — 技术细节页
- [[non-contrastive-ssl]] — 防塌缩路线谱系
- [[jepa]] — VICReg 作为 JEPA 训练判据的失效证据

[^src-vicreg]: [[source-vicreg-variance-invariance-covariance-regularization]]

[^src-jepa-focus-on-slow-features]: [[source-jepa-focus-on-slow-features]]
