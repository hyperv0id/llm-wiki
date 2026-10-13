---
title: "VICReg"
type: technique
tags:
  - self-supervised-learning
  - representation-learning
  - collapse
  - joint-embedding
created: 2026-10-13
last_updated: 2026-10-13
source_count: 3
confidence: high
status: active
---

# VICReg: Variance-Invariance-Covariance Regularization

ICLR 2022（Bardes, Ponce, LeCun）。联合嵌入自监督防塌缩损失：方差与协方差两项正则显式阻止表征塌缩，无需负样本、动量编码器、stop-gradient、分支间权重共享、归一化或 predictor[^src-vicreg]。

## 损失结构

默认采用孪生结构：encoder $f_\theta$（ResNet-50，输出 2048 维）接 expander $h_\phi$（两层 FC-BN-ReLU 加一层线性层，各层均为 8192 维），两视图嵌入记为 $Z$、$Z'$。损失直接作用于 expander 输出：

$$\ell(Z,Z') = \lambda\, s(Z,Z') + \mu\,[v(Z)+v(Z')] + \nu\,[c(Z)+c(Z')]$$

- **不变项** $s$：未归一化的均方欧氏距离（MSE），拉近同一图像两视图的嵌入。
- **方差项** $v$：逐维标准差 hinge 损失，$v(Z)=\frac{1}{d}\sum_j \max(0, \gamma - \mathrm{std}(z_j))$（采用标准差而非方差以避免均值附近梯度消失），维持 batch 内各维标准差不低于阈值 $\gamma$，防止常数塌缩（constant collapse）。
- **协方差项** $c$：将 batch 内协方差矩阵的非对角元素吸引至零以去相关，防止各维冗余导致的信息塌缩（informational collapse），机制借自 Barlow Twins。

超参数 $\lambda=\mu=25$、$\nu=1$、$\gamma=1$、$\epsilon=10^{-4}$。与 Barlow Twins 的关键区别：正则分别独立施加于各分支而非跨相关矩阵，无需标准化；两分支统计量解耦，支持不同权重、不同架构或不同模态输入[^src-vicreg]。

## 证据

线性探针，ResNet-50，ImageNet 预训练：

- 1000 epochs：线性评估 Top-1 73.2%（Top-5 91.1%），三次随机初始化极差小于 0.1%，与 Barlow Twins（73.2%）一致；半监督 1% 标签 Top-1 54.8%、10% 标签 Top-1 69.5%[^src-vicreg]。
- 迁移学习：分类任务（Places205 54.3%、VOC07 86.6%、iNat18 47.0%）与多数同期方法持平且优于 Barlow Twins；目标检测与分割（VOC07+12 AP50 82.4%、COCO det AP 39.4%、COCO seg AP 36.4%）略落后于自监督前三名[^src-vicreg]。
- 多模态（MS-COCO 图文检索，R@1）：图像编码器 ResNet-152、文本编码器 GRU，两分支赋予不同正则系数；image-to-text 33.6、text-to-image 45.2，优于 Barlow Twins（31.4 / 42.9）与对比损失 VSE++（30.3 / 41.3）[^src-vicreg]。

## 与 BYOL/SimSiam 的对照

消融实验（100 epochs 线性评估）：

- 方差正则（VR）与 predictor 功能冗余：含 VR 时引入 predictor 无显著变化；无 VR 则表征塌缩，须同时依赖 stop-gradient 与 predictor（expander 无 BN 时仅 35.1%）[^src-vicreg]。
- VR 引入 BYOL/SimSiam 带来 +0.9%（1000 epochs 缩小至 +0.2%），SimSiam 增益更小。缓慢塌缩假说：BYOL/SimSiam 嵌入层标准差虽恒为 $1/\sqrt{d}$（铺满单位球），但其架构技巧未能完美维持表征方差；引入 VR 提升了表征层标准差并加快收敛[^src-vicreg]。
- VICReg+PR 在 1000 epochs 下与 VICReg 完全一致（73.2%）[^src-vicreg]。

## 分支自由度

100 epochs 线性评估。SW=共享权重，DW=同架构不同权重，DA=不同架构：

| 方法 | SW R50 | DW R50 | DA R50/R101 | DA R50/ViT-S |
|---|---|---|---|---|
| BYOL | 69.3 | 7（塌缩） | 7（塌缩） | 7（塌缩） |
| SimCLR | 64.4 | 63.1 | 63.9 | 63.5 |
| Barlow Twins | 68.7 | 64.2 | 65.3 | 63.9 |
| VICReg | 68.6 | 66.5 | 68.1 | 66.2 |

非对称方法（BYOL）在分支去耦时直接塌缩。从 SW 到 DW，VICReg 下降 2.1%（Barlow Twins 下降 4.5%）；异构架构设置下，VICReg 在 R50/R101 与 R50/ViT-S 上分别高出 Barlow Twins 2.8% 与 2.3%[^src-vicreg]。

## 敏感性

- expander 维度：256 维时 55.9%，16384 维时 68.8%，8192 维（68.6%）后趋于饱和（继续翻倍仅 +0.2%）；协方差项继承 Barlow Twins 对 expander 维度的依赖[^src-vicreg]。
- batch size：相较默认 2048（68.6%），缩小至 128 下降 1.2%（67.3%），扩大至 4096 下降 0.8%（67.8%），对批大小不敏感且易于跨 GPU 并行[^src-vicreg]。

## 谱系位置

防塌缩机制分三类：负样本（SimCLR）、分支不对称（BYOL、SimSiam）、信息量最大化（Barlow Twins、W-MSE）。VICReg 属第三类：显式正则、单分支独立作用、无架构假设，详见 [[non-contrastive-ssl]]。LeCun 2022 立场论文将其列为 JEPA 非对比判据的具体实现[^src-autonomous-machine-intelligence]。

方差与协方差正则均以 batch 统计量为对象，VICReg 仅防止嵌入空间塌缩，不约束表征聚焦的具体特征：Sobal et al. 在 toy 环境证明，固定噪声下三项目标可同时被平凡解满足，见 [[jepa]]。

## 时序预测中的应用

WorldTS 把方差/协方差正则移入时序潜空间预测：对预测状态与编码真值未来观测所得状态两组序列施加正则，防止 state matching 中的表征塌缩；统计量不再取自图像 batch 维，而改为跨样本与预测步计算。逐超参扫描（Colbun/Climate/MMSP，MAE）中 $\lambda_{VIC}=0.1$ 最优或次优，升至 0.5 时 MMSP 误差明显上升[^src-worldts]。

## Related Pages

- [[non-contrastive-ssl]] — 防塌缩路线谱系与 VICReg 的理论定位
- [[jepa]] — VICReg 作为 JEPA 训练判据及固定噪声下的失效
- [[contrastive-learning]] — 负样本路线及其代价
- [[energy-based-model]] — 塌缩问题的能量视角
- [[worldts]] — 把 VICReg 正则用于时序潜空间 state matching 的下游工作

[^src-vicreg]: [[source-vicreg-variance-invariance-covariance-regularization]]
[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
[^src-worldts]: [[source-worldts]]
