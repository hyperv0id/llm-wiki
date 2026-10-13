---
title: "JEPA: Joint Embedding Predictive Architecture"
type: technique
tags:
  - self-supervised-learning
  - world-model
  - energy-based-model
  - representation-learning
  - joint-embedding
created: 2026-10-13
last_updated: 2026-10-13
source_count: 5
confidence: medium
status: active
---

# JEPA: Joint Embedding Predictive Architecture

LeCun 2022 年《A Path Towards Autonomous Machine Intelligence》提出的非生成式预测世界模型架构，旨在学习多层级表征[^src-autonomous-machine-intelligence]。其核心设计为不直接预测 $y$ 自身，仅在表征空间预测 $y$ 的表征。

## 问题

自监督学习的目标是捕捉输入各部分间的相互依赖，即评估观测部分 $x$ 与未观测部分 $y$ 的兼容性。架构不要求从 $x$ 生成 $y$：与给定 $x$ 兼容的 $y$ 往往存在无穷多个，显式表示该集合极为困难甚至不可解，而判断候选 $y$ 是否兼容的计算代价低得多[^src-autonomous-machine-intelligence]。

像素级重建目标在设计上不区分场景重要物体与背景细节，易导致 object-vanishing。Sobal 等人据此以 JEPA 作为重建替代方案展开实证检验[^src-jepa-focus-on-slow-features]。

## 机制

两个编码分支将 $x$、$y$ 分别编码为 $s_x$、$s_y$。两个编码器无需共享参数，允许 $x$ 与 $y$ 采用不同架构或不同模态（如视频与音频）。predictor 基于 $s_x$ 预测 $s_y$，可引入隐变量 $z$。能量定义为表征空间的预测误差[^src-autonomous-machine-intelligence]：

$$E_w(x,y,z) = D(s_y,\ \mathrm{Pred}(s_x,z)) \tag{10}$$

$$\check{z} = \arg\min_{z \in Z} E_w(x,y,z) = \arg\min_{z \in Z} D(s_y,\ \mathrm{Pred}(s_x,z)) \tag{11}$$

$$F_w(x,y) = \min_{z \in Z} E_w(x,y,z) = D(s_y,\ \mathrm{Pred}(s_x,\check{z})) \tag{12}$$

隐变量推断通过能量最小化求得最优 $\check{z}$；消去 $z$ 后的自由能 $F_w(x,y)$ 仅依赖 $x$ 与 $y$（数学上对应零温自由能，统称能量）[^src-autonomous-machine-intelligence]。

JEPA 通过两种机制建模一对多（多模态）依赖关系，避免直接在像素空间生成 $y$[^src-autonomous-machine-intelligence]：

- **编码器不变性**：多个不同 $y$ 映射到同一 $s_y$ 时能量相同，消除不可预测的细节变异。
- **隐变量 z**：$z$ 在集合 $Z$ 内变化时，$\mathrm{Pred}(s_x,Z)$ 给出多个合理解。

JEPA 结合了 Joint Embedding Architecture 与 Latent-Variable Generative Architecture，区别在于预测目标为表征 $s_y$ 而非原始输入 $y$[^src-autonomous-machine-intelligence]。

## 训练

JEPA 可使用对比方法训练，但在高维空间样本效率低（见 [[non-contrastive-ssl]]）。非对比（正则化）路线包含四项信息准则（Figure 13）[^src-autonomous-machine-intelligence]：

1. 最大化 $s_x$ 关于 $x$ 的信息量
2. 最大化 $s_y$ 关于 $y$ 的信息量
3. 使 $s_y$ 易于被 $s_x$ 预测
4. 最小化隐变量 $z$ 的信息量

准则 1、2、4 共同防止能量面塌缩。准则 4 针对特定平凡解：若 $z$ 与 $s_y$ 同维且 predictor 忽略 $s_x$ 直接复制 $z$，则 $\check{z}=s_y$ 导致预测能量恒为 0，能量面完全平坦。约束 $z$ 的方式包括离散化、低维化、稀疏化或注入噪声，具体实现分为四类：离散化/量化（VQ-VAE）、维度/秩最小化（IRMAE）、稀疏化（线性稀疏建模、LISTA、非线性稀疏建模）与模糊化（noisy AE、VAE）[^src-autonomous-machine-intelligence]。

此外可引入任务相关的归纳偏置：在 $\tilde{s}_y$ 上附加预测头，预测易于从数据导出且与任务相关的变量[^src-autonomous-machine-intelligence]。

正则化路线仍在演进：VISReg 将 VICReg 式正则解耦为 scale（方差项）与 shape（Sliced Wasserstein sketching）两个独立目标，依据 Cramér-Wold 定理对齐所有 1D 投影的边缘分布而非仅做二阶去相关，以塌缩时梯度有界为设计准则，并在 ImageNet 预训练的 ViT 上报告了超过 ImageNet-1K 预训练对比组的 OOD 精度（见 [[visreg]]）[^src-visreg]。

LeJEPA（Balestriero & LeCun，arXiv:2511.08544，2025 preprint）为该路线补充了分布层面的理论：首先证明在协方差受标量约束的分布族内，各向同性高斯是唯一最小化 worst-case 下游预测风险的嵌入分布，进而提出 SIGReg 分布匹配正则实现该目标；其预测损失与正则项的凸组合仅依赖单超参 $\lambda$，移除了 stop-gradient、teacher-student 与超参调度器。训练损失与下游准确率的 Spearman 相关约 85%，经 $\lambda$ 缩放律修正后达 94–99%，训练损失因此可用于无标签模型选择（见 [[lejepa]]、[[sigreg]]、[[isotropic-gaussian-embeddings]]）[^src-lejepa]。

## 证据：固定干扰噪声下的失效

Sobal 等人针对 JEPA 在固定干扰噪声下的表征行为展开实证检验[^src-jepa-focus-on-slow-features]：

**设置**：全离线、无奖励的预录观测-动作序列。环境为边长 1 正方形内的单移动点，动作为坐标增量，单 episode 固定 17 步（16 个动作）。预训练 1M 序列、训练探针 300k 序列、评估 10k 序列；探针为单线性层，从表征回归点坐标。干扰噪声分 structured（叠加 CIFAR-10 图像）与 uniform 两类；时间维度分 changing（每步重采样）与 fixed（序列内不变、序列间重采样）；强度由系数 $\alpha$ 相对点亮度控制[^src-jepa-focus-on-slow-features]。

**对照**：JEPA+[[vicreg|VICReg]]、JEPA+SimCLR（InfoNCE）、重建、逆动力学建模（IDM），外加 Supervised（误差下界）与 Center（误差上界）[^src-jepa-focus-on-slow-features]。

**结果**（Figure 3）：无干扰时各方法均有效；重建在 $\alpha \le 1.5$ 各设置下保持有效；JEPA 在 fixed 噪声下失效（structured 与 uniform 均失效），原因为模型聚焦背景噪声而忽略移动点；changing 噪声下各方法均有效；未重新调参时 JEPA 仍能适应更强 changing 噪声，而重建在 changing uniform 噪声下性能显著退化（附录 A.5）。IDM 在各情形均有效，但 3 点实验显示其仅捕捉与 agent 动作相关的点[^src-jepa-focus-on-slow-features]。

**理论解释**（§2.1）：VICReg 与 SimCLR 目标可被平凡解最小化：编码器直接提取时间不变的噪声作为表征，predictor 收敛为恒等映射。此时 VICReg 的三项准则同时达到全局最优：不变性预测项（均方误差）因表征时间恒定为 0（Eq 1）；方差项（hinge loss）在噪声跨 episode 标准差达到阈值 $\sigma$ 时为 0（Eq 3）；协方差项因噪声分量跨 episode 统计独立、非对角协方差为 0 而降为 0（Eq 4）。SimCLR 依据 Wang & Isola 定理 1，在正对完全对齐且编码在单位超球面上均匀分布时 InfoNCE 取极小值，平凡解同时满足两条件。该失效源于伪相关（spurious correlation）：训练目标促使模型捕捉时间维度的慢特征，当固定背景噪声为输入中最慢特征时，表征会完全被噪声占据[^src-jepa-focus-on-slow-features]。

**3 点数据集**（附录 A.6，Table 1，17 步 RMSE，3 seeds）：三个图像通道分别对应动作控制点、动作未知的随机点与静止点。VICReg 仅捕捉静止点（静止 0.066±0.026，动作控制点 0.277±0.041，随机点 0.273±0.044）；SimCLR 表现类似（0.025±0.001 / 0.193±0.001 / 0.193±0.002）；IDM 仅捕捉动作控制点（0.035±0.000）而忽略静止点（0.272±0.000）；重建对两类点均有捕捉（0.021±0.001 / 0.026±0.002），Supervised 为 0.010±0.001 / 0.005±0.000[^src-jepa-focus-on-slow-features]。

**局限**（附录 A.8）：实验基于 toy dataset，结论外推至复杂视频与大模型尚待检验，且仅覆盖 VICReg 与 SimCLR 两种 JEPA 目标。改进方案包括：输入采用图像差分或光流；引入层级架构（[[hierarchical-jepa|H-JEPA]]）；修改目标函数以惩罚表征在时间维度的恒定性[^src-jepa-focus-on-slow-features]。

## 范围

失效实证基于单点 toy 环境与线性探针，复杂场景下的表现仍待验证。层级扩展见 [[hierarchical-jepa]]，作为世界模型的定位见 [[world-model]]，能量模型背景见 [[energy-based-model]]。「编码未来观测、在表征空间预测」的思想也已落地到时序预测（TimeAlign、LatentTSF、WorldTS），见 [[latent-space-future-state-prediction]][^src-worldts]。

## Related Pages

- [[visreg]] — 正则化路线的最新实现：scale/shape 解耦与 SWD sketching
- [[lejepa]] — 各向同性高斯最优性 + SIGReg 的单超参 JEPA 训练，本页启发式手段的替代方案
- [[non-contrastive-ssl]] — JEPA 的非对比训练判据与 VICReg 的谱系定位
- [[vicreg]] — VICReg 损失结构、系数与实验证据
- [[hierarchical-jepa]] — 多层级、多时间尺度的 JEPA 堆叠与分层规划
- [[world-model]] — 论文对世界模型的职责、不确定性与数据来源的规定
- [[energy-based-model]] — 能量函数作为基本对象、塌缩与正则化路线
- [[latent-space-future-state-prediction]] — 该思想在时序预测中的技术化落地（TimeAlign/LatentTSF/WorldTS）
- [[contrastive-learning]] — 对比方法与本文主张的正则化路线之别

[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
[^src-jepa-focus-on-slow-features]: [[source-jepa-focus-on-slow-features]]
[^src-worldts]: [[source-worldts]]
[^src-visreg]: [[source-visreg]]
[^src-lejepa]: [[source-lejepa]]
