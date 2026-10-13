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
source_count: 2
confidence: medium
status: active
---

# JEPA: Joint Embedding Predictive Architecture

LeCun 2022 年立场论文《A Path Towards Autonomous Machine Intelligence》提出的非生成式预测世界模型架构，学习多层级表征[^src-autonomous-machine-intelligence]。核心取舍是不预测 y 本身，只预测 y 的表征。

## 问题

自监督学习在论文中定义为捕捉输入各部分间的相互依赖，即判断观测部分 x 与未观测部分 y 是否兼容。架构不要求模型从 x 生成 y：与给定 x 兼容的 y 可能有无穷多个，显式表示该集合困难甚至不可解，而判断候选 y 是否兼容的代价低得多[^src-autonomous-machine-intelligence]。

像素级重建目标存在固有缺陷：重建损失按设计不区分场景中的重要物体，易发生 object-vanishing。Sobal 等人据此以 JEPA 作为重建替代方案展开实证[^src-jepa-focus-on-slow-features]。

## 机制

两个编码分支将 x、y 分别编码为 $s_x$、$s_y$。两个编码器无需共享参数，允许 x 与 y 属于不同性质信号（如视频与音频）。predictor 基于 $s_x$ 预测 $s_y$，可引入隐变量 $z$。能量定义为表征空间的预测误差[^src-autonomous-machine-intelligence]：

$$E_w(x,y,z) = D(s_y,\ \mathrm{Pred}(s_x,z)) \tag{10}$$

$$\check{z} = \arg\min_{z \in Z} E_w(x,y,z) = \arg\min_{z \in Z} D(s_y,\ \mathrm{Pred}(s_x,z)) \tag{11}$$

$$F_w(x,y) = \min_{z \in Z} E_w(x,y,z) = D(s_y,\ \mathrm{Pred}(s_x,\check{z})) \tag{12}$$

隐变量推断即针对 z 的能量最小化；消去 z 后 $F_w(x,y)$ 仅依赖 x 与 y（论文注明 $F_w$ 技术上对应零温自由能，但全文统称能量）[^src-autonomous-machine-intelligence]。

JEPA 放弃由 x 直接生成 y，换取表示多模态依赖的能力，途径有二[^src-autonomous-machine-intelligence]：

- **编码器不变性**：多个不同 y 映射到同一 $s_y$ 时能量相同，简化了多模态依赖的表示。
- **隐变量 z**：z 在集合 Z 上变化时，$\mathrm{Pred}(s_x,Z)$ 给出多个合理预测。

JEPA 可视作 Joint Embedding Architecture 与 Latent-Variable Generative Architecture 的结合，区别在于预测目标为 $s_y$ 而非 y[^src-autonomous-machine-intelligence]。

## 训练

JEPA 可用对比方法训练，但对比方法在高维下样本效率低（见 [[non-contrastive-ssl]]）。非对比（正则化）路线给出四条判据（Figure 13）[^src-autonomous-machine-intelligence]：

1. 最大化 $s_x$ 关于 x 的信息量
2. 最大化 $s_y$ 关于 y 的信息量
3. 让 $s_y$ 容易被 $s_x$ 预测
4. 最小化隐变量 z 的信息量

判据 1、2、4 共同防止能量面塌缩。其中判据 4 针对特定塌缩：若 z 与 $s_y$ 同维且 predictor 忽略 $s_x$ 直接复制 z，则 $\check{z}=s_y$ 导致能量恒为 0，能量面完全平坦。约束方式包括使 z 离散、低维、稀疏或有噪声；论文列出四类实现：离散化/量化（VQ-VAE）、维度/秩最小化（IRMAE）、稀疏化（线性稀疏建模、LISTA、非线性稀疏建模）、模糊化（noisy AE、VAE）[^src-autonomous-machine-intelligence]。

论文亦提出引入任务相关归纳偏置：在 $\tilde{s}_y$ 上附加预测头，预测容易从数据导出且与任务相关的变量[^src-autonomous-machine-intelligence]。

## 证据：固定干扰噪声下的失效

Sobal 等人针对 JEPA 在固定干扰噪声下的表征行为进行实证检验[^src-jepa-focus-on-slow-features]：

**设置**：全离线、无奖励，只有预录的观测-动作序列。环境是边长 1 的正方形内的单个移动点，动作是点的坐标增量，episode 固定 17 步（16 个动作）。预训练 1M 序列、训练探针 300k 序列、评估 10k 序列；探针是单线性层，从表征回归点坐标。干扰噪声分 structured（叠加 CIFAR-10 图像）与 uniform 两类；时间上分 changing（每步重采样）与 fixed（序列内不变、序列间重采样）；亮度由系数 $\alpha$ 相对点控制[^src-jepa-focus-on-slow-features]。

**对照**：JEPA+VICReg、JEPA+SimCLR（InfoNCE）、重建、逆动力学建模（IDM），外加 Supervised（误差下界）与 Center（误差上界）[^src-jepa-focus-on-slow-features]。

**结果**（Figure 3）：无干扰时各方法均有效；重建在 $\alpha \le 1.5$ 的各设置下均保持有效；JEPA 在 fixed 噪声下失效（structured 与 uniform 均失效），作者推测原因为模型聚焦背景噪声而忽略移动点；changing 噪声下各方法均有效；不重新调参时 JEPA 仍能适应更强的 changing 噪声，而重建在 changing uniform 噪声下性能显著退化（附录 A.5）。IDM 在各情形均有效，但 3 点实验显示其仅捕捉与 agent 动作相关的点[^src-jepa-focus-on-slow-features]。

**理论解释**（§2.1）：VICReg 与 SimCLR 目标可被平凡解最小化：编码器直接复制时间上不变的噪声作为表征，predictor 收敛到恒等映射。此时 VICReg 的预测项为 0（Eq 1）、方差项在 $\sigma$ 足够大时为 0（Eq 3）、协方差项因噪声分量跨 episode 独立而为 0（Eq 4）；SimCLR 依 Wang & Isola 定理 1，当正对完全对齐且编码输出在单位球上均匀时 InfoNCE 取极小，平凡解同时满足两条件。作者将该失效归因为 spurious correlation：目标促使模型聚焦“慢特征”，当输入中最慢特征属于固定背景噪声时，表征中仅保留噪声[^src-jepa-focus-on-slow-features]。

**3 点数据集**（附录 A.6，Table 1，17 步 RMSE，3 seeds）：三个图像通道分别对应动作控制点、动作未知的随机点与静止点。VICReg 仅捕捉静止点（静止 0.066±0.026，动作控制点 0.277±0.041，随机点 0.273±0.044）；SimCLR 结果类似（0.025±0.001 / 0.193±0.001 / 0.193±0.002）；IDM 捕捉动作控制点（0.035±0.000）但忽略静止点（0.272±0.000）；重建对两类点均有捕捉（0.021±0.001 / 0.026±0.002），Supervised 为 0.010±0.001 / 0.005±0.000[^src-jepa-focus-on-slow-features]。

**局限**（附录 A.8）：实验仅基于 toy dataset，结论能否外推至更复杂视频数据与大模型尚不明确；且仅检验了 VICReg 与 SimCLR 两种 JEPA 目标。论文给出的潜在改进方案包括：输入采用图像差分或光流；引入层级架构（即 [[hierarchical-jepa|H-JEPA]]）；或修改目标函数，增加约束阻止表征在时间维度保持恒定[^src-jepa-focus-on-slow-features]。

## 范围

失效实证基于单点 toy 环境与线性探针；向复杂场景的外推在原文中以 "we hypothesize" 表述，属待验证假设。层级扩展见 [[hierarchical-jepa]]，作为世界模型的定位见 [[world-model]]，能量模型背景见 [[energy-based-model]]。

## Related Pages

- [[non-contrastive-ssl]] — JEPA 的非对比训练判据与 VICReg 细节
- [[hierarchical-jepa]] — 多层级、多时间尺度的 JEPA 堆叠与分层规划
- [[world-model]] — 论文对世界模型的职责、不确定性与数据来源的规定
- [[energy-based-model]] — 能量函数作为基本对象、塌缩与正则化路线
- [[contrastive-learning]] — 对比方法与本文主张的正则化路线之别

[^src-autonomous-machine-intelligence]: [[source-autonomous-machine-intelligence]]
[^src-jepa-focus-on-slow-features]: [[source-jepa-focus-on-slow-features]]
