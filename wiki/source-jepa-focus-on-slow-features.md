---
title: "Source: JEPA Focus on Slow Features"
type: source-summary
tags:
  - source
  - self-supervised-learning
  - world-model
  - jepa
  - negative-result
created: 2026-10-13
last_updated: 2026-10-13
source_count: 1
confidence: medium
status: active
---

# Source: JEPA Focus on Slow Features (arXiv:2211.10831)

- **作者**：Vlad Sobal、Jyothir S V、Siddhartha Jalagam、Nicolas Carion、Kyunghyun Cho、Yann LeCun（NYU；Meta AI；Prescient Design, Genentech；CIFAR）
- **时间与出处**：2022-11-20（v1），4 页 3 图，NeurIPS 2022 SSL Theory and Practice workshop 短篇
- **链接**：https://arxiv.org/abs/2211.10831
- **代码**：https://github.com/vladisai/JEPA_SSL_NeurIPS_2022
- **证据源**：raw/ 中无此文件；依据 arXiv HTML 版（v1）全文。

## 核心论点

像素级重建目标不区分场景重要物体，易引发 object-vanishing；[[jepa|JEPA]] 提供无重建替代路径[^src-jepa-focus-on-slow-features]。论文在全离线、无奖励设定下，对比 JEPA（VICReg 与 SimCLR 目标）、重建式与逆动力学模型（IDM），用单线性探针检验移动点位置表征：时变噪声下 JEPA 持平或更优，固定噪声下失效，并给出理论解释[^src-jepa-focus-on-slow-features]。

## 设置与结果

环境是边长 1 的正方形内的单个移动点，动作为坐标增量，每 episode 固定 17 步（16 个动作）；预训练 1M 序列、探针 300k、评估 10k，探针为单线性层。噪声分 structured（叠加 CIFAR-10）与 uniform，时域分 changing 与 fixed，亮度由 α 相对点控制[^src-jepa-focus-on-slow-features]。结果（Figure 3）：重建在 α ≤ 1.5 时各设置均良好；JEPA 在 fixed 噪声下失效，作者假设原因是模型聚焦背景噪声而忽略点；changing 噪声增强时无需重调参仍能适应，而重建在 changing uniform 噪声下显著退化（附录 A.5）。IDM 各情形均好，但 3 点实验显示其仅捕捉与 agent 动作相关的点[^src-jepa-focus-on-slow-features]。

## 理论解释

VICReg 与 SimCLR 目标存在平凡解：编码器直接复制时不变噪声，predictor 退化为恒等映射，VICReg 预测项、方差项与协方差项均为 0（Eq 1–4）；SimCLR 亦满足 Wang & Isola 定理 1 的均匀性条件。作者把该失效归为 spurious correlation：目标促使模型捕获慢特征，当固定背景噪声成为最慢特征时，表征中仅存噪声[^src-jepa-focus-on-slow-features]。

## 局限

附录 A.8 指出局限：仅在 toy dataset 验证，能否外推至复杂视频与更大模型尚不明确，且仅检验 VICReg 与 SimCLR 两种 JEPA 目标[^src-jepa-focus-on-slow-features]。论文提出的缓解路径包括采用图像差分或光流输入、设计分层架构（[[hierarchical-jepa|H-JEPA]]），或在损失中加入阻止表征时序恒定的正则约束[^src-jepa-focus-on-slow-features]。

## Related Pages

- [[jepa]] — 该论文所检验的架构与完整失效证据
- [[non-contrastive-ssl]] — VICReg 与 SimCLR 所属的训练判据谱系
- [[source-autonomous-machine-intelligence]] — 提出 JEPA 的立场论文

[^src-jepa-focus-on-slow-features]: [[source-jepa-focus-on-slow-features]]
