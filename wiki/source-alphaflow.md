---
title: "AlphaFlow — Understanding and Improving MeanFlow Models"
type: source-summary
tags:
  - flow-matching
  - consistency-models
  - few-step-generation
  - meanflow
  - gradient-conflict
  - arxiv-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 0
confidence: low
status: active
---

# AlphaFlow — Understanding and Improving MeanFlow Models

**作者:** Huijie Zhang, Aliaksandr Siarohin, Willi Menapace, Michael Vasilkovsky, Sergey Tulyakov, Ivan Skorokhodov（Snap Inc.）；Qing Qu（University of Michigan）。一作工作于 Snap 实习期间完成
**发表:** arXiv:2510.20771（v1，2025-10-23，cs.CV），页面无 venue 信息
**代码:** [github.com/snap-research/alphaflow](https://github.com/snap-research/alphaflow)
**raw:** `raw/alphaflow-understanding-and-improving-meanflow-models-arxiv25.pdf`（arXiv v1 下载）

## 核心论点

论文先解释 [[meanflow|MeanFlow]] 为何有效，再据此提出改进目标[^src-alphaflow]：

1. **目标分解**：代数变形可将 MeanFlow 损失恒等改写为轨迹流匹配 $L_{TFM}$ 与轨迹一致性 $L_{TCc}$ 两项之和（附录 D.1）。$L_{TCc}$ 是 $(t-r)$ 加权的连续一致性损失，自身不带边界条件；$L_{TFM}$ 隐式充当其边界条件，阻止模型坍缩为平凡解[^src-alphaflow]。
2. **梯度冲突**：DiT-B/2 在 ImageNet-256 训练中，$\cos(\nabla L_{TFM},\nabla L_{TCc})$ 通常低于 −0.4、95% 以上训练步强负相关。MeanFlow 将 75% 样本取 $r=t$ 的启发式实为 $L_{TFM}$ 的代理损失——与 $L_{TCc}$ 冲突更小，但占约四分之三训练计算[^src-alphaflow]。
3. **α-Flow 目标族**：$L_\alpha$ 以一致性步长比 $\alpha$ 在 $(r,t)$ 区间内定位中间时刻 $s=\alpha r+(1-\alpha)t$。定理 1：$\alpha=1$ 给出轨迹流匹配，$\alpha=1/2$（$\tilde v$ 取模型预测）给出 Shortcut Models（$L_{SC}=\tfrac12 L_\alpha$），$\alpha\to0$ 梯度等价 MeanFlow；$z_0$ 参数化且 $r\equiv0$ 时还涵盖离散/连续一致性训练。训练采用 sigmoid 课程将 $\alpha$ 从 1 退火至 0（温度 $\gamma=25$，两端夹紧 $\eta=5\times10^{-3}$），并推导出自适应损失权重 $\omega=\alpha/(\|\Delta\|^2+c)$[^src-alphaflow]。

## 实验结果（作者报告）

ImageNet-1K 256² 类条件生成，与 MeanFlow 相同的 DiT 骨干与 SD-VAE 潜空间。240 epochs 同预算对比：α-Flow-XL/2 FID 2.95 / FDD 164.6（1-NFE）、2.34 / 105.7（2-NFE），相对 MeanFlow-XL/2 同设置复现值（3.47 / 2.46）FID 提升 15%。α-Flow-XL/2+（240+60 epochs、batch 1024 微调）FID 2.58（1-NFE）、2.15（2-NFE；均衡类采样 1.95），论文称在 vanilla DiT 从头训练的少步模型中最优。消融：延长轨迹流匹配预训练起点（起点消融中以 Sigmoid150K→250K 最优）与更长更平滑的过渡（过渡时长消融中以 Sigmoid0K→400K 最优）均改善结果；α-Flow 在 25–50% 的 $r=t$ 流匹配占比下最佳，而 MeanFlow 需 75%[^src-alphaflow]。

## 范围与局限（论文自述）

- 连续（$\alpha\to0$）目标仍不可省略；大规模 + 引导设置下 MeanFlow 与 α-Flow 均偶发不稳定，作者明言这不解决一致性模型的稳定性问题。
- 梯度分析属实证性质，未从理论上解释流匹配监督为何关键（附录 B）。
- 附录 C 记录多项未成功尝试：分项损失加权、LoRA/独立预测头、约 50 组噪声调度消融、表征对齐加速等。

## 相关页面

[[alphaflow]] · [[meanflow]] · [[loft]] · [[trajectory-consistency-flow-matching]]

[^src-alphaflow]: [[source-alphaflow]]
