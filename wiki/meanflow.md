---
title: "MeanFlow"
type: technique
tags:
  - flow-matching
  - consistency-models
  - one-step-generation
  - few-step-generation
  - arxiv-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: low
status: active
---

# MeanFlow

**MeanFlow**（Geng, Deng, Bai, Kolter, He；arXiv:2505.13447，[[loft|LOFT]] 著录其为 NeurIPS Vol.38[^src-loft]）是少步流生成框架：训练网络 $u_\theta(z_t,r,t)$ 直接预测区间 $[r,t]$ 上的平均速度 $\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau$，推理时可从任意 $t$ 一步跳到 $r<t$[^src-alphaflow]。

> [!note] 来源限制
> MeanFlow 原文（arXiv:2505.13447）尚未单独收录。本页机制描述转述自 [[alphaflow|α-Flow]] 论文（其将 MeanFlow 作为分析与改进对象），细节以原文为准[^src-alphaflow]。

## 训练目标（据 α-Flow 转述）

$$L_{MF}(\theta)=\mathbb{E}_{t,r,z_t}\Bigl[\bigl\|u_\theta(z_t,r,t)-v_t+(t-r)\tfrac{d\,u_{\theta^-}(z_t,r,t)}{dt}\bigr\|^2\Bigr]$$

含 JVP（Jacobian-vector product）项，离散化即步长 $\Delta t\to0$ 的一致性目标[^src-alphaflow]。作者经验发现 75% 样本取 $r=t$（该切片上目标退化为普通流匹配监督）效果最好[^src-alphaflow]。

## 分解视角与实证位置

- [[alphaflow|α-Flow]] 将 $L_{MF}$ 分解为轨迹流匹配 $L_{TFM}$ 与轨迹一致性 $L_{TCc}$，并把 75% $r=t$ 启发式解释为 $L_{TFM}$ 的代理损失[^src-alphaflow]
- [[loft|LOFT]] 将 MeanFlow 与 Consistency-FM、[[shortcut-models|Shortcut Models]] 并列为一致性模型路线在图像生成已被验证的代表，并将自身定位为其在时空插补中的推广[^src-loft]
- α-Flow 论文在 ImageNet-256 复现 MeanFlow-XL/2（240 epochs）：FID 3.47（1-NFE）/ 2.46（2-NFE）[^src-alphaflow]

## 相关页面

- [[alphaflow]] — 对 MeanFlow 的分解分析与改进目标族
- [[consistency-models]] — 一致性模型源头工作
- [[shortcut-models]] — 同期少步生成方法
- [[loft]] — MeanFlow 路线在时空插补中的对照
- [[flow-matching]] — 理论基础

[^src-alphaflow]: [[source-alphaflow]]
[^src-loft]: [[source-loft]]
