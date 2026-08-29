---
title: "MeanFlow — Mean Flows for One-step Generative Modeling"
type: source-summary
tags:
  - flow-matching
  - one-step-generation
  - consistency-models
  - jvp
  - arxiv-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 0
confidence: low
status: active
---

# MeanFlow — Mean Flows for One-step Generative Modeling

**作者:** Zhengyang Geng, J. Zico Kolter（CMU）；Mingyang Deng, Xingjian Bai, Kaiming He（MIT）
**发表:** arXiv:2505.13447（v1，2025-05-19，cs.LG）；PDF 内无接收信息，[[loft|LOFT]] 著录的 NeurIPS Vol.38 未在 PDF 内核实
**raw:** `raw/geng-meanflow-one-step-generative-arxiv-2025.pdf`（arXiv v1 下载）

## 核心论点

1. **平均速度场**：定义 $u(z_t,r,t)\triangleq\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau$（Eq. 3）——位移除以时间区间——与 Flow Matching 建模的瞬时速度 $v$ 相对；$u$ 是 $v$ 诱导的泛函、不依赖任何网络，论文将其定位为新的 ground-truth field[^src-meanflow]。
2. **MeanFlow Identity**：$u=v-(t-r)\frac{du}{dt}$（Eq. 6），由定义式对 $t$ 求导（$r$ 固定）得到；沿轨迹全导数 $\frac{du}{dt}=v\,\partial_z u+\partial_t u$（Eq. 8），即 Jacobian 对切向量 $[v,0,1]$ 的 JVP。训练目标 $u_{tgt}=v_t-(t-r)(v_t\,\partial_z u_\theta+\partial_t u_\theta)$（Eq. 11）仅以条件速度 $v_t=\epsilon-x$ 为监督信号，目标端整体 stop-gradient 避免高阶优化；全程从头训练，无需预训练、蒸馏或课程学习[^src-meanflow]。
3. **CFG 内建于目标场**：构造 $v^{cfg}=\omega v(\cdot|c)+(1-\omega)v(\cdot)$ 的平均速度并由网络直接建模，采样保持 1-NFE（Sec. 4.2；附录 B.1 引入 κ 混合进一步改进）[^src-meanflow]。

## 实验结果（作者报告）

ImageNet-256 类条件生成（DiT 骨干、SD-VAE 潜空间、从头训练 240 epochs，Tab. 2）：MeanFlow-XL/2 1-NFE FID 3.43（对照 Shortcut-XL/2 10.60、iCT-XL/2 34.24，均为 1-NFE），2-NFE 2.93；XL/2+（1000 epochs）2-NFE 2.20，论文称与 250×2 NFE 的 DiT-XL/2（2.27）相当。消融（B/4，Tab. 1）：$r\ne t$ 占比 25% 最优、0%（即 Flow Matching）1-NFE 失效（FID 328.91）；自适应加权 $p=1$、lognorm(−0.4, 1.0) 时间采样、$(t, t-r)$ 条件化最优。CIFAR-10 无条件 1-NFE FID 2.92（Tab. 3）。

## 范围与局限

- 论文无独立局限性章节；实验限于 ImageNet-256 类条件与 CIFAR-10 无条件生成。
- JVP 有训练开销：作者报告 JAX 实现 B/4 上 0.045→0.052 s/iter，约 16% wall-clock（附录 B.4；正文称低于总训练时间 20%）。
- 正文引 SiT-XL/2 为 FID 2.15，Tab. 2 列 2.06——论文内两处不一致，分别归因记录。

## 相关页面

[[meanflow]] · [[alphaflow]] · [[shortcut-models]] · [[consistency-models]] · [[average-velocity-modeling]] · [[one-step-flow-generation]]

[^src-meanflow]: [[source-meanflow]]
