---
title: "Feedback Guidance of Diffusion Models"
type: source-summary
tags:
  - diffusion-models
  - dynamic-guidance
  - classifier-free-guidance
  - image-generation
  - neurips-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# Source: Feedback Guidance of Diffusion Models (FBG)

**作者**: Felix Koulischer, Florian Handke, Johannes Deleu, Thomas Demeester, Luca Ambrogioni（Ghent University - imec；Radboud University Donders Institute；Deleu 与 Ambrogioni 为共同资深作者）
**发表**: NeurIPS 2025 poster（arXiv:2506.06085，v1 2025-06-06 / v2 2025-10-09）
**代码**: https://github.com/FelixKoulischer/FBG_using_edm2
**raw 版本**: arXiv v2（`feedback-guidance-diffusion-models-arxiv25.pdf`，14 页含附录）

## 核心论点

论文提出 FeedBack Guidance（FBG），将扩散模型的引导尺度从固定超参数变为去噪状态与时间的函数，按需自调节引导量；出发点是 [[classifier-free-guidance|CFG]] 的固定引导无条件作用于所有样本，损害多样性并诱发 memorization，且近年工作（如 LIG）表明全轨迹施加引导既无必要也有害[^src-fbg]。

## 方法

论文把引导公式解释为对误差模型的反转——无条件分布如何"污染"学到的条件分布：CFG 隐式假设乘性混合，FBG 改为加性混合 $p_{\theta,t}(x_t|c) = (1-\pi)p_t(x_t) + \pi p_t(x_t|c)$，由此从第一性原理导出状态与时间相关的引导尺度 $\lambda(x_t,t)$，表达为后验似然的函数：后验比高时 $\lambda \to 1$，趋近 $1-\pi$ 时发散[^src-fbg]。后验通过沿反向马尔可夫链追踪条件/无条件似然比迭代估计，附加偏移 $\delta$ 修正条件模型"用自己输出评自己"的自我参照偏差；$\tau,\delta$ 重参数化为更直观的 $t_0,t_1$[^src-fbg]。论文自称是首个从第一性原理导出状态与时间相关引导尺度的工作[^src-fbg]。

## 关键结果（作者报告）

- ImageNet 512×512、EDM2-XS、64 NFE、随机采样器：FBG_pure 优于 CFG 与 LinCFG，与 LIG 相当（FID 3.76 vs CFG 5.00；FD_DinoV2 89.0 vs CFG 100.2，表 1）；PFODE 设定下亦优于 CFG++（FID 2.50 vs 3.66，表 1）；FBG_LIG 组合在 FD_DinoV2 上优于两个组成部分[^src-fbg]
- Stable Diffusion 2 文生图、MS-COCO 3k prompts：FBG_pure FID 18.63 vs CFG 19.64（表 2）；FBG 对越难的 prompt 自动施加越强引导，对 memorized prompt 引导尺度接近 1（图 4）[^src-fbg]

## 与 FENCE 的关系

[[fence|FENCE]]（AAAI 2026）将 FBG 的引导尺度公式与后验追踪机制引入时空交通数据插补，并增加聚类感知引导与两阶段训练[^src-fence]。两点差异：FENCE 采用省略 $p_\theta(c)$ 归一化的公式简写；FENCE 报告 $\pi=0.5$ 最佳，而 FBG 论文在 ImageNet 上用 $\pi \geq 0.999$ 且结果对 $\pi$ 不敏感[^src-fbg][^src-fence]。

## 局限性（论文自述）

- 加性混合模型与先验选择均以数学简单性为准，未必贴近训练后模型的真实系统偏差[^src-fbg]
- 实验限于 EDM2-XS 规模，更大架构（EDM2-L、DiT）与更大规模 T2I 评测留待后续[^src-fbg]
- 论文认为 MS-COCO prompts 均匀且偏简单，不是比较引导方法的理想基准[^src-fbg]

[^src-fbg]: [[source-feedback-guidance-diffusion-models-arxiv25]]
[^src-fence]: [[source-fence]]
