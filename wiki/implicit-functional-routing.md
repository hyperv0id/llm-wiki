---
title: "隐性功能路由（Implicit Functional Routing）"
type: concept
tags:
  - residual-networks
  - routing
  - mixture-of-experts
  - interpretability
created: 2026-10-25
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# 隐性功能路由

标准 ResNet 没有 router，也不跳过任何块，却在**功能贡献**分配上与稀疏 MoE 相似：不同输入依赖不同残差 interaction，而这些项围绕共享字典组织。论文称这种「执行全 dense、贡献按输入变化」为 implicit soft routing[^src-resnets-route-without-routers]。

## 判据

论文只接受两条同时成立的证据才承认路由式解释（§2.1）[^src-resnets-route-without-routers]：

1. **集中性**：少量 interaction 就能保住读数（见 [[implicit-residual-experts]]）；
2. **系统性的输入依赖**：主导项的身份随输入、随类别变化，而不是所有输入共用一套固定的大项。

只有集中不构成路由：固定大项对谁都成立时，那只是可压缩表示。第二条用支撑集的 Jaccard 重叠测：

$$J_K^r(x_i,x_j)=\frac{|\mathcal{A}_K^r(x_i)\cap\mathcal{A}_K^r(x_j)|}{|\mathcal{A}_K^r(x_i)\cup\mathcal{A}_K^r(x_j)|},\qquad \Delta^r_{\mathrm{class}}(K)=\mathbb{E}\left[J_K^r\mid y_i=y_j\right]-\mathbb{E}\left[J_K^r\mid y_i\neq y_j\right]$$

向量读数另有按归一化质量加权的 $J_K^w$；对照为标签打乱后的 $\Delta_{\mathrm{class}}$（应接近 0）、随机 $K$ 集的 $J$、以及数据集级全局 Top-$K$ 集 $G_K$ 的重叠[^src-resnets-route-without-routers]。

## 证据

**类内比类间更相似，但差距集中在标量读数**[^src-resnets-route-without-routers]：

- 标量（predicted-class）读数类差峰值 ResNet-18 0.125、ResNet-34 0.044（Figure 5a,b），打乱标签后降到近 0。
- 向量读数类差峰值小得多：0.036 与 0.032（Table 11 在 1.18%/1.00% 保留率给 0.0362/0.0321，lift 1.061/1.066）。
- 标量读数是 decision-conditioned 的（坐标取该图全掩码预测类），故论文把它当补充诊断，主检验用无标签向量读数[^src-resnets-route-without-routers]。

**输入之间不是互斥路由，而是共享核上的偏移**：13% 保留率时向量支撑与全局集 $G_K$ 的平均 Jaccard 为 0.871/0.892（ResNet-18/34）；$K=32$ 时 1-NN 类检索 24.2%/9.9%，随机猜测 0.1%[^src-resnets-route-without-routers]。

| 模型 | 保留率 | $K$ | $J^{\mathrm{same}}$ | $J^{\mathrm{diff}}$ | $\Delta_{\mathrm{class}}$ | Lift | $J^{\mathrm{rand}}$ | $J^{\mathrm{global}}$ |
|---|---|---|---|---|---|---|---|---|
| 18 | 1.18% | 3 | 0.6334 | 0.5972 | 0.0362 | 1.061 | 0.0069 | 0.6900 |
| 18 | 3.14% | 8 | 0.6623 | 0.6318 | 0.0305 | 1.048 | 0.0167 | 0.7312 |
| 18 | 12.55% | 32 | 0.8329 | 0.8150 | 0.0179 | 1.022 | 0.0683 | 0.8705 |
| 18 | 25.10% | 64 | 0.9179 | 0.9096 | 0.0083 | 1.009 | 0.1438 | 0.9371 |
| 34 | 0.05% | 32 | 0.4068 | 0.3825 | 0.0243 | 1.064 | 0.0003 | 0.5004 |
| 34 | 1.00% | 656 | 0.5162 | 0.4842 | 0.0321 | 1.066 | 0.0050 | 0.5980 |
| 34 | 13.00% | 8,520 | 0.8639 | 0.8502 | 0.0137 | 1.016 | 0.0695 | 0.8921 |
| 34 | 25.00% | 16,384 | 0.8576 | 0.8431 | 0.0145 | 1.017 | 0.1429 | 0.8862 |

（Table 11，向量读数，类平衡 ImageNet-1K 验证子集，每类 10 张共 $N=10{,}000$；全掩码 top-1 准确率 ResNet-18 69.94%、ResNet-34 73.55%，见附录 I。）类差随保留率上升收窄，说明差异集中在少量高幅值项；类内重叠 0.41–0.92、与全局集重叠 0.50–0.94，变化幅度小于共享核。

**困难度几乎不改变 interaction 复杂度**[^src-resnets-route-without-routers]：难度用负置信度、负 margin、负 log-likelihood 度量，与支撑规模、阶、达到给定质量/误差阈值所需 $K$ 的 Spearman 相关最大 $|\rho|$ 仅 0.088（ResNet-18）与 0.074（ResNet-34）；NLL 与有效 interaction 数 $N^{h}_{\mathrm{eff}}=(\sum_S a_S^h)^2/\sum_S (a_S^h)^2$ 近乎无关（$\rho=0.031$ 与 0.027，Figure 5g,h）。变化发生在「**调用了谁**」而非「**调用了多少**」。

## 与既有机制的分界

- **稀疏 MoE / 条件计算**（[[mixture-of-experts]]）：router 显式选中少量专家、其余不算；这里的选择由功能贡献分布决定，所有块照样执行，无算力节省[^src-resnets-route-without-routers]。
- **path-ensemble 视角**：$2^L$ 条结构路径确实存在，但掩码子网络输出不能相加还原 logits（Table 1），故路径数不等于功能贡献项数；这里的项是 inclusion–exclusion 后可加的那一组[^src-resnets-route-without-routers]。
- **动态残差门控**（Highway、Residual Gates、SkipNet、BlockDrop）：它们显式学门或按输入跳块；本文不训练门，只把掩码当分析探针[^src-resnets-route-without-routers]。
- **MoE 方向相反的研究**：多数工作从零构造专家再学路由；这里问已训练 dense 网络中是否已有隐式专家结构[^src-resnets-route-without-routers]。

## 边界

- 只覆盖 ResNet-18/34 与 ImageNet 分类，未验证语言、扩散或其他骨干；结论段自述更深网络需要高阶 interaction 的高效估计器[^src-resnets-route-without-routers]。
- 路由是**功能性**的，不是现成的稀疏执行；论文区分 functional routing 与 sparse execution，不主张「很多计算没用」或可加速[^src-resnets-route-without-routers]。
- 标量与向量读数的类结构强度差一个量级，读数选择会改变结论强度，引用时须写明用哪个[^src-resnets-route-without-routers]。

[^src-resnets-route-without-routers]: [[source-resnets-route-without-routers]]
