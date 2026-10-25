---
title: "隐式残差专家（Implicit Residual Experts）"
type: concept
tags:
  - residual-networks
  - mixture-of-experts
  - sparsity
  - interpretability
created: 2026-10-25
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# 隐式残差专家

ResNet 没有专家模块，但 Möbius 分解出的单个残差修正或高阶 interaction 若幅值足够大，就承担了类似「被选中专家」的功能。论文称这类项为 **implicit residual expert**，并检验字典稀疏到什么程度[^src-resnets-route-without-routers]。分解见 [[residual-interaction-decomposition]]。

## 定义

字典为全部非空 interaction $\mathcal{R}=\{S\subseteq[L]:S\neq\emptyset\}$，$|\mathcal{R}|=2^L-1$（ResNet-18 255，ResNet-34 65,535）。幅值 $a_S^v=|\Delta_S v_x|$、$a_S^h=\lVert\Delta_S h_x\rVert_2$；输入 $x$ 的有效专家集是前 $K$ 大项 $\mathcal{A}_K^r(x)=\operatorname{TopK}_{S\in\mathcal{R}}a_S^r(x)$[^src-resnets-route-without-routers]。

「稀疏 Möbius 表示」指少量项就能恢复读数：

$$v_x(\mathbf{1}_{[L]})\approx\Delta_\emptyset v_x+\sum_{S\in\mathcal{A}_K^v(x)}\Delta_S v_x,\qquad K\ll|\mathcal{R}|.$$

这是**贡献稀疏**而非执行稀疏：普通前向仍跑完全部残差块。Top-$K$ 支撑从事后完整分解里挑出，度量 representational sparsity，不是可执行路径[^src-resnets-route-without-routers]。

## 系数质量集中

| 保留比例 | $K$（18 / 34） | 18 标量 $\Gamma^v$ | 18 向量 $\Gamma^h$ | 18 相对误差 | 18 一致率 | 34 标量 $\Gamma^v$ | 34 向量 $\Gamma^h$ | 34 相对误差 | 34 一致率 |
|---|---|---|---|---|---|---|---|---|---|
| 1% | 3 / 656 | 0.2445 | 0.1450 | 4.0016 | 0.0019 | 0.3077 | 0.1400 | 76.1582 | 0.0000 |
| 5% | 13 / 3,277 | 0.6140 | 0.4345 | 4.0782 | 0.0063 | 0.6597 | 0.4409 | 65.1384 | 0.0020 |
| 10% | 26 / 6,554 | 0.8161 | 0.6548 | 3.6881 | 0.0339 | 0.8125 | 0.6545 | 42.4853 | 0.0000 |
| 25% | 64 / 16,384 | 0.9751 | 0.9364 | 1.7686 | 0.3048 | 0.9528 | 0.8810 | 35.5789 | 0.0020 |
| 50% | 128 / 32,768 | 0.9988 | 0.9971 | 0.2850 | 0.8766 | 0.9947 | 0.9790 | 13.8741 | 0.0060 |
| 100% | 255 / 65,535 | 1.0000 | 1.0000 | $1.04\times10^{-6}$ | 1.0000 | 1.0000 | 1.0000 | $<10^{-12}$ | 1.0000 |

（Table 8；$\Gamma$ 为保留的平方系数质量比例，误差为向量相对 $\ell_2$，一致率对比全掩码预测；ResNet-18 $N=10{,}000$、ResNet-34 $N=1{,}000$。）

读法：**质量集中，决策不集中**[^src-resnets-route-without-routers]。

- ResNet-18 取 25% 字典（$K=64$）已得 93.64% 向量质量，但相对误差 1.7686、余弦 0.4896、一致率仅 30.48%；50%（$K=128$）才压到 0.2850 与 87.66%。
- ResNet-34 更极端：50% 字典仍有 13.8741 误差、0.6% 一致率，平均需 99.495% 的非空项才把向量误差降到 10% 以下。
- 90% 向量质量平均只需 21.870%（ResNet-18）与 27.754%（ResNet-34）的字典；5% 向量误差平均要 73.191% 与 99.833%（Table 10）。

论文把差值归因于带符号系数抵消与向量系数一般不正交：幅值大不等于净贡献可截断[^src-resnets-route-without-routers]。

## 是「哪些项」在起作用，不是「第几阶」

- 幅值 Top-$K$ 优于随机 $K$、保持各阶数量的 order-matched 随机与低阶截断：ResNet-18 在 $K=64$ 时一致率 30.48% 对 0.28%/0.44%；ResNet-34 在 25%/50% 预算下只是误差更小，三者一致率都接近 0（Table 9）[^src-resnets-route-without-routers]。
- 低阶截断另有一组反例：ResNet-18 保留到 7 阶（254/255 项）误差仍 2.6881、一致率 20.12%，补上唯一的 8 阶项才精确还原；ResNet-34 保留到 15 阶（65,534/65,535 项）误差 6.8803、一致率 2.10%[^src-resnets-route-without-routers]。
- 幅值 Top-$K$ 集合在阶上的分布偏向中高阶，与 [[interaction-order-spectrum|阶谱]] 的峰一致（Figure 9f/10f）[^src-resnets-route-without-routers]。

## 与其他「专家」概念的边界

- 与 [[mixture-of-experts|稀疏 MoE]]：MoE 专家是显式模块、由 router 按输入选中、选中即省算力；这里的「专家」是掩码干预下才可见的函数项，正常推理全部被计算[^src-resnets-route-without-routers]。
- 与 path-ensemble 视角：后者数结构路径（$2^L$ 条），这里的项是 inclusion–exclusion 去掉真子集效应后的**功能贡献**，不是同一组对象（Table 1 的朴素聚合失败即证据）[^src-resnets-route-without-routers]。
- 与特征/神经元归因（Taylor、Shapley）：那里的参与者是输入特征或通道，这里换成残差支路开关，得到支路及其组合的贡献[^src-resnets-route-without-routers]。

## 边界

只测 ResNet-18/34 两个 ImageNet 检查点，ResNet-34 侧 $N=1{,}000$；Top-$K$ 是事后解析支撑，不能推出「很多计算没用」或可加速[^src-resnets-route-without-routers]。集中性随深度减弱（ResNet-34 的决策重构接近 dense），且只在样本内验证。

[^src-resnets-route-without-routers]: [[source-resnets-route-without-routers]]
