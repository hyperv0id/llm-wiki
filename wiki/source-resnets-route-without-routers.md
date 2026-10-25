---
title: "Do ResNets Route? Sparse Interaction Experts in Residual Networks"
type: source-summary
tags:
  - residual-networks
  - mixture-of-experts
  - interpretability
  - mobius-inversion
  - implicit-routing
created: 2026-10-25
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# Do ResNets Route? Sparse Interaction Experts in Residual Networks

Liang Yan、Siying Chen、Kaijie Chen、Bo Li、Jinghao Zhang、Mu Miao（复旦大学、华盛顿大学、同济大学、山东大学、Datacanvas），arXiv:2610.02907v1（cs.LG，2026-10-02，PDF 首页标注 Preprint，venue 未在文内出现）[^src-resnets-route-without-routers]。源文件：`raw/resnets-route-without-routers.txt`。

## 问题

标准 ResNet 对每个输入执行全部残差块，没有路由器。path-ensemble 视角只数结构路径（$2^L$ 条），不说明各残差支路单独或联合怎样支撑某次预测[^src-resnets-route-without-routers]。论文问：没有显式 router 的 ResNet，是否已在**功能贡献层面**表现出输入相关的路由。

## 方法

把训练好的 ResNet 写成二元掩码 $m\in\{0,1\}^L$ 上的集合函数（$m_l=0$ 只关残差支路），对布尔格做 Möbius 反演，得到唯一精确分解：shortcut-only 基线 $\Delta_\emptyset$、一阶修正、高阶 interaction，$2^L$ 项之和严格等于全掩码输出[^src-resnets-route-without-routers]。高阶项按幅值排序后称 implicit residual expert，Top-$K$ 集合即输入相关的「功能路由表」。

Proposition 2.3：残差映射与读出足够光滑时，$k$ 阶系数满足 $|\Delta_S v_x^{(\lambda)}|\le B_{S,x}\lambda^k$，即 $\lambda\to0$ 压制高阶 interaction[^src-resnets-route-without-routers]。

## 证据

ImageNet 预训练 ResNet-18（$L=8$，穷举 256 个掩码）与 ResNet-34（$L=16$，穷举 65,536 个掩码）[^src-resnets-route-without-routers]：

- 朴素聚合（直接求和 / 平均 / 基线居中）重构不出 logits：ResNet-18 相对误差 79.2/0.977/77.2，最优缩放后仍 0.959–0.961，top-1 一致率 0.6%；Möbius 重构在浮点精度内精确（Table 1）。
- 平方系数质量在 $k=5$（ResNet-18）与 $k=10$（ResNet-34）达峰，高于中点 $L/2$；有效阶 $\kappa^v=5.001/9.887$，$M_L/M_1=32.21/341.4$（Table 4、5，Figure 2）。
- 推理期缩放 $\lambda$ 压低 $\kappa$、抬升 $C_{\le3}$，但预测同时改变：ResNet-18 在 $\lambda=0.8$ 时与 $\lambda=1$ 的 top-1 一致率只剩 0.7314（Table 6、7）。
- 质量集中：ResNet-18 保留 25% 字典得 93.64% 向量质量，但相对误差 1.7686、一致率 30.48%；ResNet-34 保留 50% 仍有 13.8741 误差、0.6% 一致率（Table 8）。
- Top-$K$ 支撑类内比类间更相似（标量类差 0.125/0.044），13% 保留率下与全局集合的 Jaccard 达 0.871/0.892；困难样本的 interaction 复杂度几乎不变（最大 $|\rho|\le0.088$）（Figure 5、Table 11）。

结论：执行层面完全 dense，功能层面是围绕共享全局核的输入相关软路由[^src-resnets-route-without-routers]。

## 局限

只覆盖 ResNet-18/34 两个 ImageNet 检查点；ResNet-34 因穷举成本只取 $N=1{,}000$ 张图；Top-$K$ 是事后解析支撑，不代表稀疏执行或加速[^src-resnets-route-without-routers]。结论段自述，推广到更深网络需要高阶 interaction 的高效估计器。

[^src-resnets-route-without-routers]: [[source-resnets-route-without-routers]]
