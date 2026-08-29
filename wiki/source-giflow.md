---
title: "GiFlow: Spatiotemporal Imputation with Graph-Informed Flow Matching"
type: source-summary
tags:
  - flow-matching
  - spatio-temporal-imputation
  - graph-signal-processing
  - prior-distribution
  - icml-2026
created: 2026-08-29
last_updated: 2026-08-29
source_count: 0
confidence: low
status: active
---

# Source: GiFlow — Spatiotemporal Imputation with Graph-Informed Flow Matching

**GiFlow** 由 Zepeng Zhang、Aref Einizade、Jhony H. Giraldo、Olga Fink（EPFL、Télécom SudParis、Télécom Paris）发表于 ICML 2026（arXiv:2606.06682，2026-06-04），代码开源[^src-giflow]。

**raw:** `raw/giflow-graph-informed-flow-matching-imputation-arxiv26.pdf`

## 核心论点

论文认为扩散类时空插补方法依赖问题无关的各向同性高斯先验，与目标分布差异大，且需多步去噪与多次采样平均，效率和鲁棒性受限[^src-giflow]。GiFlow 利用流匹配对源分布的灵活性（论文引 Tong et al. 2024），以可观测信号经自适应时空图滤波构造**图信息先验**，使源分布贴近目标分布、缩短生成路径[^src-giflow]。

## 方法要点

- 将 vec(X₁ᴹ) 视为空间图与时间图笛卡尔积上的图信号，联合滤波算子为拉普拉斯 Kronecker 和，矩阵形式 X_τ = e^{−τηLη}X₁ᴹe^{−τξLξ}，其中 τη、τξ 控制空间/时间感受野[^src-giflow]。
- 滤波因子由"信号对齐 + α_τ 加权拉普拉斯平滑"的最小化问题在训练数据上以 SGD 优化、推理时固定[^src-giflow]。
- Proposition 3.1 给出 Taylor 截断误差界：滤波因子越小所需截断阶数越低，(τη, τξ) 构成自适应时空感受野[^src-giflow]。
- Theorem 3.2：取 α_τ=0 的最优滤波因子时，图信息先验出发的期望二次传输代价不高于高斯先验[^src-giflow]。
- 条件流为线性路径（起点为先验、终点为真实信号），论文称其在动能误差界意义下最优（引 Lipman et al. 2023）；向量场由空间注意力、时间注意力与时空传播（GNN）混合参数化[^src-giflow]。

## 实验结果（作者报告）

合成数据、Air-36、AQI、PeMS08 上 point/block missing（主表 ρ=20%，Air-36 另测 20–60%），基线含 FP、BRITS、SAITS、SPIN、GRIN、OPCR、PriSTI、CoSTI。GiFlow 在各设置的 MAE/RMSE/MAPE 总体最优（如 Air-36 point MAE 9.54 vs GRIN 9.94，PeMS08 point MAE 12.66 vs OPCR 12.77）；推理时间 Air-36 0.28 min vs PriSTI 9.30 / CoSTI 0.37，确定性输入无需多次采样平均；5 个 Euler 步仍优于次优基线；先验消融（FM-Gauss/TFM/GFM/GiFlow）中传输代价 299.62→104.29 单调下降、MAE 12.79→9.54，先验影响大于架构组件；滤波因子随缺失率增大，block missing 下 τη 增幅大于 τξ[^src-giflow]。

## 范围与局限

滤波因子优化依赖含完整真值的训练数据；性能依赖图结构质量，二值化阈值极端（0.02/0.6）时明显退化；实验仅覆盖 point/block 注入缺失；未与 LOFT 等同期低秩先验插补工作直接对比[^src-giflow]。

## 相关页面

[[giflow]] · [[graph-informed-prior]] · [[loft]] · [[tsflow]]

[^src-giflow]: [[source-giflow]]
