# Ingest 报告：2412.18730v4（Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers）

日期：2026-09-09。源：`raw/2412.18730v4.pdf`（79 页，arXiv v4，ICML 2025 Poster）。

## 创建
- wiki/source-2412-18730.md — WHY：每份 raw/ 源文件对应一个 source-summary 页，总结核心论点（denoiser 决定 ODE 动力学）、主要定理、证据与论文自述局限
- wiki/fm-ode-trajectory-stages.md — WHY：论文主框架"denoiser 吸引/吸收元定理 + 三阶段轨迹图景"是可独立引用的核心概念，补上 wiki 中 FM 采样动力学（相对 FM 训练侧）的理论空白，与 flow-matching/meanflow/one-step-flow-generation 等现有页面形成交叉
- wiki/fm-ode-terminal-convergence.md — WHY：终端收敛定理（作者称首个覆盖低维子流形支撑的 FM ODE 收敛结果）+ 两类终端奇异性 + denoiser→投影收敛 + 流映射等变性自成知识链条，直接支撑 consistency-models 与少步采样页面
- wiki/diffusion-memorization-geometry.md — WHY：记忆现象的几何解释（Voronoi 吸收强度 σ₀、重复样本抬权重、渐近最优 denoiser 记忆必然性、终端正则化建议）是独立的可引用概念，连接复制/隐私/正则化话题

## 修改
- wiki/flow-matching.md — WHY：FM 主概念页的相关页面列表补三条理论页反向链接（仅结构性变更，无新事实断言），last_updated 更新
- wiki/x-prediction.md — WHY：$x_1$-prediction 即 denoiser，正文补"Denoiser 的动力学角色"一节（带 [^src-2412-18730] 引用）把参数化问题与轨迹动力学接通，source_count 2→3
- wiki/probability-flow-ode.md — WHY：补"轨迹收敛与终端阶段"小节（分布级收敛 ≠ 轨迹收敛；终端位移 $O(\sigma_t^{\zeta/2})$ 是少步 ODE 采样的理论依据之一），source_count 4→5
- wiki/consistency-models.md — WHY：补"轨迹收敛：学习 Ψ₁ 的理论前提"小节（CM 学习的一致性函数以轨迹良好终点为默认前提），source_count 6→7
- wiki/index.md — WHY：登记 4 个新页面（1 source + 3 concepts），last_updated 更新
- wiki/log.md — WHY：按仅追加约定记录本次 ingest（含矛盾检查结论：无冲突）

## 新建交叉链接
- [[source-2412-18730]] ↔ [[fm-ode-trajectory-stages]] / [[fm-ode-terminal-convergence]] / [[diffusion-memorization-geometry]]
- [[fm-ode-trajectory-stages]] ↔ [[flow-matching]]、[[meanflow]]、[[x-prediction]]、[[fm-ode-terminal-convergence]]
- [[fm-ode-terminal-convergence]] ↔ [[consistency-models]]、[[probability-flow-ode]]
- [[diffusion-memorization-geometry]] ↔ [[classifier-free-guidance]]、[[fm-ode-trajectory-stages]]

## 校验口径
- venue ICML 2025 Poster 来自 icml.cc（poster/44549）与 OpenReview（f5czhqYK3H）著录；论文标题、定理编号、数值（σ_init=13、σ_cluster=0.65、σ_terminal=0.007/0.004、CIFAR-10 σ₀≈0.17、相对误差 <1%、距离 ~1.0 vs 均值范数 27.2）均出自 PDF 正文/附录（pdftotext 全文读取）。
- 三个概念页 confidence: medium（单一来源支持、无反驳证据），source-summary 页 source_count: 0（按仓库既有 source 页惯例）。
