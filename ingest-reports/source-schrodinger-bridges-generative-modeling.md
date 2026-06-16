# Ingest 报告：source-schrodinger-bridges-generative-modeling

## 创建
- wiki/source-schrodinger-bridges-generative-modeling.md — WHY：220 页综合性 SB 教程，涵盖静/动态 SB、SOC、6 种构造方法、6 种问题变体、连续/离散生成建模
- wiki/schrodinger-bridge.md — WHY：SB 核心概念，统一扩散/得分/流匹配等建模范式
- wiki/entropic-optimal-transport.md — WHY：SB 的理论起源—从 OMT 经 EOT 到 static SB
- wiki/sinkhorn-algorithm.md — WHY：求解 static SB 的经典交替投影算法，与 IMF/DSBM 结构同构
- wiki/hopf-cole-transform.md — WHY：将非线性 HJB-FP 线性化的关键技术
- wiki/girsanov-theorem.md — WHY：path-space 测度变换的数学基础
- wiki/stochastic-optimal-control-sb.md — WHY：SOC 视角的 SB，HJB 方程与三种训练 loss
- wiki/building-schrodinger-bridges.md — WHY：6 种互补 SB 构造方法综述
- wiki/doob-h-transform.md — WHY：通过概率倾斜构造条件随机过程
- wiki/iterative-markovian-fitting.md — WHY：交替 Markovian/reciprocal 投影的收敛理论
- wiki/diffusion-schrodinger-bridge-matching.md — WHY：IMF 的参数化实现算法
- wiki/adjoint-matching.md — WHY：基于 adjoint state 的高效 SB 训练方法
- wiki/adjoint-schrodinger-bridge-sampler.md — WHY：从 Boltzmann 分布采样的交替半桥优化
- wiki/conditional-score-flow-matching.md — WHY：免仿真的 conditional flow/score matching
- wiki/gaussian-schrodinger-bridge.md — WHY：Gaussian 边际下的闭式解，Bures-Wasserstein 几何
- wiki/discrete-schrodinger-bridge.md — WHY：CTMC 上的离散 SB
- wiki/fractional-schrodinger-bridge.md — WHY：长程依赖的 fBM + OU 近似
- wiki/multi-marginal-schrodinger-bridge.md — WHY：多时间点约束的 SB 扩展
- wiki/unbalanced-schrodinger-bridge.md — WHY：允许质量变化的非平衡 SB
- wiki/generalized-schrodinger-bridge.md — WHY：含 mean-field 交互的广义 SB
- wiki/branched-schrodinger-bridge.md — WHY：多模态终端的分叉 SB

## 修改
- wiki/index.md — WHY：添加所有 SB 页面到概念和技术分类
- wiki/log.md — WHY：记录本次 ingest 操作

## 新建交叉链接
实体 ↔ 概念：schrodinger-bridge ↔ 所有 21 个子页面
技术关联：sinkhorn-algorithm ↔ diffusion-schrodinger-bridge-matching ↔ iterative-markovian-fitting
统一框架：扩散模型、得分模型、流匹配 ↔ SB 统一视角
