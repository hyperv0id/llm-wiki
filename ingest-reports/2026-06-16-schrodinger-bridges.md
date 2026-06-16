# Ingest 报告：2603.18992 — Foundations of Schrödinger Bridges for Generative Modeling

**日期**: 2026-06-16
**源文件**: raw/2603.18992.pdf (46MB, 220 页)

## 创建

| 页面 | WHY |
|------|-----|
| [[source-schrodinger-bridges-generative-modeling]] | 源文件摘要，220 页 SB 教程的完整结构和核心公式 |
| [[schrodinger-bridge]] | 主概念页，统一 diffusion/score/flow 的理论框架 |
| [[entropic-optimal-transport]] | EOT 基础概念，static SB 的前置理论 |
| [[sinkhorn-algorithm]] | 静态 SB 的核心求解算法 |
| [[hopf-cole-transform]] | 非线性 HJB-FP → 线性 PDE 的关键技术 |
| [[girsanov-theorem]] | Path measure change-of-measure 的数学基础 |
| [[building-schrodinger-bridges]] | 六种 SB 构造方法综述 |
| [[stochastic-optimal-control-sb]] | SOC 视角的 SB，HJB/value function/loss families |
| [[doob-h-transform]] | 条件随机过程的概率倾斜构造法 |
| [[iterative-markovian-fitting]] | 交替 Markovian/reciprocal KL 投影求解 SB |
| [[diffusion-schrodinger-bridge-matching]] | DSBM 算法，参数化 IMF 实现 |
| [[adjoint-matching]] | 通过 adjoint state + corrector matching 学习 SB |
| [[adjoint-schrodinger-bridge-sampler]] | 从 Boltzmann 分布采样的 SB-AM 方法 |
| [[conditional-score-flow-matching]] | [SF]²M，无需仿真的 SB 学习 |
| [[gaussian-schrodinger-bridge]] | Gaussian SB 的闭式解与 Bures-Wasserstein 几何 |
| [[discrete-schrodinger-bridge]] | 离散状态空间的 CTMC 扩展 |
| [[fractional-schrodinger-bridge]] | 引入长程依赖的 fBM reference |
| [[multi-marginal-schrodinger-bridge]] | 多中间约束的 SB |
| [[unbalanced-schrodinger-bridge]] | 允许质量变化的 SB |
| [[generalized-schrodinger-bridge]] | 含 mean-field 交互的 SB |
| [[branched-schrodinger-bridge]] | 多模态终端的 SB |

## 修改

| 页面 | WHY |
|------|-----|
| [[index]] | 新增 22 个页面到 Concepts 和 Techniques 分类 |
| [[log]] | 记录本次 ingest 操作 |

## 新建交叉链接

- [[schrodinger-bridge]] ↔ [[entropic-optimal-transport]] ↔ [[sinkhorn-algorithm]]
- [[schrodinger-bridge]] ↔ [[building-schrodinger-bridges]] → [[doob-h-transform]], [[iterative-markovian-fitting]]
- [[iterative-markovian-fitting]] → [[diffusion-schrodinger-bridge-matching]]
- [[schrodinger-bridge]] → [[gaussian-schrodinger-bridge]], [[fractional-schrodinger-bridge]], [[discrete-schrodinger-bridge]]
- [[adjoint-matching]] ↔ [[adjoint-schrodinger-bridge-sampler]] ↔ [[stochastic-optimal-control-sb]]
- SB 概念与已有页面 [[diffusion-models]], [[optimal-transport]], [[flow-matching]], [[score-based-generative-modeling]] 的桥接
