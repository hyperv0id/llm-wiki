---
title: "Temporal Query Network for Efficient Multivariate Time Series Forecasting"
type: source-summary
tags:
  - time-series
  - forecasting
  - multivariate
  - attention
  - periodicity
  - channel-dependence
created: 2026-04-28
last_updated: 2026-10-25
source_count: 1
confidence: medium
status: active
---

# Temporal Query Network (TQNet)

## 论文信息

- **标题**: Temporal Query Network for Efficient Multivariate Time Series Forecasting
- **作者**: Shengsheng Lin, Haojun Chen, Haijie Wu, Chunyun Qiu, Weiwei Lin（华南理工大学）
- **发表**: ICML 2025，PMLR 267（PDF 首页页脚：Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada）
- **arXiv**: 2505.12917v2（2025-09-11）
- **raw**: `raw/2505.12917v2.md`
- **代码**: https://github.com/ACAT-SCUT/TQNet

## 核心贡献

### Temporal Query (TQ) 技术

给定周期长度 $W$，初始化可学习参数 $\theta_{\mathrm{TQ}}\in\mathbb{R}^{C\times W}$。时刻 $t$ 样本自 $t \bmod W$ 循环截取长度 $L$ 片段 $\theta_{\mathrm{TQ}}^{t,L}$ 作注意力 Query，K、V 仍取自输入 $X_t$（式 1–3）：

$$Q_h=\theta_{\mathrm{TQ}}^{t,L}W_h^{Q},\quad K_h=X_tW_h^{K},\quad V_h=X_tW_h^{V}$$

由构造 $\theta_{\mathrm{TQ}}^{t,L}=\theta_{\mathrm{TQ}}^{t+iW,L}$（式 2）：相隔 $W$ 步样本共用 query，参数经多样本共同训练，等效对局部噪声取平均。论文以此区分两类相关性：Q 承载全局稳定的变量间模式，K/V 承载样本级局部信息[^src-tqn]。

### 模型

TQNet 由单层 TQ-MHA、两层 GeLU MLP（均带残差）、输出线性层及可选 Instance Normalization 组成（Figure 2、Algorithm 1）。注意力在通道轴计算 $[C,C]$ 打分矩阵，$L$ 仅进入 Q/K/V 宽度[^src-tqn]。

## 实验与证据

- 12 个数据集（ETT×4、Electricity、Solar、Traffic、Weather、PEMS03/04/07/08），$L=96$：24 项误差指标中 22 项进前二，TimeXer 11 项、CycleNet 9 项（Table 2 Count 行）[^src-tqn]。
- Table 3 Query/Key 组合对比：七个高维数据集中六个以 (Q=TQ, K=Raw) 最优，优于 (Q=Raw, K=Raw) 与 (Q=TQ, K=TQ)；Traffic 例外，MSE 略高于全自注意力（0.445 vs 0.426）[^src-tqn]。
- 效率：单层注意力加浅层 MLP，参数量与训练时间均低于对比模型（Figure 7 左）；训练开销随通道数接近线性，$C<1000$ 时接近 DLinear（§5，Figure 7 右）[^src-tqn]。
- $W$ 敏感性（Figure 6）：$W$ 对齐真实周期（Electricity 取 168）效果最好；$W=24$ 只能捕捉部分周期；$W=23$、$167$ 等不对齐取值引入语义不一致，性能下降[^src-tqn]。

## 与本 Wiki 已收录论文的关系

- **CycleNet**：周期参数思路源于 CycleNet 可学习循环周期；CycleNet 用参数预测残差，TQ 将参数放入注意力作 Query[^src-tqn]。
- **iTransformer**：同为通道维注意力，但 iTransformer 的 Q/K/V 全部来自样本，TQ 仅替换 Query 来源[^src-tqn]。
- **PENGUIN**：两者均向注意力注入周期先验；TQ 替换 Query 来源，PENGUIN 改 key 侧加性偏置。

## 局限性

- 强依赖数据周期性：$W$ 需按稳定周期设定，缺乏明显周期的数据上泛化受限（§5）[^src-tqn]。
- 变量间弱相关时，强制多变量建模可能引入多余复杂度并降低性能（§5）[^src-tqn]。
- 回溯窗足够长时，多变量建模收益下降，长窗还会引入更多噪声（§5，Figure 8）[^src-tqn]。
- 每个数据集只设单一 $W$，多周期耦合（如周与月）需退化为日周期或集成多模型；论文称多周期建模仍是开放问题（附录 A.3）[^src-tqn]。

[^src-tqn]: [[source-tqn]]
