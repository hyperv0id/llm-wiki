---
title: "GWNet 为何在 2026 年仍是主力基线"
type: analysis
tags:
  - traffic-forecasting
  - baseline-analysis
  - adaptive-graph-learning
  - spatio-temporal
created: 2026-09-04
last_updated: 2026-09-04
source_count: 10
confidence: high
status: active
---

# GWNet 为何在 2026 年仍是主力基线

Graph WaveNet（[[gwnet|GWNet]]，Wu et al., IJCAI 2019）发布已 7 年，但 2024–2026 年的论文仍把它作为标准基线、默认骨干，甚至在最新大规模基准上保有竞争力。本页从仓库已有源文件中整理证据，并区分「论文自述」「实验结果」「本 wiki 分析」三个层级。

## 证据一：大规模基准上仍有一战之力（STGformer, 2024）

[[stgformer|STGformer]]（arXiv:2410.00385）在 LargeST 三个子集（SD/BA/LA，716–3,834 节点）上把 GWNET 作为 14 个基线之一对比，数字来自 Table I[^src-stgformer]：

| 场景 | GWNET 参数 | GWNET 平均 MAE | 场景最优（MAE） | 差距 |
|------|-----------|---------------|----------------|------|
| San Diego (N=716) | 311K | 17.74 | STGformer 17.36 | +2.2% |
| Bay Area (N=2,352) | 344K | 20.91 | STGformer 19.98 | +4.7% |
| Los Angeles (N=3,834) | 374K | 21.20 | STGformer 19.58 | +8.3% |

两个细节值得注意[^src-stgformer]：

1. **BA 平均 RMSE 一格 STGformer 自己承认被 GWNET 超过**（33.50 vs 33.41）——2024 年的图 Transformer 在部分指标上输给 2019 年的模型，论文原文按格如实报告。
2. **跨年泛化（Table II，2019 训练 → 2020 测试）**：GWNET 在 LA 平均 RMSE 40.85 / MAPE 22.51% 优于 STGformer 的 41.76 / 27.04%；SD 平均 MAE 24.58 距最优的 STGformer（23.92）仅 2.8%，中间只隔 STID（24.52）。换句话说，在分布偏移下 7 年前模型的稳健性没有掉队。

## 证据二：2026 年新方法仍把它当默认骨干

- **SCALE（ICML 2026）**：谱共形预测工作在 METR-LA/PEMS04/07/08 上的附录 E.1 稳健性消融（Table 4 骨干配置、Table 7）把 GWNet（tsl 默认超参）列为第一阶段点预测骨干之一（与 Transformer、DCRNN 并列），并另以 GWNet 骨干给出全套区间结果（附录 Table 6）[^src-scale]。
- **USTD（SIGSPATIAL 2024）**：统一时空扩散框架的编码器直接采用 GWNet 式骨干（gated 1D conv + GCN + skip connections）做预训练，并称以此解决了此前扩散时空模型打不过确定性基线的问题[^src-ustd]。
- **IGSTGNN（KDD 2026）**：把 ICSF/TIID 作为即插即用模块集成到 GWNET 时同样带来一致提升——插件式改进仍以 GWNet 为宿主之一[^src-incident-guided-st-forecasting]。

## 证据三：后续研究以「修 GWNet 的毛病」为论文骨架

一个模型被反复修补，恰是它仍处于问题中心的证据。仓库内的修法至少覆盖四条独立轴[^src-mage][^src-ragc-efficient-traffic-forecasting][^src-st-ood]：

| 被修的毛病 | 修补工作 | 修法 |
| O(N²) 邻接构建 | [[bigst|BigST]]（PVLDB 2024） | PRF 核因式分解线性化为 O(N)，扩到 ~100K 节点[^src-bigst] |
| ReLU 截断放大伪边 | [[mage|MAGE]]（NeurIPS 2025） | 去掉 ReLU + kernel 近似，[[edge-noise-amplification|边噪声放大]]理论 |
| 静态图不随时间变 | [[dpgnet|DPGNet]]（ICLR 2026 u.r.） | [[adaptive-graph-learner|AGL]] 逐时刻动态图，即插即用替换（MAE 改善 3.52–5.51%）[^src-dpgnet] |
| 节点嵌入占 72% 参数 | [[ragc|RAGC]]（arXiv 2026） | ECO 正则 + 余弦相似度图卷积 |
| 跨年泛化弱 | ST-OOD 基准（IEEE TMC 2025） | 消融：自适应邻接助 IN、OUT 误差 +9.7% |

## 为什么「还能打」：机制层面的四个原因

以下为本 wiki 的分析，基于上述源文件，非任一论文的自述。

1. **公式够简单，坏处可修**。$\tilde{A}_{adp} = \text{SoftMax}(\text{ReLU}(E_1E_2^\top))$ 只有一步矩阵乘、一个激活、一个归一化。正因为它足够「素」，后续工作能精确定位单个组件（ReLU、N×N 构建、静态性）逐一替换并归因收益——MAGE 能单独做 w/o ReLU 消融并测得全线下降[^src-mage]，这种「可拆解性」是复杂架构给不了的。

2. **计算剖面仍然优秀**。非自回归输出（12 步一次前向）使其推理速度在 2019 年即为对比方法最快（2.27s vs DCRNN 18.73s）[^src-gwnet]；在 2026 年的共形预测流水线里，它的定位是「提供高质量残差的第一阶段骨干」——残差流水线对骨干的要求正是快、稳、点预测够好，而这三点 GWNet 至今满足[^src-scale]。

3. **基线惯性 + 工程可用性**。tsl 等标准库的默认实现使其成为新论文验证「我的复杂方法确实在起作用」的最便宜对照组：SCALE 直接以 tsl 默认超参跑 GWNet 做第一阶段骨干，不调参也能得到合理残差[^src-scale]。基线存在本身增强了基线的存在——这是评测生态的自我强化，论文普遍如此使用但不以此为荣。

4. **它定义的问题还没被解决完**。自适应图学习范式（数据里学空间依赖，而非用路网硬编码）仍是 2026 年论文的核心命题——MAGE 的 MoE 图专家、DPGNet 的动态图、RAGC 的正则化都在这一范式内工作。**GWNet 的持续在场不是因为它没被超越，而是因为它提出的坐标系仍在被使用。**

## 边界：什么情况下 GWNet 已经不够

- **跨年 OOD 场景**：ST-OOD 基准测得 GWNet 自适应邻接在 OUT 年误差约 +9.7%，属于「结构组件是双刃剑」的证据之一；简单模型（STID/MLP）跨年平均更稳[^src-st-ood]。
- **10K+ 节点规模**：MAGE 报告 AGCRN/GWNet/D²STGNN 在 Shanghai Mobile（3,042 节点）与 Milan Internet（10,000 节点）上因 O(N²) 复杂度直接 OOM 无法部署[^src-mage]——此时线性化的继承者（BigST/MAGE/GSNet）才是可用选项。
- **纯精度竞赛**：在 MAGE 的 17 数据集评测中，GWNet 是被比较的 14 个基线之一，94% 指标上 SOTA 是 MAGE 而非 GWNet[^src-mage]。「能打」指不掉队与仍被用作试金石，不指它仍是精度上限。

## 结论

「2019 年的模型在 2026 年仍然能打」这个判断在仓库证据下成立，但需要拆开说：**作为精度上限它已被超越**（MAGE 等在指标上全面领先）；**作为基础设施它仍然在场**（2026 年新论文的骨干、基线、修补对象）；**作为问题坐标系它仍然有效**（自适应图学习仍是活跃命题）。一个模型的生命周期终点不是被超越，而是其提出的问题被终结——对 GWNet 而言这个终点还没有到。

## 相关页面

- [[gwnet]] — GWNet 技术页：架构、消融、谱系表
- [[source-gwnet]] — 原论文摘要
- [[stgformer]] — LargeST 对比数字的来源页
- [[mage]] / [[edge-noise-amplification]] — ReLU 缺陷与线性化修补
- [[ragc]] / [[node-embedding-regularization]] — 参数占比问题
- [[bigst]] / [[linearized-spatial-convolution]] — O(N) 线性化路线
- [[dpgnet]] / [[adaptive-graph-learner]] — 动态图修补路线
- [[st-ood]] — 跨年泛化基准
- [[traffic-forecasting]] — 领域总览

[^src-gwnet]: [[source-gwnet]]
[^src-stgformer]: [[source-stgformer]]
[^src-scale]: [[source-scale]]
[^src-ustd]: [[source-ustd]]
[^src-incident-guided-st-forecasting]: [[source-incident-guided-st-forecasting]]
[^src-mage]: [[source-mage]]
[^src-bigst]: [[source-bigst]]
[^src-dpgnet]: [[source-dpgnet]]
[^src-ragc-efficient-traffic-forecasting]: [[source-ragc-efficient-traffic-forecasting]]
[^src-st-ood]: [[source-st-ood]]
