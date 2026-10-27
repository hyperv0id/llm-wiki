---
title: "Manifold-Constrained Hyper-Connections (mHC)"
type: technique
tags:
  - residual-connections
  - hyper-connections
  - manifold-constraint
  - doubly-stochastic
  - training-stability
  - large-language-models
created: 2026-06-22
last_updated: 2026-10-27
source_count: 3
confidence: high
status: active
---

# Manifold-Constrained Hyper-Connections (mHC)

mHC 属于宏观架构设计框架，通过对 [[hyper-connections|Hyper-Connections]] 的残差连接空间施加流形约束，在扩展残差流宽度的同时恢复残差连接的恒等映射[^src-mhc-manifold-constrained-hyper-connections]。

## 动机

标准残差连接 $x_{l+1} = x_l + F(x_l, W_l)$ 依赖恒等映射：深层表示可分解为浅层信号与沿途残差函数之和，从而抑制梯度消失或爆炸[^src-mhc-manifold-constrained-hyper-connections]。HC 则将残差流从 $C$ 维扩展至 $n \times C$ 维，并引入可学习的 $H_l^{res}$ 在多个残差流间混合信息；该无约束操作破坏了恒等映射，导致多层复合映射可能无界放缩信号[^src-mhc-manifold-constrained-hyper-connections]。

## 数学形式

mHC 沿用 HC 的单层更新（原文式 3）：

$$x_{l+1} = H_l^{res} x_l + H_l^{post\top} F(H_l^{pre} x_l, W_l)$$

其中第 $l$ 层的隐藏状态 $x_l \in \mathbb{R}^{n \times C}$ 组织为 $n$ 个并行残差流；$H_l^{pre}, H_l^{post} \in \mathbb{R}^{1 \times n}$ 分别从 $n$ 流残差聚合出 $C$ 维层输入，并将层输出映射回流；$H_l^{res} \in \mathbb{R}^{n \times n}$ 负责残差流间的信息交换[^src-mhc-manifold-constrained-hyper-connections]。多层复合时（原文式 4），浅层信号项由复合残差映射主导：

$$x_L = \Big(\prod_{i=1}^{L-l} H_{L-i}^{res}\Big) x_l + \sum_{i=l}^{L-1} \Big(\prod_{j=1}^{L-1-i} H_{L-j}^{res}\Big) H_i^{post\top} F(H_i^{pre} x_i, W_i)$$

HC 的不稳定源于 $\prod_{i=1}^{L-l} H_{L-i}^{res}$ 无约束：它不保持特征全局均值，导致前向与反向传播中信号无界放缩或衰减（§1、§3.1）[^src-mhc-manifold-constrained-hyper-connections]。

关键区别在于 $H_l^{res}$ 被投影到 [[birkhoff-polytope|Birkhoff 多面体]] 上，即满足（原文式 6）：

$$H_l^{res} \mathbf{1}_n = \mathbf{1}_n, \quad \mathbf{1}_n^\top H_l^{res} = \mathbf{1}_n^\top, \quad H_l^{res} \geq 0$$

的双随机矩阵流形[^src-mhc-manifold-constrained-hyper-connections]。

## 理论性质

论文在 §4.1 列举双随机约束带来的三条性质。全文未设定理（Theorem/Lemma/Proof）环境，附录仅列超参表，各性质均未以定理形式陈述或证明。以下证明骨架基于双随机定义补全推导链条[^src-mhc-manifold-constrained-hyper-connections]。

1. **非扩张性（论文称 Norm Preservation；作者论列，内容为可证的标准事实）**：双随机矩阵的谱范数不超过 1，即 $\|H_l^{res}\|_2 \leq 1$，因此 $\|H_l^{res} x_l\|_2 \leq \|x_l\|_2$，抑制前向信号爆炸与反向梯度爆炸[^src-mhc-manifold-constrained-hyper-connections]。证明骨架：由 Birkhoff–von Neumann 定理写 $H = \sum_k \lambda_k P_k$（$\lambda_k \geq 0$、$\sum_k \lambda_k = 1$、$P_k$ 为置换矩阵），谱范数是凸函数，故 $\|H\|_2 \leq \sum_k \lambda_k \|P_k\|_2 = 1$，其中最后一步用置换矩阵是 $\ell_2$ 等距算子。
2. **乘法封闭性（论文称 Compositional Closure；标准事实）**：若 $A, B$ 双随机，则 $(AB)\mathbf{1}_n = A(B\mathbf{1}_n) = \mathbf{1}_n$，同理 $\mathbf{1}_n^\top (AB) = \mathbf{1}_n^\top$，且非负性在乘积下保持，故 $AB$ 双随机。因此任意深度的复合残差映射 $\prod_{i=1}^{L-l} H_{L-i}^{res}$ 仍是双随机矩阵，行/列和为 1 的守恒性质在全深度成立[^src-mhc-manifold-constrained-hyper-connections]。
3. **置换凸组合解释（论文称 Geometric Interpretation via the Birkhoff Polytope；前半为经典定理，后半为作者论断）**：[[birkhoff-polytope|Birkhoff 多面体]]是置换矩阵的凸包（Birkhoff–von Neumann 定理），$H_l^{res} x_l$ 可理解为对 $n$ 流特征的置换凸组合[^src-mhc-manifold-constrained-hyper-connections]。论文进一步断言重复应用此类矩阵会单调增加跨流信息混合、充当稳健的特征融合机制——「混合单调增加」「稳健融合」这两个判断论文未证明，按作者论断对待。

## 恒等映射恢复的推导骨架

双随机约束恢复恒等映射的推导链条（命题出自原文 §1/§4.1，证明步骤由本页补全）[^src-mhc-manifold-constrained-hyper-connections]：

前向 $\mathbf{1}_n^\top (H_l^{res} x_l) = \mathbf{1}_n^\top x_l$ 保持跨流信号均值不变；反向经 $H_l^{res\top}$ 运算，转置同样双随机，梯度质量同理守恒。复合映射在深度 0 为双随机的恒等矩阵；归纳步设前 $k$ 层复合 $P_k$ 双随机，则 $P_{k+1} = H_{l}^{res} P_k$ 仍双随机（性质 2），且 $\|P_{k+1}\|_2 \leq 1$（性质 1）。式 (4) 的浅层信号项因而满足 $\big\|\prod_i H_i^{res}\, x_l\big\|_2 \leq \|x_l\|_2$，在任意深度排除无界放缩。实测对照中，HC 复合映射 Amax Gain Magnitude 峰值近 3000（§3.1、Fig. 3b），mHC 降至约 1.6（§5.4、Fig. 7b）[^src-mhc-manifold-constrained-hyper-connections]。$n = 1$ 时唯一的 $1 \times 1$ 双随机矩阵为 $[1]$，$H_l^{res} x_l = x_l$，严格恢复原始残差连接的恒等映射（§4.1）[^src-mhc-manifold-constrained-hyper-connections]。

## Sinkhorn 迭代的收敛前提

论文在式 (9) 后断言迭代「converges to a doubly stochastic matrix as $t_{max} \to \infty$」，仅引用 Sinkhorn & Knopp (1967)，未讨论矩阵支撑或正性条件等收敛前提，附录亦无证明；实验取 $t_{max} = 20$（§4.2）[^src-mhc-manifold-constrained-hyper-connections]。经典 Sinkhorn–Knopp 定理要求初始矩阵具有 total support，此处 $M^{(0)} = \exp(\tilde H_l^{res})$ 逐元素严格为正，满足该前提。§5.4 实测了有限迭代的影响：单层映射的后向梯度增益略偏离 1（Fig. 7a），复合映射偏差有界、最大约 1.6（Fig. 7b）[^src-mhc-manifold-constrained-hyper-connections]。

## 参数化与投影

mHC 沿用 HC 的动态/静态映射分解：把 $x_l \in \mathbb{R}^{n \times C}$ 展平为 $\vec{x}_l \in \mathbb{R}^{1 \times nC}$ 做 RMSNorm，经线性投影得到 $\tilde H_l^{pre}, \tilde H_l^{post}, \tilde H_l^{res}$（式 7）。与 HC 的系数公式（式 5）相比，mHC 去掉了 $\tanh$。最终约束映射由式 (8) 给出：

$$H_l^{pre} = \sigma(\tilde H_l^{pre}), \qquad H_l^{post} = 2\sigma(\tilde H_l^{post}), \qquad H_l^{res} = \operatorname{Sinkhorn-Knopp}(\tilde H_l^{res})$$

其中 $\sigma(\cdot)$ 为 Sigmoid；$H_l^{post}$ 带幅度因子 2，取值范围 $(0, 2)$。对 $H_l^{pre}, H_l^{post}$ 施加非负特殊流形投影，防止正负系数复合导致信号抵消（§4.1）[^src-mhc-manifold-constrained-hyper-connections]。[[sinkhorn-algorithm|Sinkhorn-Knopp 算法]] 先对 $\tilde H_l^{res}$ 逐元素取指数，再交替做行/列归一化（式 9），实验取迭代数 $t_{max} = 20$[^src-mhc-manifold-constrained-hyper-connections]。

## 效率优化

为使 $n$ 流设计在硬件上可行，mHC 实现了三项系统级优化；$n = 4$ 时大规模训练的时间开销为 6.7%（§1、§4.3）[^src-mhc-manifold-constrained-hyper-connections]。

核融合（§4.3.1）包含三个系数核与两个应用核：系数侧由统一核计算 $\tilde{\tilde H}_l = \vec x_l \varphi_l$ 与 RMSNorm 的分母 $r = \sqrt{\vec x_l \vec x_l^\top / nC}$（式 14–15，把除以范数移到矩阵乘之后，数学等价）；小核完成 $1/r$ 缩放、$\alpha$ 门控与 $\sigma$/$2\sigma$（式 16–18）；Sinkhorn-Knopp 迭代由单核实现，反向核在片上重算中间结果并重走迭代（式 19）。除式 (14)–(15) 外，多数核用 TileLang 实现。应用侧包含 $F^{pre} \coloneqq H_l^{pre} x_l$ 与 $F^{post,res} \coloneqq H_l^{res} x_l + H_l^{post\top} F(\cdot, \cdot)$ 两核，后者把 $H_l^{res}$、$H_l^{post}$ 的施加与残差合并融合，读取元素从 $(3n+1)C$ 降到 $(n+1)C$，写入从 $3nC$ 降到 $nC$；无融合时 HC 每 token I/O 为读 $(5n+1)C + n^2 + 2n$、写 $(3n+1)C + n^2 + 2n$（Table 2），开销近似正比于 $n$。选择性重计算（§4.3.2）在前向后丢弃 mHC 核的中间激活，反向重跑 mHC 核而不重算昂贵的层函数 $F$；每 $L_r$ 层的块只持久保存块首输入 $x_{l_0}$（每 token $nC$ 元素），另逐层保存 $F(H_l^{pre} x_l, W_l)$ 输出（每 token $C$），其余为块内瞬态激活（Table 3）。总内存为驻留项 $nC \cdot L / L_r$ 加活跃块瞬态项 $(n+2)C \cdot L_r$，式 (20) 的目标关于 $L_r$ 求驻点 $-nCL/L_r^2 + (n+2)C = 0$，得 $L_r^* \approx \sqrt{nL/(n+2)}$；流水线并行要求重计算块不跨 stage 边界，且理论最优值通常与每个流水线 stage 的层数吻合，重计算边界据此与 stage 对齐。DualPipe 通信重叠（§4.3.3、Fig. 4）扩展流水线调度以覆盖 mHC 引入的两类开销：跨 stage 的 $n$ 流通信量与 stage 边界上 $L_r$ 层的重计算；MLP 的 $F^{post,res}$ 核放入专用高优先级计算流以免阻塞通信流；attention 层不用长驻 persistent kernel，使被重叠的计算可被抢占；各 stage 首层激活 $x_{l_0}$ 已在本地缓存，重计算与流水线通信解耦。

## 实验表现

实验设置（§5.1、Table 5）：采用 DeepSeek-V3 风格 MoE 语言模型，HC 与 mHC 均取 $n = 4$、门控因子初始化 $\alpha = 0.01$、Sinkhorn-Knopp 迭代 $t_{max} = 20$。主实验为 27B 模型（30 层、激活参数 4.14B、总参数 27.0B、262B tokens、50k 步）；另训 3B（39.3B tokens）与 9B（105B tokens）考察 compute scaling，并以 3B 固定 1.05T tokens 考察 token scaling。

**稳定性与损失（27B）**：HC 在约 12k 步出现一次损失骤升，与梯度范数不稳定高度相关（Fig. 2）；mHC 无此尖峰，最终训练损失比基线低 0.021，梯度范数曲线与基线相当（Fig. 5）[^src-mhc-manifold-constrained-hyper-connections]。传播中，HC 复合映射的 Amax Gain Magnitude 峰值近 3000（Fig. 3b），mHC 降至约 1.6，相差三个数量级（§5.4、Fig. 7）[^src-mhc-manifold-constrained-hyper-connections]。

**下游评测（27B，Table 4）**：

| Benchmark（指标，shot 数） | Baseline | HC | mHC |
|---|---|---|---|
| BBH（EM，3-shot） | 43.8 | 48.9 | 51.0 |
| DROP（F1，3-shot） | 47.0 | 51.6 | 53.9 |
| GSM8K（EM，8-shot） | 46.7 | 53.2 | 53.8 |
| HellaSwag（Acc.，10-shot） | 73.7 | 74.3 | 74.7 |
| MATH（EM，4-shot） | 22.0 | 26.4 | 26.0 |
| MMLU（Acc.，5-shot） | 59.0 | 63.0 | 63.4 |
| PIQA（Acc.，0-shot） | 78.5 | 79.9 | 80.5 |
| TriviaQA（EM，5-shot） | 54.3 | 56.3 | 57.6 |

mHC 在全部八项任务上均高于基线，并在多数任务上超过 HC；相对 HC 的主要增益为 BBH +2.1 与 DROP +2.3。例外是 MATH：mHC 26.0 略低于 HC 26.4，正文未作解释[^src-mhc-manifold-constrained-hyper-connections]。

**Scaling 与消融**：compute scaling 曲线（3B、9B、27B，各为 compute-optimal 配置）显示，相对基线的损失优势在更大计算预算下保持，仅轻微衰减（§5.3、Fig. 6a）；3B 固定语料的 token scaling 考察训练内动态，表明 mHC 在大规模场景有效（Fig. 6b，图内逐点数值文本层不可读）。HC 组件消融（Table 1）：仅 $H^{res}$ 时绝对损失差 −0.022，加 $H^{pre}$ 达 −0.025，三者齐全 −0.027，说明残差流内的信息交换是主要收益来源（§3）[^src-mhc-manifold-constrained-hyper-connections]。

## 在 DeepSeek-V4 中的落地

DeepSeek-V4 系列将 mHC 用于全部 Transformer 块：两模型均取 $n_{hc}=4$、Sinkhorn-Knopp 迭代 $t_{max}=20$；约束实现为 $A_l=\sigma(\tilde A_l)$、$C_l=2\sigma(\tilde C_l)$（式 6–7）；mHC 的静态偏置与门控因子由 AdamW 更新，其余模块走 Muon（§2.2、§2.4、§4.2.1）[^src-deepseek-v4-2606-19348]。工程侧与上节三项优化同构：融合 kernel、选择性重计算与 DualPipe 调度调整，将 mHC 的 wall-time 开销压至重叠 1F1B 流水线阶段的 6.7%（§3.4.2）[^src-deepseek-v4-2606-19348]，与 mHC 论文口径一致。$C_l$ 带因子 2 与 mHC 原文式 (8) 的 $H_l^{post} = 2\sigma(\tilde H_l^{post})$ 一致（2026-10-27 已对照原文）。

V4.1 把该实现改成 Single-Pass mHC：输入混合系数错开一个 block（式 6 用 $A_{l-1}$ 代替 $A_l$），输入混合不再等本层系数归约完成，作者报告性能退化可忽略；配合 Mega-mHC 单核把残余更新、输入混合与系数预测融成一次遍历，mHC 与 Single-Pass 的激活读写分别降到 $(3n+2)d$ 与理想下界 $(2n+2)d$（原三核实现为 $(4n+4)d$，两趟实现亦为 $(3n+2)d$）。预训练仍用多核实现，只有部署用 Mega-mHC（§2.4.1）[^src-deepseek-v4-1-2609-19969]。

## 相关概念

[[hyper-connections|Hyper-Connections (HC)]] 是 mHC 的前置方法，[[birkhoff-polytope|Birkhoff 多面体]] 是 mHC 使用的双随机矩阵流形，[[sinkhorn-algorithm|Sinkhorn-Knopp 算法]] 是实现 Birkhoff 投影的迭代归一化方法，[[identity-mapping-property|Identity Mapping Property]] 是 mHC 力图恢复的残差核心性质，[[residual-connections-as-diffusion|Residual Connections as Euler Steps of Reverse Diffusion]] 提供了残差连接的另一种宏观视角。
谱系后续：[[attnres|Attention Residuals]] 把（m）HC 的流间混合读作深度维线性注意力，改用深度维 softmax 注意力直接访问各层输出。

[^src-mhc-manifold-constrained-hyper-connections]: [[source-mhc-manifold-constrained-hyper-connections]]
[^src-deepseek-v4-2606-19348]: [[source-deepseek-v4-2606-19348]]
[^src-deepseek-v4-1-2609-19969]: [[source-deepseek-v4-1-2609-19969]]
