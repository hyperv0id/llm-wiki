---
title: "Consistency-FM"
type: entity
tags:
  - flow-matching
  - consistency-models
  - velocity-consistency
  - few-step-generation
  - arxiv-2024
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# Consistency-FM (Consistency Flow Matching)

**Consistency-FM** 是 Yang 等 9 人（北京大学、UT Austin 等）于 2024-07-02 提交 arXiv 的少步生成方法（v1，arXiv:2407.02398）：不依赖迭代 reflow 或 minibatch OT 配对，而是直接定义"从不同起始时刻指向同一终点"的直线流，通过速度场一致性约束训练，使少步欧拉积分即可近似 ODE 解[^src-yang-consistency-fm-arxiv24]。

> [!note] venue 状态（2026-08-29 录入时核查）
> arXiv v1 后未更新；OpenReview 存在提交记录（bS76qaGbel）但未见正式接收著录，社区文献仍以 arXiv 预印本引用。本页关于论文内容的论断均为论文自述口径。

## 问题与定位

FM 逐时刻监督向量场，学到的边缘速度场弯曲，采样需多步数值积分。论文将既有直线化路线的代价归纳为：OT-CFM 依赖 minibatch OT 配对，[[rectified-flow|Rectified Flow]] 依赖迭代 reflow 重采样；而 [[consistency-models|Consistency Models]] 的自洽映射作用在样本空间而非速度场[^src-yang-consistency-fm-arxiv24]。

## 机制

### 一致性等价（Lemma 1）

速度场沿 ODE 解轨迹恒定，当且仅当端点预测 $\gamma_x(t)+(1-t)v(t,\gamma_x(t))$ 对所有 $t$ 取值相同。该等价关系与 [[trajectory-consistency-flow-matching|LOFT Lemma 4.1]]（速度恒定 ⟺ 端点线性插值轨迹）是同一命题；LOFT 自述其 $L_{CT}$ 受本文启发[^src-loft]，两文表述可互相印证[^src-yang-consistency-fm-arxiv24]。

### 损失（Eq. 6）

$$L_\theta=\mathbb{E}\,\|f_\theta(t,x_t)-f_{\theta^-}(t+\Delta t,x_{t+\Delta t})\|^2+\alpha\,\|v_\theta(t,x_t)-v_{\theta^-}(t+\Delta t,x_{t+\Delta t})\|^2$$

- $f_\theta(t,x_t)=x_t+(1-t)v_\theta(t,x_t)$：第一项约束端点预测一致，第二项（权重 $\alpha$）约束两个时刻的速度一致[^src-yang-consistency-fm-arxiv24]
- 目标端以 EMA 参数 $\theta^-$ 稳定；$(x_t,x_{t+\Delta t})$ 取自预定义路径分布（论文举例 OT path、VP-SDE）[^src-yang-consistency-fm-arxiv24]
- **多段训练**：$[0,1]$ 均分 $K$ 段做 piecewise linear（段内权重 $\lambda^i$），动机是轨迹中段向量场更难训练（论文引 SD3/Esser et al. 的观察）；另附从预训练 FM 蒸馏的变体（Eq. 11）[^src-yang-consistency-fm-arxiv24]
- 理论支撑：Theorem 1（无 EMA 情形）论证目标在渐近意义下平衡速度估计精度与一致性约束；Theorem 2 + Corollary 2.1 给出段内误差分解，真速度在段内一致时可被恢复[^src-yang-consistency-fm-arxiv24]

## 实验结果（作者报告）

无条件图像生成，论文自述 preliminary：CIFAR-10 上 NFE=2 FID 5.34（Consistency Model 5.83、1-Rectified Flow 378，Table 2）；AFHQ-Cat 256 上 6 NFE FID 22.5（Rectified Flow 61.5、RF+Bellman Sampling 36.2，Table 3）；作者报告收敛比 Consistency Model 快 4.4 倍、比 Rectified Flow 快 1.7 倍（Fig 1）[^src-yang-consistency-fm-arxiv24]。

## 与 LOFT 的关系（实现差异）

[[loft|LOFT]]（KDD 2026）的 $L_{CT}$ 在结构上取自本文的速度一致性约束（同一线性插值路径上对齐 $t<s$ 两个时刻的速度、以较晚时刻为目标），但有四点差异，引用时不可混同[^src-yang-consistency-fm-arxiv24][^src-loft]：

| 维度 | Consistency-FM | LOFT $L_{CT}$ |
|------|----------------|----------------|
| 目标稳定化 | EMA 参数 $\theta^-$ | stop-gradient |
| 损失组成 | f 项 + $\alpha$ 加权速度项 | 仅速度项 |
| $\alpha$ 的含义 | 两项损失的固定权重标量 | 矫正混合系数（见 [[uncertainty-aware-rectification]]） |
| 多段线性化 | 有（$K$ 段 piecewise） | 论文未采用 |

另注：本文无时序/插补实验；LOFT Table 1 中 Consistency-FM 的插补数字（PeMS04 SC-TC，2/20 NFE 记 43.57/44.16）是 LOFT 自行适配复现的结果，非本文报告[^src-loft]。

## 相关页面

- [[source-yang-consistency-fm-arxiv24]] — 源文件摘要
- [[trajectory-consistency-flow-matching]] — LOFT 的速度一致性目标（本文为其方法来源）
- [[consistency-models]] — 样本空间自洽映射的先源工作
- [[rectified-flow]] — 迭代 reflow 直线化路线（本文动机中的对照）
- [[shortcut-models]] — 步长条件化自洽
- [[flow-matching]] — FM 基础框架
- [[loft]] — 将速度一致性引入时空插补的后续工作

[^src-yang-consistency-fm-arxiv24]: [[source-yang-consistency-fm-arxiv24]]
[^src-loft]: [[source-loft]]
