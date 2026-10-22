---
title: "Motion MoE"
type: technique
tags:
  - mixture-of-experts
  - trajectory-prediction
  - routing
  - intersection
created: 2026-10-20
last_updated: 2026-10-20
source_count: 1
confidence: medium
status: active
---

# Motion MoE

**Motion MoE** 是 [[trajar|TrajAR]] 编码器中的专家层，把道路使用者的离散属性与连续运动趋势分给两类专家：类型/信号 **indicator experts** 按 one-hot 指标直接选中，运动趋势 **route experts** 用 top-$K_r$ 路由加权[^src-trajar]。论文把它定位为编码器的第二层，作用是「建模道路使用者的运动意图」，也是其贡献 2 的一部分[^src-trajar]。

## 结构

设运动趋势层输出为 $z^{f\nu}_t$（使用者 $\nu$ 在 $t$ 步的趋势表征），Motion MoE 的输出为残差叠加[^src-trajar]：

$$
z^\nu_t = z^{f\nu}_t + \sum_{n=1}^{N_a} I^a_n \mathrm{FFN}^{(a)}_n(z^{f\nu}_t) + \sum_{n=1}^{N_l} I^s_n \mathrm{FFN}^{(l)}_n(z^{f\nu}_t) + \sum_{n=1}^{8} g_n \mathrm{FFN}^{(r)}_n(z^{f\nu}_t)
$$

- **indicator experts（$N_a$ 个类型专家、$N_l$ 个信号专家）**：$I^a_n$、$I^s_n$ 是 one-hot 指示向量，$I^a_n$ 标记当前使用者类型所在位置，$I^s_n$ 标记当前信号状态所在位置。它们没有门控网络，本质是按属性查表选专家；实验里 $N_a$、$N_l$ 直接取该数据集唯一使用者类型数与唯一信号状态数[^src-trajar]。
- **route experts（8 个）**：门控值 $g_n = g'_n/\sum_{j=1}^{8}g'_j$，其中

$$
g'_n=\begin{cases}p_{n,t}, & p_{n,t}\in \mathrm{TopK}(\{p_{j,t}\}_{j=1}^{8},K_r)\\ 0, & \text{otherwise}\end{cases},\qquad p_{n,t}=\mathrm{Sigmoid}\big((z^{f\nu}_t)^\top c_n\big)
$$

$p_{n,t}$ 是论文所称的「trend-to-expert affinity」，$c_n$ 是第 $n$ 个 route expert 的质心向量，$K_r$ 是激活的 route expert 数[^src-trajar]。

论文给出的分工理由是：indicator experts 处理离散特征（使用者类型、信号状态），route experts 处理连续特征（运动趋势）[^src-trajar]。本页分析认为，与常规的 token/FFN 稀疏 MoE 相比，这里的专家不是在某个维度的不同投影或不同 patch 上分工，而是绑定「离散语义属性」与「连续运动模式」两种输入类型[^src-trajar]。

## 证据

**消融（表 3，SinD-Tianjin / SinD-Xi'an）**[^src-trajar]：

- 完整模型 0.409/0.798、0.509/0.942。
- 只把 Motion MoE 置 ×、其余模块保留的一行：0.892/1.642（SinD-Tianjin），相对完整模型 +118.1%/+105.8%，即正文对 Motion MoE 的归因；把 GAT & ESA（正文称 Interaction-Aware 层）置 × 的一行为 0.882/1.621（+115.6%/+103.1%）；把编码器整体换成全连接层为 1.231/2.542[^src-trajar]。
- 表 3 四列依次是 Motion MoE、GAT & ESA、Fully Connection、Decoder；完整表与逐行核算见 [[trajar]] 的实验结果节。

**专家级分析（图 3(b)(c)、图 4）**[^src-trajar]：

- 去掉用户类型专家后 ADE/FDE 为 0.613/1.197，去掉信号专家为 0.531/1.037，把 route 专家路由打乱（Shuffle Route）为 0.736/1.436，完整模型 0.409/0.798（SinD-Tianjin）。论文称打乱路由造成的退化最严重，据此支持 route experts 的必要性。
- 图 3(b) 按左转、右转、直行抽出 route expert 的负载比例，各转向的分布不同；论文将其读作「路由专门化」的证据。这是可视化对比，没有给定量差异或显著性。
- 图 4 扫「Top value of MoE」（$K_r$）与「route expert 数」，论文称 8 个专家配合 top-$K_r$ 时 ADE 最低。

**逐数据集超参数（实现设置）**：按验证集选出的 $\{K_r, l_d, N_a, N_l\}$ 在 SinD-Tianjin 为 {2,4,7,6}，SinD-Xian 为 {2,6,7,4}，InD-Bendplatz 为 {4,6,4,−}，InD-Frankenburg 为 {2,8,4,−}[^src-trajar]。InD 数据集不含信号信息时省略信号专家，以免干扰[^src-trajar]。

## 边界

- 只在路口轨迹预测的四个数据集上验证；图 3(c) 的三种破坏实验（去类型、去信号、打乱路由）只报告 SinD-Tianjin 一组设定[^src-trajar]。
- 论文没有给出专家负载均衡项（如 aux loss、capacity factor）的说明，也没有报告专家利用率、路由熵等定量指标[^src-trajar]。8 个 route expert 的数值来自图 4 的扫描结论，正文未给出扫描结果表[^src-trajar]。
- indicator experts 是查表式选择，参数量随类型/信号状态数线性增长；论文未讨论类型或信号状态组合很多时的参数与稀疏性问题[^src-trajar]。

## Related Pages

- [[trajar]] — 模型页
- [[source-trajar]] — 论文源摘要
- [[mixture-of-experts]] — MoE 通用机制（专家、门控、稠密/稀疏）
- [[moe-in-time-series-and-spatio-temporal]] — 时序/时空 MoE 的插入位置与门控谱系
- [[heterogeneous-moe-routing]] — FaST 的异质性感知路由（另一种「按输入属性分流」的做法）
- [[source-adaptraj]] — 同任务（多智能体轨迹预测）的多源域泛化工作，把每个源域的 extractor 当专家集合

[^src-trajar]: [[source-trajar]]
