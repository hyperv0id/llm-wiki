---
title: "Consistency Flow Matching: Defining Straight Flows with Velocity Consistency"
type: source-summary
tags:
  - flow-matching
  - consistency-models
  - velocity-consistency
  - few-step-generation
  - arxiv-2024
created: 2026-08-29
last_updated: 2026-08-29
source_count: 0
confidence: low
status: active
---

# Consistency-FM — Defining Straight Flows with Velocity Consistency

Yang 等 9 人（北京大学、UT Austin 等）于 2024-07-02 提交 arXiv 的方法论文（v1，arXiv:2407.02398），提出 Consistency-FM：在流匹配框架内显式施加速度场一致性约束，直接定义"从不同起始时刻指向同一终点"的直线流，使少步欧拉积分即可近似 ODE 解。论文将自身定位为 FM 的基础性改进，与 Rectified Flow 的迭代 reflow、OT-CFM 的 minibatch OT 配对属不同直线化路线。截至本页录入（2026-08-29），论文为 arXiv 预印本（v1 后未更新；OpenReview 存在提交记录 bS76qaGbel，未见正式接收著录，接收状态未核实）。

## 核心论点

- **等价关系（Lemma 1）**：速度场沿 ODE 解轨迹恒定，当且仅当端点预测 $\gamma_x(t)+(1-t)v(t,\gamma_x(t))$ 对所有 $t$ 相同——即从任意时刻出发均指向同一终点的直线流。
- **损失（Eq. 6）**：f 项（端点预测一致性）与 $\alpha$ 加权的速度项（同一路径上 $t$ 与 $t+\Delta t$ 两个时刻的速度对齐）之和；目标端以 EMA 参数 $\theta^-$ 稳定，$(x_t, x_{t+\Delta t})$ 取自预定义路径分布（论文举例 OT path、VP-SDE）。
- **理论**：Theorem 1（无 EMA 情形）论证该目标在渐近意义下平衡"速度估计精度"与"一致性约束"；Theorem 2 + Corollary 2.1 给出多段设定下的误差分解，真速度在段内一致时可被恢复。
- **多段训练**：将 $[0,1]$ 均分 $K$ 段做 piecewise linear（段内权重 $\lambda^i$），动机是轨迹中段向量场更难训练（论文引 SD3/Esser et al. 的观察）；另附从预训练 FM 蒸馏的变体（Eq. 11）。

## 证据（作者报告）

实验为无条件图像生成，论文自述 preliminary：CIFAR-10 上 NFE=2 FID 5.34（Consistency Model 5.83、1-Rectified Flow 378，Table 2）；AFHQ-Cat 256 上 6 NFE FID 22.5（Rectified Flow 61.5、RF+Bellman Sampling 36.2，Table 3）；作者报告收敛比 Consistency Model 快 4.4 倍、比 Rectified Flow 快 1.7 倍（Fig 1）。骨干为 DDPM++ U-Net，Euler 求解器，NFE ∈ {2, 6, 8}。

## 局限与范围

- 实验仅覆盖三个无条件图像数据集（CIFAR-10、AFHQ-Cat 256、CelebA-HQ 256），论文自述为 preliminary experiments；无时序、时空或插补任务。
- Future work 自述为 text-to-image 扩展与从扩散模型蒸馏。
- 预印本、接收状态未核实（截至 2026-08-29）。

## 相关页面

[[consistency-fm]] · [[trajectory-consistency-flow-matching]] · [[loft]] · [[consistency-models]] · [[rectified-flow]] · [[flow-matching]]
