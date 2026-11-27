---
title: "分组超参搜索（Grouped Hyperparameter Search）"
type: technique
tags:
  - time-series-forecasting
  - hyperparameter-search
  - preprocessing
created: 2026-10-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# 分组超参搜索（Grouped Hyperparameter Search）

以往的做法几乎总是给一个 benchmark 固定一套预处理超参，隐含假设是数据集内部同质；可实际上不同序列、不同预测长度的最优预处理可以差很远。走到另一极端——对每个预测步、每条序列都独立搜一遍——能把这种异质性纳入搜索，但预算按组合爆炸增长。分组搜索就是在这两个极端之间连续移动的一套组织方式[^src-2606-27282-ltsf-linear-models]。

## 机制

- **Horizon 分组 $g_h$**：把 $H$ 个预测步切成宽 $g_h$ 的连续块，块内共享超参。$g_h = 1$ 是逐步调参，$g_h = H$ 退回全局基线[^src-2606-27282-ltsf-linear-models]。
- **Series 分组 $g_s$**：$C$ 条序列分成大小为 $g_s$ 的组。$g_s = 1$ 是逐序列搜索，$g_s = C$ 是全部共享[^src-2606-27282-ltsf-linear-models]。
- 一对 $(g_h, g_s)$ 在 horizon × series 平面上划出一组搜索单元，每个单元独立跑 Optuna（树结构 Parzen 估计器，TPE 采样器），联合搜 $L$、归一化 scope/method、$r$、增广类型与强度；$\alpha$ 由 21 点内层网格逐 trial 选出，每单元 20 个 trials[^src-2606-27282-ltsf-linear-models]。
- **交叉验证**：每个 trial 用 $k$ 折 expanding-window 交叉验证，折按时间顺序构造以尊重时序，主实验取 $k = 3$；只有最优超参才用于 held-out test，test 不参与搜索[^src-2606-27282-ltsf-linear-models]。

## 搜索结论

- **horizon 分组几乎不敏感**：$g_h$ 从 1 到 48 的退化曲线很平，最优在 48 或 6。换 $g_h$ 几乎不改 MSE，但会改变学到的权重——这是分组的一个副作用，精度上无代价[^src-2606-27282-ltsf-linear-models]。
- **序列共享程度是数据集属性**：参与分组实验的两个 ETT 基准（ETTh1、ETTm2）均以全共享最好，per-series 搜索反而退化，论文推测物理相关的序列互相起正则作用；Weather 全 per-series 最好，退化随 $g_s$ 单调增长，因为它的 21 个通道（气压、湿度、风、降雨）量纲互不兼容；Electricity 偏好中间粒度[^src-2606-27282-ltsf-linear-models]。
- **分歧来自归一化与正则而非回看长度**：ETTh1 的超参沿序列分布均匀、随 horizon 变化更多；Weather 的 local ratio 与 $\alpha$ 逐通道大幅发散。ETTm2 与 Exchange 居中，按测量类型分成少数几簇大概就能以很小的成本捕获大部分异质性[^src-2606-27282-ltsf-linear-models]。

## 成本与边界

逐序列搜索的成本随通道数增长，这是论文取较宽的 horizon 分组与中等序列分组的现实原因；结论限于 Ridge 加搜索式预处理这个测试床（论文自述）[^src-2606-27282-ltsf-linear-models]。

## 关联页面

- [[searchcast]] — 跨槽位正交：采用该分组搜索组织方式的线性预测管线。
- [[trailing-fraction-local-normalization]] — 跨槽位正交：搜索空间中的归一化比例轴。
- [[lookback-horizon-scaling]] — 跨槽位正交：搜索空间中的回看长度轴。
- [[channel-independence]] — 退化特例：$g_s = 1$ 即退化为逐序列通道独立搜索，$g_s = C$ 为跨通道全共享。

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
