---
title: "分组超参搜索（Grouped Hyperparameter Search）"
type: technique
tags:
  - time-series-forecasting
  - hyperparameter-search
  - preprocessing
created: 2026-10-24
last_updated: 2026-10-24
source_count: 1
confidence: medium
status: active
---

# 分组超参搜索（Grouped Hyperparameter Search）

arXiv:2606.27282 提出的搜索组织方式，在"全数据集一套超参"与"逐 horizon 步 × 逐序列独立搜索"两极端间连续移动（§3.3）[^src-2606-27282-ltsf-linear-models]。动机：前作几乎总对每个 benchmark 固定一套预处理超参，假定了数据集内部同质性，但该假设在实践中很少成立；反向的全独立搜索又面临组合爆炸的预算问题[^src-2606-27282-ltsf-linear-models]。

## 机制

- **Horizon grouping $g_h$**：把 $H$ 个预测步切成宽 $g_h$ 的连续块，块内共享超参。$g_h = 1$ 即逐步调参；$g_h = H$ 即全局基线[^src-2606-27282-ltsf-linear-models]。
- **Series grouping $g_s$**：$C$ 个序列分成大小 $g_s$ 的组。$g_s = 1$ 即逐序列搜索；$g_s = C$ 即全共享[^src-2606-27282-ltsf-linear-models]。
- $(g_h, g_s)$ 定义 horizon × series 空间的搜索单元网格；每单元独立运行 Optuna 搜索（TPE 采样器），联合搜索 $L$、归一化 scope/method、$r$、增广类型与强度，$\alpha$ 由内层 21 点网格逐 trial 选出（§3.2–3.3）。每单元 20 个 trials（§4.1）[^src-2606-27282-ltsf-linear-models]。
- **CV**：每个 trial 用 $k$-fold expanding-window 交叉验证（chronological、保时序），主实验 $k=3$；最优超参才用于 held-out test，test 不参与搜索[^src-2606-27282-ltsf-linear-models]。

## 搜索结论（§5.2、Fig. 4；§5.5）

- **Horizon 分组几乎不敏感**：$g_h$ 从 1 到 48 的退化曲线平坦（$\leq 0.4\%$），最优在 $g_h = 48$（ETTm2、Electricity）或 $g_h = 6$（ETTh1、Weather）；论文默认取 $g_h = 48$ 以节省预算，转而搜索 $g_s$[^src-2606-27282-ltsf-linear-models]。
- **Series 分享程度是数据集属性**：
  - ETT 两 benchmark 全共享最好——per-series 搜索退化 0.5–4%，论文推测物理相关序列相互起到正则作用[^src-2606-27282-ltsf-linear-models]；
  - Weather 全 per-series 最好——退化随 $g_s$ 单调增长，$g_s = 21$（全共享）时达 10%，其 21 个通道（气压/湿度/风/降雨）量纲互不兼容[^src-2606-27282-ltsf-linear-models]；
  - Electricity 偏好中间粒度 $g_s = 32$[^src-2606-27282-ltsf-linear-models]。
- **分歧来源**（§5.5）：ETTh1 超参沿序列分布均匀、随 horizon 变化更多；Weather 的 local ratio 与 $\alpha$ 逐通道大幅发散。ETTm2 与 Exchange 居中，论文推测按测量类型分组为少数簇即可用少量成本捕获大部分异质性[^src-2606-27282-ltsf-linear-models]。
- **增广轴**（§5.6、Fig. 8）：增广在每个 benchmark 的 60–70% horizon 组被选中，时域/频域大致各半（时域略多）；最优 $\sigma$ 的离散度以 Weather、Exchange 最宽[^src-2606-27282-ltsf-linear-models]。

## 成本与边界

逐序列搜索成本随通道数线性增长，这是论文取 $g_h = 48$ 与中等 $g_s$ 的现实原因（Appendix D）[^src-2606-27282-ltsf-linear-models]。结论限于 Ridge + 搜索式预处理这一测试床（论文自述，Appendix D）[^src-2606-27282-ltsf-linear-models]。

## 关联页面

- [[searchcast]] — 使用该方案的管线与主结果
- [[trailing-fraction-local-normalization]] — 搜索空间中的归一化轴
- [[lookback-horizon-scaling]] — 搜索空间中的 context length 轴
- [[channel-independence]] — 逐序列建模与跨序列共享的谱系背景
- [[source-2606-27282-ltsf-linear-models]] — 源摘要

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
