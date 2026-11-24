# Ingest 报告：timebridge

- **分级**：Tier 1（ICML 2025, PMLR 267；在变量内与跨变量两个作用域正交解耦非平稳性的双重角色，提出「去趋势 $Q,K$ / 留平稳 $V$ + 压缩后跨变量非平稳注意力」范式，覆盖长/短时与金融三类任务，且被后续 ReNF、PESD-TSF 等直接复用其非对称注意力与时频加权 MAE 损失）

## 创建
- wiki/timebridge.md — WHY：TimeBridge 核心机制与维度流页，完整拆解两阶段作用域正交设计、标准算子还原、四阶段 `aligned` 张量维度流、与既有工作的有向关系及三行实验判词。
- wiki/source-timebridge.md — WHY：ICML 2025 原文精简摘要页，概括人话大义、去黑话配方、三行核心实证与证据缺口；未为内部模块 `Integrated Attention` / `Cointegrated Attention` 单独建 `source_count: 1` 页。

## 修改
- wiki/source-pesd-tsf.md — WHY：补写 PESD-TSF 的 `IntAttention` 直接继承自 TimeBridge 的瓶颈接力 $\Delta$ 关系（并刷新 `source_count: 1→2`）。
- wiki/source-renf.md — WHY：为既有的 TimeBridge 时频 MAE 损失复用与 Traffic $0.357$ vs $0.365$ / FLOPs 对照补上双向链接与跨槽位正交/实证复现对齐标注（并刷新 `source_count: 1→2`）。
- wiki/itransformer.md — WHY：补写与 TimeBridge 的同槽位互斥对照（单全局 variate token vs $M$ 压缩时间槽跨变量非平稳注意力，刷新 `source_count: 5→6`）。
- wiki/patchtst.md — WHY：补写从 PatchTST 到 TimeBridge 的瓶颈接力 $\Delta$（CI patch 骨架改写为非对称稳态 $Q/K$ + 压缩后跨变量协作，刷新 `source_count: 16→17`）。
- wiki/timemixer.md — WHY：补写与 TimeBridge 的同槽位互斥与复现口径对照（多尺度纯 MLP vs 分层非平稳注意力及固定 $L=720$ 对比，刷新 `source_count: 6→7`）。
- wiki/channel-independence.md — WHY：补写 TimeBridge「前段 CI 去趋势、后段压缩 CD 留平稳」的跨槽位正交阶段分工与顺序/作用域消融结论（刷新 `source_count: 20→21`）。
- wiki/instance-normalization.md — WHY：补写 TimeBridge 相对于序列级全局 RevIN 的特例演进对照（下沉至 patch 级且仅作用于变量内 $Q,K$，刷新 `source_count: 10→11`）。
- wiki/dataset-stationarity-estimation.md — WHY：补写数据集级检索控制平稳性度量与模型内部作用域分层处置非平稳性的跨槽位正交对照（刷新 `source_count: 1→2`）。
- wiki/index.md — WHY：按字母序在 Sources 与 Entities 段插入 `[[source-timebridge]]` 与 `[[timebridge]]`。
- wiki/log.md — WHY：末尾追加本次 Tier 1 ingest 日志条目。

## 新建交叉链接（标明关系类型）
- [[timebridge]] ↔ [[patchtst]]（瓶颈接力 $\Delta$：CI patch 骨架改写为非对称稳态 $Q,K$ 与跨变量协作）
- [[timebridge]] ↔ [[itransformer]]（同槽位互斥：单序列 variate token vs $M$ 压缩时间槽跨变量非平稳注意力）
- [[timebridge]] ↔ [[timemixer]]（同槽位互斥 / 复现口径对照：多尺度纯 MLP vs 分层非平稳注意力）
- [[timebridge]] ↔ [[channel-independence]]（跨槽位正交：仅变量内阶段保持 CI，压缩后切入跨变量非平稳注意力）
- [[timebridge]] ↔ [[instance-normalization]]（特例演进：全局 RevIN → patch 级仅作用于 $Q,K$，省去反归一化）
- [[timebridge]] ↔ [[dataset-stationarity-estimation]]（跨槽位正交：外部平稳性检索调度信号 vs 内部作用域分层处置非平稳）
- [[timebridge]] ↔ [[source-pesd-tsf]]（瓶颈接力 $\Delta$：PESD-TSF 直接继承 `IntAttention` 非对称去趋势设计并改为 GAP 适配超大规模）
- [[timebridge]] ↔ [[source-renf]]（跨槽位正交 / 实证复现对齐：ReNF 复用时频 MAE 损失并在 Traffic 上复现 MSE/FLOPs 对照）
