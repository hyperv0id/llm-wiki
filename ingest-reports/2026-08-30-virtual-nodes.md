# Ingest 报告：virtual-nodes-zhuang-2025

- 源文件：`raw/virtual-nodes-zhuang-2025.pdf`（Virtual Nodes Improve Long-term Traffic Prediction，Dingyi Zhuang、Xiaoyang Cao、Jinhua Zhao、Shenhao Wang）
- 版式核实：PDF 首页水印 arXiv:2501.10048v1 [cs.LG] 17 Jan 2025；每页页脚 "A PREPRINT - JANUARY 20, 2025"；全文无 "Under review"/"Published as a conference paper" 等会议排版标识（ICLR/ICML 等字样仅见于参考文献著录）。结论：arXiv v1 预印本版式，venue 未在 PDF 内核实，页面按预印本著录。

## 创建

- `wiki/virtual-nodes-traffic.md` — WHY：论文核心机制页，承担「长期预测与 over-squashing 问题 → 虚拟节点 + semi-adaptive 邻接矩阵机制 → Table 2/Fig 5-7 实验证据 → 范围与论文自述局限」知识链；用户设计分析中 virtual node 设计点的 wiki 锚点。所有数字逐条注明章节/表号与「论文提出/作者报告」口径。
- `wiki/source-virtual-nodes.md` — WHY：每份 raw 文件对应一个 source-summary 页（402 汉字，300-500 区间内）。选择短名 `source-virtual-nodes`，与脚注 slug `[^src-virtual-nodes]` 一一对应；页面首节注明 raw 文件名与 venue 核实结果。

## 修改

- `wiki/traffic-forecasting.md` — WHY：方法版图新增 "Virtual Node Augmentation" 小节（置于 Large-Scale Linear-Complexity Modeling 之后），补 `[^src-virtual-nodes]` 脚注定义，source_count 52→54（旧值 52 已落后于实际引注数，更新后与正文 54 个唯一源一致；lint 审查修正）。
- `wiki/stgcn.md` — WHY：本文基座模型；「局限与后续演进」节后补一句基座用途与最优配置数字（含引注），关联页面列表加行，source_count 2→3，last_updated 2026-06-08→2026-08-30。
- `wiki/mtgnn.md` — WHY：本文 Aadapt 反对称公式（ReLU(E1E2ᵀ − E2E1ᵀ)）明确引用 Wu et al. 2020（即 MTGNN）的单向关系结论；「意义与局限」段补沿用关系（含引注），相关页面加行，source_count 2→3，last_updated 2026-05-30→2026-08-30。
- `wiki/graph-learning-layer.md` — WHY：MTGNN 图学习层是同族公式页，相关页面列表加行（结构性链接，无新事实断言），source_count 不变，last_updated→2026-08-30。
- `wiki/gwnet.md` — WHY：自适应邻接范式谱系表（Legacy）补 Virtual Nodes 行（含引注）+ Related Pages 加行，source_count 2→4（旧值 2 已落后于实际引注数，更新后与正文 4 个唯一源一致；lint 审查修正），last_updated 2026-06-09→2026-08-30。
- `wiki/large-scale-spatial-temporal-graph.md` — WHY：数据集基准节补本文对 LargeST SD 子集的使用配置（5 分钟采样、2017-2021，Table 1）与主结果数字（含引注），相关页面加行，source_count 7→8。
- `wiki/index.md` — WHY：登记 2 个新页面（Sources/Techniques 尾部追加块）。
- `wiki/log.md` — WHY：按仅追加原则记录本次 ingest。

## 未修改（并行安全 / 排查后判断）

- `graphgps.md`、`rwse.md`、`source-graphgps.md`、`detr.md`、`object-queries.md`、`source-detr.md`：并行代理未提交改动，按规则未触碰；新页面仅对 [[graphgps]] 建立单向出链。
- `over-smoothing-in-gnns.md`、`wire.md`、`quest-attention.md`、`topology-aware-graph-transformer.md`、`source-2509-22259.md`：GraphGPS 审查代理的未提交改动文件，未触碰；新页面仅建立指向 [[over-smoothing-in-gnns]] 的单向出链。
- `stgformer.md`、`stg-attention.md`、`source-stgformer.md`：stgformer 审查链未提交改动，未触碰；新页面仅建立指向 [[stgformer]] 的单向出链。
- `staeformer.md`：本文未使用 STAEformer，无论断交集，不加链。
- `pristi.md`：grep 证实其正文无 "virtual node" 字样（log.md 中该字样出自 PriSTI 的 ingest 记录文字，与本文无关），不加链。
- raw/ 未做任何写操作；未执行任何 git 写操作。

## 矛盾检查

- grep 证实 wiki 此前无任何虚拟节点页面与论断，无矛盾触发，无 status 变更、无争议章节。
- `large-scale-spatial-temporal-graph.md` 的 LargeST 表为 15min/2019 口径（原 LargeST 论文设置），本文以 5min/2017-2021 使用 SD 子集——同一基准的不同配置，非矛盾，已在页面中分立表述。
- `over-smoothing-in-gnns.md` 主题为 over-smoothing（层间表征趋同），与本文动机 over-squashing（长程信息压缩）机制不同，新页面已作区分表述，未改动该页。

## 数字复核（对照 PDF）

- Table 2 逐格核对：Semi-10 V.N. 在 H5/H10/H15/H20/Average/Average(75-100 min.) 全部 6 列 × RMSE/MAPE 共 12 格均为 12 种配置最低（H5 34.10/0.1346、H10 38.06/0.1558、H15 41.28/0.1709、H20 43.38/0.1750、Avg 37.82/0.1537、Avg75-100 42.32/0.1735），与论文 Sec 5.3 "consistently provides the lowest RMSE and MAPE" 自述一致。
- 45.15→42.32 对应 RMSE −6.27%、0.1827→0.1735 对应 MAPE −5.04%，与论文自述降幅一致（Sec 5.3）。
- SD 子集：716 节点、17,319 边、平均度 24.2、密度 0.0338、5 分钟采样、525,888 帧、0.38B 数据点（Table 1）；训练/测试 2019-2020 共 35,040 帧（Sec 5.3）；预测步长 1-20 horizon（每个 5 分钟）。
- 敏感性（Fig 6）：adaptive 全部不优于距离基线、semi 随 1→10 改善、10 最优、20 回落；可视化（Fig 7）：VN 3/8/10 连接最强、VN 8 高权重位于交叉口。均为论文自述归因。

## 新建交叉链接

- [[virtual-nodes-traffic]] ↔ [[source-virtual-nodes]]（机制页 ↔ 源摘要）
- [[virtual-nodes-traffic]] ↔ [[stgcn]]（基座模型）
- [[virtual-nodes-traffic]] ↔ [[mtgnn]]（Aadapt 反对称公式出处 Wu et al. 2020）
- [[virtual-nodes-traffic]] ↔ [[graph-learning-layer]]（同族图学习公式）
- [[virtual-nodes-traffic]] ↔ [[gwnet]]（自适应邻接范式谱系）
- [[virtual-nodes-traffic]] ↔ [[traffic-forecasting]]（方法版图小节）
- [[virtual-nodes-traffic]] ↔ [[large-scale-spatial-temporal-graph]]（LargeST SD 使用配置）
- [[virtual-nodes-traffic]] → [[over-squashing]]（动机问题；目标页面由并行 ingest 代理创建中，本代理未触碰。注：若并行代理最终命名不同（如 over-squashing-in-gnns），需同步修正 virtual-nodes-traffic.md 中 2 处链接）
- [[virtual-nodes-traffic]] → [[graphgps]]（over-squashing 的 global attention 缓解路线对照，单向出链）
- [[virtual-nodes-traffic]] → [[over-smoothing-in-gnns]]（相邻病理导航，单向出链）
- [[virtual-nodes-traffic]] → [[stgformer]]（单层长程建模路线对照，单向出链）
