# Ingest 报告：On the Bottleneck of Graph Neural Networks and its Practical Implications（over-squashing, Alon & Yahav）

日期：2026-08-30
raw 文件：`raw/over-squashing-alon-2020.pdf`（未修改）
抽取方式：`pdftotext` 全文阅读（/tmp/oversquash.txt，1617 行）

## 版式核实

- PDF 每页页眉「Published as a conference paper at ICLR 2021」——ICLR 2021 会议论文版式，venue 在 PDF 内核实。
- 首页水印「arXiv:2006.05205v4 [cs.LG] 9 Mar 2021」，与任务描述一致。
- 首页署名：Uri Alon & Eran Yahav, Technion, Israel。

## 创建

- `wiki/over-squashing.md` — WHY：本次 ingest 的核心概念页。over-squashing 此前在 wiki 中无任何条目（grep 证实零提及，仅 [[graphgps]] 正文提及一次该词），论文的定义、problem radius / 感受野指数增长机制、Tree-Neighbors-Match 证据、组合下界、与 under-reaching / over-smoothing 的区分全部需要落页。type: concept，source_count 2，confidence medium。
- `wiki/fully-adjacent-layer.md` — WHY：论文的缓解方案 FA 层（Sec 4.2 定义 + Appendix B 系统消融：2×d 仅 −5.5%、All-FA +1520%、Penultimate-FA −45.2%、partial-FA 比例-误差相关）内容量足以独立成链（问题 → 方案 → 证据 → 边界），与概念页分开承担知识链条。type: technique，source_count 1，confidence medium。注意：任务提示中的「directed 邻接」方案不在本 PDF 中——本 PDF 的方案是 fully-adjacent layer，已按 PDF 为准建页，未建 directed-邻接页面。
- `wiki/source-over-squashing.md` — WHY：每个 raw 文件一个 source-summary 的规定产物。347 汉字（300-500 区间内），机制、定理口径、实验数字全部标注章节/表号，「提出/证明」表述与论文实际（提出概念 + combinatorial counting argument，非定理-证明）一致。

## 修改

- `wiki/over-smoothing-in-gnns.md` — WHY：用户明确要求两概念页不得混写。新增「与 Over-Squashing 的区分」章节：区分两种现象、论文的假设口径（长程问题退化的解释是 over-squashing 而非 over-smoothing，Sec 6）与两个可独立发生的构造性例子（Appendix E）；补 [^src-over-squashing] 引注与脚注。source_count 3→4。未将原文任何 over-smoothing 论断降级或改为 superseded——论文不反驳 over-smoothing，只限定其证据范围，不构成矛盾解决策略的任何情形。
- `wiki/topology-aware-graph-transformer.md` — WHY：TGT 动机段声称局部聚合无法捕获长程依赖；补 over-squashing 作为该动机的容量侧论证（论文口径 + 引注），并以显式「wiki organizational note」标注 wiki 层面的路线对照（TGT 全局注意力分支与 FA 层同属绕过逐层局部聚合的路线，非任一论文声明）。新增 Related Pages 节。source_count 2→3。
- `wiki/index.md`、`wiki/log.md` — WHY：ingest 工作流第 6-7 步（最后一步追加，见并行安全说明）。

## 逐页判断后未修改（及理由）

- `wiki/graphgps.md` — 并行安全规则禁止改动（该页 Sec 2 相关句已提及 over-squashing 且引注 [^src-graphgps]，与本源一致）。遗留项：该页正文「over-squashing」为纯文本、未链 [[over-squashing]]，建议后续 lint 阶段在无并行冲突时补链接。
- `wiki/rwse.md` — 并行安全规则禁止改动；且 RWSE（结构编码）与 over-squashing（消息传递容量）无机制交集。
- `wiki/wire.md` — 无机制交集：WIRE 解决图位置编码（谱坐标旋转），不处理消息传递容量瓶颈；强行链接属无源推断。
- `wiki/grin.md` — GRIN 的源（source-2108-00298）不讨论 over-squashing；GRIN 虽为消息传递 GNN，但把 over-squashing 归给 GRIN 属无源推断，不加。
- `wiki/centralized-message-passing.md` — STOP 阻断节点间消息的动机是 OOD 鲁棒性而非容量瓶颈，源内无 over-squashing 论述。

## 新建交叉链接

- [[over-squashing]] ↔ [[over-smoothing-in-gnns]]（双向；两页均有显式区分章节，警告 callout 注明不可混写）
- [[over-squashing]] ↔ [[fully-adjacent-layer]]（瓶颈概念 ↔ 缓解方案）
- [[over-squashing]] → [[graphgps]]（GPS 把 over-squashing 列为 MPNN 限制之一；graphgps 侧反向链接因并行安全暂缺，见遗留项）
- [[over-squashing]] → [[topology-aware-graph-transformer]]（TGT 侧双向：Related Pages + 动机段）
- [[over-squashing]] / [[fully-adjacent-layer]] → [[source-over-squashing]]（脚注义务）
- [[source-over-squashing]] → [[over-squashing]]、[[fully-adjacent-layer]]（Related Pages）

## 与现有页面的矛盾核对

- grep 证实 ingest 前 wiki 无任何 over-squashing 页面与论断（log 中 "over-squashing" 字样仅为并行代理的并行安全声明）。
- [[over-smoothing-in-gnns]] 既有论断（低通滤波机制、缓解策略）与新源不冲突；新源只把 over-smoothing 的实证证据限定在短程任务并提出长程任务的另一解释（假设口径）。不触发矛盾解决策略情况 1/2/3，无 status 变更、无争议章节。
- [[graphgps]] 「MPNN 受 over-smoothing、over-squashing 和 1-WL 表达力限制」与本源一致，无矛盾。

## 并行安全说明

未触碰 detr/object-queries/source-detr/graphgps/rwse 及其未提交改动；未 git add/commit/push。wiki/index.md 与 wiki/log.md 的追加在全部页面工作完成后执行；期间 topology-aware-graph-transformer.md 的第一次 Edit 因内容锚不匹配（`\\alpha` 双反斜杠）失败一次，经字节级核实后重试成功，非并发冲突。
