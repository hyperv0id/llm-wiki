# Ingest 报告：graphcast-lam-2022

源文件：`raw/graphcast-lam-2022.pdf`（pdftotext 抽取全文阅读；未修改 raw/）。
版式核实：首页水印 arXiv:2212.12794v2 [cs.LG] 4 Aug 2023，全文无 Science 期刊排版标识（正文中出现的 "Science" 仅属参考文献著录，如 Science Robotics），Science 著录未在 PDF 内核实，按 arXiv v2 preprint 著录。

## 创建
- wiki/graphcast.md — WHY：GraphCast 为 wiki 谱系中被多处提及（8+ 页）但无专页的核心中期预报模型；以 PDF 为准记录 encode-process-decode GNN、multi-mesh、自回归 6h 步长、验证协议（ERA5/HRES-fc0、06z/18z 对齐、1380 目标）与论文自述局限（确定性/模糊化/分辨率差距）
- wiki/multi-mesh-representation.md — WHY：多分辨率 icosahedron mesh 是论文最具辨识度的机制（用户设计谱系中「多分辨率思想来源」），含 Table 4 规模数字、Grid2Mesh/Mesh2Grid 连接规则与 Supp 7.3.1 消融证据；StormInsight 明确复用该机制，需独立节点承接
- wiki/source-graphcast.md — WHY：每个 raw/ 文件对应一个 source-summary（340+ 汉字，含 venue 未核实口径）

## 修改
- wiki/precipitation-nowcasting.md — WHY：正文断言「GraphCast 等已在中程超越 NWP」，GraphCast 自身证据可直接支撑，补 [^src-graphcast] 内联引用与脚注定义（source_count 4→5）；补 [[graphcast]] 链接
- wiki/spherical-geometry-inductive-bias.md — WHY：GraphCast 条目的「多分辨率 mesh + 消息传递」属 GraphCast 论文直接支撑的事实，拆分归因（mesh 事实归 [^src-graphcast]、周期性定性保留 [^src-cirt]）；补相关页链接（source_count 1→2）
- wiki/storminsight.md — WHY：「Multi-mesh Message Passing（GraphCast 风格）」补 [[graphcast]]/[[multi-mesh-representation]] 链接（该条目整体归 [^src-storminsight]，不加新引注）
- wiki/circular-patching.md — WHY：对比表 Mesh 行补 [[multi-mesh-representation]]/[[graphcast]] 链接与相关页条目
- wiki/cirt.md — WHY：指标表与 vs 条目补链接与相关页条目（对比数字仍归 [^src-cirt]）
- wiki/subseasonal-to-seasonal-forecasting.md — WHY：两处 GraphCast 提及补链接与相关页条目
- wiki/uniextreme.md — WHY：正文与对比表补 [[graphcast]] 链接
- wiki/source-uniextreme.md — WHY：基线列举句补 [[graphcast]] 链接
- wiki/source-climatear.md — WHY：基线列举句补 [[graphcast]] 链接
- wiki/source-cirt.md — WHY：基线列举句补 [[graphcast]] 链接与相关页条目
- wiki/masked-generative-modeling.md — WHY：生成范式对比表补 [[graphcast]] 链接
- wiki/generative-time-series-forecasting.md — WHY：ClimateAR 段基线列举补 [[graphcast]] 链接
- wiki/weather-foundation-model.md — WHY：天气模型谱系枢纽页补 [[graphcast]] 相关页条目（纯导航链接，无新事实断言）

## 新建交叉链接
- [[graphcast]] ↔ [[multi-mesh-representation]]
- [[graphcast]] ↔ [[spherical-geometry-inductive-bias]]
- [[graphcast]] ↔ [[storminsight]]
- [[graphcast]] ↔ [[cirt]]
- [[graphcast]] ↔ [[circular-patching]]
- [[graphcast]] ↔ [[subseasonal-to-seasonal-forecasting]]
- [[graphcast]] ↔ [[precipitation-nowcasting]]
- [[graphcast]] ↔ [[weather-foundation-model]]
- [[graphcast]] ↔ [[uniextreme]]
- [[graphcast]] ↔ [[source-uniextreme]]、[[source-climatear]]、[[source-cirt]]、[[masked-generative-modeling]]、[[generative-time-series-forecasting]]

## 矛盾核对
grep 全 wiki 的 GraphCast 提及页逐页核对：既有论断（cirt.md「仅做局部消息传递」、circular-patching「无投影」、spherical-geometry「不显式编码空间周期性」、masked-generative-modeling「自回归逐 token 串行」等）均为他人论文的对比口径且与 PDF 内容不冲突，无矛盾触发，无 status 变更、无争议章节。

## 并行安全
未触碰 virtual-nodes-traffic/source-virtual-nodes/over-squashing/source-over-squashing/fully-adjacent-layer/graphgps/rwse/detr/object-queries/performer 等并行代理文件及 raw/；index.md 与 log.md 的追加在全部其他工作之后以 Read→Edit 完成；未执行任何 git add/commit/push。
