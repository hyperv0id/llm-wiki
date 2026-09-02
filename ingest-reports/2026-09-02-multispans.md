# Ingest 报告：MultiSPANS (WSDM 2024)

源文件：`raw/multispans-wsdm2024.pdf`（MultiSPANS: A Multi-range Spatial-Temporal Transformer Network for Traffic Forecast via Structural Entropy Optimization，WSDM '24，DOI 10.1145/3616855.3635820）

## 创建

- wiki/source-multispans.md — WHY：每个 raw/ 文件必须有对应的 source-summary 页。
- wiki/multispans.md — WHY：MultiSPANS 是可被后续源文件引用与对比的完整模型（type: technique），承接 [[pdformer]] 的注意力掩码路线；页面含完整机制（MFCL / 位置嵌入 / 编码树掩码 / 层级相关分数）与全部可核验实验数字（主实验 prose、表 2 长窗口、表 3 消融）。
- wiki/structural-entropy.md — WHY：结构熵/编码树（Li & Pan 2016 理论）是可复用的图理论概念，MultiSPANS 自述首次将其用于优化空间注意力；独立成页便于后续结构熵类源文件（SE-GSL、结构熵图池化、结构熵 RL 状态抽象等）挂接，也是 [[graph-node-clustering]] 拓扑驱动层次聚类的理论支撑。

## 修改

- wiki/pdformer.md — WHY：MultiSPANS 的 related work 直接概括 PDFormer 的地理/语义掩码设计；在「在后续工作与 wiki 中的位置」补该掩码路线的延续（结构熵多层掩码 + 相对结构熵位置编码），source_count 6→7。
- wiki/traffic-forecasting.md — WHY：Methods 索引页增设 Structural-Entropy-Guided Attention 小节（紧随 Transformer-Based），把 MultiSPANS 纳入方法路线总览，source_count 54→55。
- wiki/graph-node-clustering.md — WHY：编码树最小化即拓扑驱动的层次聚类；在「拓扑结构驱动」类补 deDoc 式结构熵聚类条目（与 Louvain/METIS 单层划分对比、Infomap 消融对照），source_count 12→13。
- wiki/source-stg-mamba.md — WHY：该页实验段落已有 MultiSPANS 基线裸文本，转为 wikilink 接入图谱。
- wiki/index.md — WHY：登记 3 个新页面（Sources / Entities / Concepts）。
- wiki/log.md — WHY：追加 ingest 记录。

## 新建交叉链接

- [[pdformer]] ↔ [[multispans]]（注意力掩码路线：二值地理/语义掩码 → 结构熵多层掩码）
- [[structural-entropy]] ↔ [[graph-node-clustering]]（编码树 = 拓扑驱动多粒度层次聚类）
- [[traffic-forecasting]] ↔ [[multispans]]（方法路线索引）
- [[source-stg-mamba]] ↔ [[multispans]]（基线对比）
