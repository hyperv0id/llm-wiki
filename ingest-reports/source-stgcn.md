# Ingest 报告：STGCN（Yu et al., IJCAI 2018）

## 创建
- **wiki/source-stgcn.md** — WHY：首篇纯卷积时空图交通预测论文，需建立 source-summary 以记录核心设计（谱域图卷积+GLU 门控时间卷积+三明治 ST-Conv Block）、实验性能（14× 训练加速、三数据集 SOTA）、局限（预定义图、单数据集）和后续影响链（→GWNet→DiffSTG→SpecSTG→UrbanDiT）
- **wiki/stgcn.md** — WHY：STGCN 是时空图神经网络（STGNN）演化之河的奠基节点，wiki 中 traffic-forecasting、diffstg、most 等多页面已多次以纯文本引用但无专属技术页，创建后可为全 wiki 提供统一的交叉引用锚点

## 修改
- **wiki/index.md** — WHY：添加 source-stgcn（Sources 类）和 stgcn（Techniques 类）两个新页面条目到目录索引
- **wiki/log.md** — WHY：追加 ingest 活动记录，包含创建/更新的页面列表
- **wiki/traffic-forecasting.md** — WHY：第 32 行纯文本 "STGCN (2018)" → `[[stgcn|STGCN (2018)]]`，作为里程碑模型列表中的第一个提及点
- **wiki/diffstg.md** — WHY：第 22 行纯文本 "STGCN" → `[[stgcn|STGCN]]`，DiffSTG 继承 STGCN 的门控卷积+时空块设计
- **wiki/spatio-temporal-foundation-model.md** — WHY：第 18 行纯文本 "STGCN" → `[[stgcn|STGCN]]`，作为 task-specific 模型的代表与 foundation model 对比
- **wiki/most.md** — WHY：第 45 行纯文本 "STGCN" → `[[stgcn|STGCN]]`，MoST zero-shot 超越 full-shot STGCN 的对比语境
- **wiki/mtgnn.md** — WHY：第 53 行纯文本 "STGCN" → `[[stgcn|STGCN]]`，MTGNN 在多步预测上与 STGCN 持平（无需预定义图）的对比
- **wiki/source-mtgnn.md** — WHY：第 41 行纯文本 "STGCN" → `[[stgcn|STGCN]]`，MTGNN 源文件摘要中的基线对比
- **wiki/hybrid-periodicity-decoupling.md** — WHY：第 38 行纯文本 "STGCN" → `[[stgcn|STGCN]]`，single-pathway 隐式周期建模的代表模型

## 新建交叉链接
- [[stgcn]] ↔ [[source-stgcn]] — 技术页与源文件摘要的必然配对
- [[stgcn]] ↔ [[traffic-forecasting]] — STGCN 是交通预测 Deep Graph-Based 方法的里程碑
- [[stgcn]] ↔ [[diffstg]] — DiffSTG 继承 STGCN 的 GLU 门控卷积 + 时空块设计思路
- [[stgcn]] ↔ [[specstg]] — SpecSTG 在谱域扩展 STGCN 的图卷积，从确定到概率的演进
- [[stgcn]] ↔ [[mtgnn]] — MTGNN 用自适应图学习解决 STGCN 预定义图的局限
- [[stgcn]] ↔ [[spatio-temporal-foundation-model]] — STGCN 作为 task-specific 模型与基础模型的范式对比
- [[stgcn]] ↔ [[most]] — MoST zero-shot 超越 full-shot STGCN 的性能对比
- [[stgcn]] ↔ [[urbandit]] — UrbanDiT 是 STGCN 演化河的远代终点（纯卷积→扩散→预训练→通用基础模型）
- [[stgcn]] ↔ [[hybrid-periodicity-decoupling]] — STGCN 的隐式周期建模 vs HyperD 的显式解耦
