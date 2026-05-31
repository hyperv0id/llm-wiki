# Ingest 报告：DCRNN（Li et al., ICLR 2018）

## 创建
- **wiki/source-dcrnn.md** — WHY：DCRNN 是时空图预测领域的奠基之作（ICLR 2018, 3,000+ citations），首次将扩散卷积（有向图双向随机游走）+ DCGRU + Seq2Seq + Scheduled Sampling 整合为端到端框架。wiki 中 traffic-forecasting、diffstg、specstg 等多页面已多次以纯文本引用但无专属页面，需建立 source-summary 和 entity 页面作为交叉引用锚点。
- **wiki/dcrnn.md** — WHY：DCRNN 是 STGNN 演化树的根节点之一（与同年 STGCN 并列，分别代表 RNN 系和 CNN 系两条技术路线）。现有的 diffstg、specstg、ragc 等页面都直接继承其扩散建模范式，创建后可作为全 wiki 的统一技术参考。

## 修改
- **wiki/index.md** — WHY：添加 source-dcrnn（Sources 类）和 dcrnn（Entities 类）两个新条目到目录索引
- **wiki/log.md** — WHY：追加 ingest 活动记录
- **wiki/traffic-forecasting.md** — WHY：第 32 行纯文本 "DCRNN (2018)" → `[[dcrnn|DCRNN]]`，作为 Deep Graph-Based 方法范式的开创者
- **wiki/diffstg.md** — WHY：第 22 行纯文本 "DCRNN" → `[[dcrnn|DCRNN]]`，DiffSTG 明确提及 DCRNN 作为确定性 STGNN 的代表
- **wiki/specstg.md** — WHY：第 25 行纯文本 "DCRNN" → `[[dcrnn|DCRNN]]`，SpecSTG 将 DCRNN 列为传统确定性模型的代表

## 新建交叉链接
- [[dcrnn]] ↔ [[source-dcrnn]] — entity 页与 source-summary 的必然配对
- [[dcrnn]] ↔ [[traffic-forecasting]] — DCRNN 是 Deep Graph-Based 范式的开创者
- [[dcrnn]] ↔ [[diffstg]] — DiffSTG 将扩散模型引入 STG，直接继承 DCRNN 的扩散卷积思想
- [[dcrnn]] ↔ [[specstg]] — SpecSTG 在谱域执行扩散，是 DCRNN 物理扩散→概率生成的技术演进
- [[dcrnn]] ↔ [[stgcn]] — 同年（2018）的两条并列技术路线：RNN 系（DCRNN 扩散卷积+DCGRU）vs CNN 系（STGCN 谱图卷积+GLU），共同奠定 STGNN 基础
