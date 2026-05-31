# Ingest 报告：source-gwnet

## 创建
- **wiki/source-gwnet.md** — WHY：GWNet 是 STGNN 谱系关键一环（STGCN→GWNet→DiffSTG→SpecSTG→UrbanDiT），首篇将自适应图学习引入时空图建模的论文，需建立 source-summary 记录其核心创新（自适应邻接矩阵、扩张因果卷积、非自回归输出）、实验性能（METR-LA/PEMS-BAY SOTA）、局限（静态图、无不确定性量化）和对后续模型的范式影响
- **wiki/gwnet.md** — WHY：GWNet 被 12 个已有页面以纯文本引用但无 wikilink，且作为自适应图学习范式的奠基工作，需要完整技术页面记录其架构细节、消融分析、训练配置和后续演进链

## 修改
- **wiki/stgcn.md** — WHY：STGCN 局限表行中"GWNet 自适应图学习"为纯文本，现已转为 wikilink
- **wiki/hybrid-periodicity-decoupling.md** — WHY："Single-pathway models" 列表中"GWNet"为纯文本，现已转为 wikilink
- **wiki/diffstg.md** — WHY："传统确定性 STGNN" 列表中的"GWNet"为纯文本，现已转为 wikilink
- **wiki/ragc.md** — WHY：节点嵌入过参数化表中"GWNet 占 72%"为纯文本，现已转为 wikilink
- **wiki/pristi.md** — WHY："Graph WaveNet 的图卷积"为纯文本，现已转为 wikilink
- **wiki/guided-layer-normalization.md** — WHY：GLN 扩展性段中"GWNet"为纯文本，现已转为 wikilink
- **wiki/specstg.md** — WHY："传统交通预测模型" 列表中"GWNet"为纯文本，现已转为 wikilink
- **wiki/dcrnn.md** — WHY：局限性分析段中"GWNet's adaptive adjacency matrix"为纯文本，现已转为 wikilink
- **wiki/node-embedding-regularization.md** — WHY：参数占比表中"GWNet"为纯文本，现已转为 wikilink
- **wiki/spatiotemporal-mirage.md** — WHY："GWNet（基线）" 为纯文本，现已转为 wikilink
- **wiki/source-2401-08119-specstg.md** — WHY："DCRNN, GWNet, STAEformer" 中 GWNet 为纯文本，现已转为 wikilink
- **wiki/source-2312-00516-std-mae.md** — WHY：两处"GWNet"为纯文本（主结果段和预测器无关性表），现已转为 wikilink
- **wiki/source-astgcn.md** — WHY：局限性段中"Graph WaveNet"为纯文本，现已转为 wikilink
- **wiki/index.md** — WHY：添加 [[source-gwnet]] 至 Sources 段，添加 [[gwnet]] 至 Entities 段
- **wiki/log.md** — WHY：记录本次 ingest 操作

## 新建交叉链接
- [[gwnet]] ↔ [[stgcn]] — STGCN 局限表直接指向 GWNet 解决方案
- [[gwnet]] ↔ [[dcrnn]] — DCRNN 局限性段和遗产表均已引用 GWNet
- [[gwnet]] ↔ [[mtgnn]] — 同团队工作，共同开创自适应图学习范式
- [[gwnet]] ↔ [[diffstg]] — DiffSTG 动机段将 GWNet 列为确定性基准
- [[gwnet]] ↔ [[specstg]] — SpecSTG 使用 Spectral Graph WaveNet 作为去噪网络
- [[gwnet]] ↔ [[ragc]] — RAGC 解决 GWNet 节点嵌入过参数化问题（72%→ECO）
- [[gwnet]] ↔ [[node-embedding-regularization]] — GWNet 是节点嵌入正则化的典型研究对象
- [[gwnet]] ↔ [[traffic-forecasting]] — GWNet 已被列为图方法里程碑
- [[gwnet]] ↔ [[pristi]] — PriSTI 采用 Graph WaveNet 的图卷积设计
- [[gwnet]] ↔ [[guided-layer-normalization]] — GLN 实验证明对 GWNet 的通用适用性
- [[gwnet]] ↔ [[std-mae]] — STD-MAE 以 GWNet 为预测器骨干之一
