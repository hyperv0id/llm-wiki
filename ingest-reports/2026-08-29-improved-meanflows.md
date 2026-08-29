# Ingest 报告：Improved Mean Flows (iMF)

- 日期：2026-08-29
- 源文件：`raw/geng-improved-meanflows-arxiv-2025.pdf`
- 版本核实：PDF 首页水印 `arXiv:2512.02012v2 [cs.CV] 9 May 2026`，为 **v2**；题名页印作 *Improved Mean Flows: On the Challenges of Fastforward Generative Models*（主标题仍为 Improved Mean Flows，副标题即用户著录的改题内容）；PDF 内无 venue 信息
- 页面命名：按任务约定使用 `improved-meanflows`，页面标题注明完整题名；按 PDF v2 实际结构组织（两挑战 + 架构 + 实验），未套用任何 v1 框架

## 创建

- `wiki/improved-meanflows.md` — WHY：论文核心贡献（v-loss 再参数化、JVP 输入修正、CFG 条件化、in-context conditioning、1.72 FID）需要独立技术页，并被多个现有页面引用
- `wiki/source-improved-meanflows.md` — WHY：每个 raw/ 文件对应一份 source-summary（约 400 字，含版本核实记录）
- `ingest-reports/2026-08-29-improved-meanflows.md` — WHY：本报告

## 修改

- `wiki/meanflow.md` — WHY：本文是 MeanFlow 同团队后续工作，新增「后续工作：iMF 对原目标的修正」节（目标网络依赖 + CFG 固定两项修正、Fig. 3 训练动态观察与原论文"更稳定训练"表述的张力分别归因）；r=t 切片设计节补记 iMF 配置表 r≠t 占比 50% 与原 MF 25% 的差异（论文未讨论）；source_count 3→4。处理方式：同一团队后续改进，正文新增 + 互链，不设 superseded（论文未声明取代原 MF，Tab. 2 中两者数字并列报告）
- `wiki/alphaflow.md` — WHY：新增「iMF 的对照解释（本课程层面）」节——α-Flow 归因 L_TFM/L_TCc 梯度冲突，iMF 归因 JVP 切向量误用条件速度，两种机制解释并存、iMF 未回应 α-Flow 的冲突分析（iMF Sec. 2 仅概括性提及）；记录 iMF Tab. 3 转引 α-Flow-XL/2+ 2-NFE 1.95 与 α-Flow 原文自报 2.15（1.95 为均衡类采样设定）的口径差异；source_count 2→3
- `wiki/one-step-flow-generation.md` — WHY：对比表补 iMF 行；正文补 iMF 1-NFE FID 1.72（作者报告）；source_count 2→3
- `wiki/average-velocity-modeling.md` — WHY：「与 MeanFlow 的关系」节补 iMF 对目标的再参数化修正与 1.72 数字；source_count 3→4
- `wiki/consistency-models.md` — WHY：MeanFlow 定位节补 iMF 对 CM 系的 fastforward 概括（iMF Sec. 2）及 Tab. 3 中 iCT/iMF 对照数字；source_count 4→5
- `wiki/shortcut-models.md` — WHY：「vs MeanFlow」节补 iMF 对 Shortcut Models 的概括（两时刻与 midpoint 的关系，iMF Sec. 2）及 Tab. 3 数字；source_count 3→4
- `wiki/classifier-free-guidance.md` — WHY：应用节补 iMF 的引导尺度条件化（$\omega/\Omega$ 作条件变量，1-NFE 下可变引导，与原 MF 训练前固定 $\omega$ 对照）；source_count 8→9
- `wiki/flow-matching.md` — WHY：相关页面列表补 [[improved-meanflows]] 链接（纯结构性，无事实新增，source_count 不变）
- `wiki/index.md` — WHY：按仓库惯例在尾部追加 Sources/Techniques 条目
- `wiki/log.md` — WHY：追加 ingest 条目

## 新建交叉链接

- [[improved-meanflows]] ↔ [[meanflow]]（后续工作互链，双向）
- [[improved-meanflows]] ↔ [[alphaflow]]（两种机制解释对照）
- [[improved-meanflows]] ↔ [[one-step-flow-generation]]（一步生成全景表）
- [[improved-meanflows]] ↔ [[average-velocity-modeling]]（平均速度建模谱系）
- [[improved-meanflows]] ↔ [[consistency-models]]、[[shortcut-models]]（fastforward 路线概括）
- [[improved-meanflows]] ↔ [[classifier-free-guidance]]（引导尺度条件化）
- [[improved-meanflows]] ↔ [[flow-matching]]（相关页面链接）

## 论文口径核对说明

- 所有实验数字取自 PDF 并标注表/图/章节号；对比基线数字（iCT 34.24、Shortcut 10.60、α-Flow 2.58/1.95、FACM 1.76 等）均注明为 iMF 论文报告
- 等价改写（Eq. 9–10）标注为论文证明的论断；Fig. 3 训练动态标注具体设置（B/2、ℓ2、无加权、无 CFG、仅 t≠r 样本）；方差放大解释标注为论文的分析表述
- α-Flow 2-NFE 数字口径差异（1.95 vs 2.15/均衡类采样）双源分别归因，未替任何一方取舍
- 与 α-Flow 的对照明确标注「本课程层面」，与论文自述（"orthogonal to other concurrent improvements"）分开
- 未为 Decoupled MeanFlow / CMT / TiM 建页（仅在 iMF 页按论文 Sec. 2 口径提及，无独立引用页面）
