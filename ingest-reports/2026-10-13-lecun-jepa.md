# Ingest 报告：A Path Towards Autonomous Machine Intelligence (LeCun, v0.9.2 2022-06-27) + JEPA Focus on Slow Features (arXiv:2211.10831)

源文件均不在 `raw/`（该目录不可变，未写入）。证据源：
- LeCun 全文：v0.9.2 PDF，1641225 字节，md5 `253ede84d0f7bb1d2883e719e83aab8b`，pdftotext -layout 抽取 2780 行
- Sobal et al.：arXiv HTML v1（https://arxiv.org/abs/2211.10831）

## 创建
- wiki/jepa.md — WHY: JEPA 完整技术页；Sobal et al. 的固定噪声失效属 JEPA 自身局限，并入本页而非另建页
- wiki/hierarchical-jepa.md — WHY: H-JEPA 的层级堆叠与分层规划机制与单层 JEPA 不同，且是两条贡献的交汇点
- wiki/non-contrastive-ssl.md — WHY: 塌缩分型、对比/正则化路线、VICReg 与隐变量容量控制此前无承载页
- wiki/autonomous-agent-architecture.md — WHY: 六模块可微架构与 configurator 的硬件复用/知识共享论证此前无承载页
- wiki/intrinsic-cost-and-critic.md — WHY: IC/TC 二分、IC 不可变的理由、critic 训练式与行为规定的四种方式构成独立机制簇
- wiki/mode-1-mode-2-planning.md — WHY: Mode-1/Mode-2、七步 MPC、Mode-2→Mode-1 摊销与「推理即能量最小化」是 actor 侧核心
- wiki/world-model.md — WHY: 职责界定、三类待解问题、七种不确定性、键值记忆与五种数据来源此前无承载页
- wiki/source-autonomous-machine-intelligence.md — WHY: 工作流要求的源摘要页，并记录证据源校验值
- wiki/source-jepa-focus-on-slow-features.md — WHY: 第二篇源文件的摘要页

## 修改
- wiki/energy-based-model.md — WHY: 原页只覆盖「概率 = 能量的归一化负指数」的狭义用法；广义定义（能量函数为基本对象）、隐变量 EBM 与塌缩分型属 EBM 概念本身（source_count 2→3，confidence low→medium）
- wiki/contrastive-learning.md — WHY: 补对比样本设计与维度灾难两条批评及 components/vectors 之别；并修正 source_count 基数（原 9 与正文实际引用不符，底部另有两条未被引用的孤立定义，按正文唯一引用数重算为 9）
- wiki/index.md / wiki/log.md — WHY: 工作流步骤 6–7

## 归因措辞
初稿沿用「论文自述/论文主张」逐句前缀，与 2026-10-08 MoBA 维护记录的约定不符。已重写为：机制直接陈述、靠脚注定位来源，实证数字靠「设置 + 图表/节号」定位，只有定位性与优劣比较的断言保留归因。

## 新建交叉链接
- [[jepa]] ↔ [[hierarchical-jepa]] ↔ [[non-contrastive-ssl]]
- [[jepa]] ↔ [[energy-based-model]] ↔ [[contrastive-learning]]
- [[world-model]] ↔ [[autonomous-agent-architecture]] ↔ [[mode-1-mode-2-planning]] ↔ [[intrinsic-cost-and-critic]]
- [[source-jepa-focus-on-slow-features]] ↔ [[jepa]] ↔ [[source-autonomous-machine-intelligence]]

## 未做
- 未建 JEPA 下游工作页（I-JEPA / V-JEPA / DINO-WM 等）：不在两份源文件范围内，不据外部知识补写
- 未建 LeCun、Sobal 等作者实体页：单源信息量不足
- 未改 [[videomae]] / [[mae]]：掩码自编码器归为对比方法只在 LeCun 的归类语境下成立，已在该页挂链，不反向改写其方法定性
- 未给 8.3 节立场性论断（规模不够 / reward 不够 / 不需要符号）单独立页：属论辩内容而非可复用机制
- [[contrastive-learning]] 的两条孤立脚注定义保留未删：属该页历史 ingest 遗留，需与其它页面一并决定
