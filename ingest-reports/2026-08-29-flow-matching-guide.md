# Ingest 报告：Flow Matching Guide and Code（arXiv:2412.06264）

日期：2026-08-29
raw：`raw/lipman-flow-matching-guide-arxiv-2024.pdf`（arXiv:2412.06264v1，水印 2024-12-09；扉页 Date: December 10, 2024；pdfinfo 核实 83 页：正文至第 75 页 + 参考文献/附录）。作者 Lipman、Havasi、Holderrieth、Shaul、Le、Karrer、Chen、Lopez-Paz、Ben-Hamu、Gat（FAIR at Meta / MIT CSAIL / Weizmann Institute）。配随 PyTorch 库 flow_matching。
定位：教程型长文。本 wiki 已有丰富的 FM 方法页（flow-matching、rectified-flow、interflow、meanflow、tsflow、loft 等），本次 ingest 的价值是**用指南的框架口径为现有页面补充归类与公式体系引用**，不为指南提到的每个方法建页。全程使用「指南认为/指南将…归类为/指南转述」归因口径；未用指南二手数字替换任何原文数字（指南本页未引入任何实验数字）。

## 创建

- wiki/source-flow-matching-guide.md — WHY：每个 raw 文件对应一份 source-summary；记录版本核实（v1 水印/扉页日期/83 页）、章节结构、作者自述定位与自述边界（"at the time of writing" 类论断、CFG 采样分布未知、10.6 节规则自称非正式等）。confidence medium（source_count 1）。
- wiki/flow-matching-design-space.md — WHY：指南最具复用价值的贡献是把 FM 组织为显式设计选择体系（路径/耦合/参数化/条件化/损失结构）。wiki 此前无框架归类页，该页作为 hub 挂接 rectified-flow（耦合轴）、interflow（双侧条件化）、x-prediction（参数化转换）、tsflow（multisample couplings）、classifier-free-guidance（CFG 速度形式）等具体方法页。引用 4 源（guide + FM 原文 + SI + RF），confidence high。
- wiki/generator-matching.md — WHY：指南第 8-9 章（CTMP/generator、universal characterization、GM 损失、组合模型、多模态）是指南相对单篇论文最独特的框架贡献，wiki 完全没有对应页面；diffusion/discrete diffusion/FM 的统一视角对后续 ingest 有归类价值。confidence medium（guide 为主要源，另挂 score-based-sde 于扩散联系处）。

## 修改

- wiki/flow-matching.md — WHY：指南将原论文"定理 1/2"重新命名为 Marginalization Trick（定理 3）并推广到一般条件化 $Z$ 与 Bregman 散度（命题 1）；补充线性 conditional OT 路径的 kinetic energy 上界刻画、三种条件化等价条件与插值不足警示、扩散=特例的精确口径。正文唯一源数 4→5，source_count 保持 5 与之对齐（ingest 时曾误记 5→6，审查修正）。
- wiki/rectified-flow.md — WHY：指南为"直线化"提供框架背景（单步 Euler 精确性；multisample couplings 为耦合侧拉直途径；指南未讨论 reflow，与其 rectification 重生成耦合的对照为 wiki 归类并已在页面注明）。source_count 5→6。
- wiki/interflow.md — WHY：指南把 stochastic interpolant 归为双侧条件化并给出插值不足的形式化警示（$C^2$ 反例），该警示与 SI 原文的额外条件互补，是 wiki 此前未记录的口径。source_count 1→2。
- wiki/stochastic-interpolant.md — WHY：同上（概念页侧的对应记录）。source_count 1→2。
- wiki/tsflow.md — WHY：其 mini-batch OT 耦合在指南体系中对应 multisample couplings 类别，补框架归类引用。source_count 2→3。
- wiki/classifier-free-guidance.md — WHY：补 CFG 的 FM 速度场形式（指南式 4.93，转述 Zheng et al. 2023）与指南两点口径（CFG 采样分布未知；写作时点最流行）。source_count 9→10。
- wiki/probability-flow-ode.md — WHY：指南第 10 章给出 PF-ODE 的 FM/GM 归位（时间约定反转、ODE 采样等价、SDE 采样=叠加 divergence-free Langevin、"真"时间反转非必要）——与本页既有 SDE/FP 论述互补。source_count 3→4。
- wiki/x-prediction.md — WHY：指南 Table 1 的参数化转换体系（velocity/$x_1$/$x_0$/score）、可去奇点与 scheduler 等价/训练后变换，是该页 Li & He 流形论证之外的另一独立口径。source_count 1→2。
- wiki/loft.md — WHY：LOFT 动机 1（弯曲轨迹→多步积分）与指南框架论断（直路径误差更小、两条拉直途径）直接对应，补框架层面对照 callout；因 raw/ 无 LOFT PDF，明确标注"LOFT 是否引用指南未核实"。source_count 6→7。

## 新建交叉链接

- [[flow-matching]] ↔ [[flow-matching-design-space]]、[[generator-matching]]
- [[flow-matching-design-space]] ↔ [[rectified-flow]]、[[interflow]]、[[stochastic-interpolant]]、[[x-prediction]]、[[classifier-free-guidance]]、[[probability-flow-ode]]、[[tsflow]]、[[optimal-transport]]
- [[generator-matching]] ↔ [[diffusion-model]]、[[score-based-sde]]、[[generative-vector-field]]、[[flux-matching]]
- [[source-flow-matching-guide]] ← flow-matching、flow-matching-design-space、generator-matching、rectified-flow、interflow、stochastic-interpolant、tsflow、classifier-free-guidance、probability-flow-ode、x-prediction、loft

## 矛盾核对

未触发矛盾解决策略（无页面 status 变更）。两处"命名/口径差异"按分立归因处理、未按矛盾处理：

1. 定理编号与命名：flow-matching.md 正文的"定理 1/2"沿用 FM 原论文编号，指南将前者命名为 Marginalization Trick（定理 3）并把梯度等价推广为 Bregman 散度命题（命题 1/定理 4）——记为命名与推广关系，在 flow-matching.md 中明确对应关系。
2. 双侧插值充分性：stochastic-interpolant/interflow 页原口径（SI 的二次目标在温和可积条件下刻画速度场）与指南警示（单纯插值不足以保证边缘速度生成边缘路径、需 SI 原文的额外条件）不矛盾——SI 原文本身提供这些条件，指南是对条件强弱与可验证性的评价，按指南口径补记。

## 未做的事

- 未为指南提及的 Discrete FM、Riemannian FM、CTMC、多模态蛋白质生成等单立页面（指南转述他人工作，非指南独有贡献；如后续有对应原文入库再建页）。
- 未改 raw/；未执行任何 git 写操作；未动另一并行论文（guo 评测综述）的文件。
