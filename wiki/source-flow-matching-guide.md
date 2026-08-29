---
title: "Flow Matching Guide and Code"
type: source-summary
tags:
  - flow-matching
  - tutorial
  - generative-model
  - arxiv-2024
  - meta-ai
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# Flow Matching Guide and Code

**raw:** `raw/lipman-flow-matching-guide-arxiv-2024.pdf`（arXiv:2412.06264v1，水印 2024-12-09 [cs.LG]，扉页 Date: December 10, 2024；pdfinfo 核实全文 83 页：正文至第 75 页，其后为参考文献与附录）。作者 Yaron Lipman、Marton Havasi、Peter Holderrieth、Neta Shaul、Matt Le、Brian Karrer、Ricky T. Q. Chen、David Lopez-Paz、Heli Ben-Hamu、Itai Gat（FAIR at Meta、MIT CSAIL、Weizmann Institute of Science）。配随 PyTorch 库 flow_matching（github.com/facebookresearch/flow_matching），示例含 standalone FM、图像与文本生成[^src-flow-matching-guide]。

**作者自述定位**（Sec. 1）：双目标——(1) 作为 Flow Matching 的全面自足参考，梳理其设计选择与社区扩展；(2) 让新手快速采纳并在自己的应用上构建 FM。作者称 FM 已在图像、视频、语音、音频、蛋白质与机器人等大规模应用中推动 state-of-the-art（作者口径）[^src-flow-matching-guide]。

**章节结构**：第 2 章 standalone PyTorch quick tour（CondOT 路径 + CFM 损失的极简实现）；第 3 章 flow 模型数学基础（flow ODE、Continuity Equation、Instantaneous Change of Variables、simulation 训练）；第 4 章 FM 核心（coupling、条件路径、Marginalization Trick、Bregman 散度损失、conditional OT 路径的 kinetic energy 刻画、affine/Gaussian 路径、速度参数化转换与训练后 scheduler 变换、multisample couplings、guidance）；第 5 章 Riemannian FM（geodesic 条件流与 premetrics）；第 6-7 章 CTMC 与 Discrete Flow Matching（factorized 路径/速度、mixture 路径）；第 8-9 章 CTMP 与 Generator Matching（generator、Kolmogorov 方程、universal characterization、组合模型与多模态）；第 10 章与扩散及其他 denoising 模型的关系[^src-flow-matching-guide]。

**自述边界**：多处论断标注 "at the time of writing"（CFG 是最流行的条件训练做法、Gaussian 路径是最常用的 affine 路径类）；明言 CFG 精确采样的分布未知；第 10.6 节"denoising 模型 = GM 特例"的规则自称非正式、可能有例外；$x_1$/$x_0$-预测参数化在端点存在实践中待处理的奇点；紧致流形上 geodesic/premetric 条件流有奇点问题——均为指南自述[^src-flow-matching-guide]。

[^src-flow-matching-guide]: [[source-flow-matching-guide]]
