# Ingest 报告：hyper-connections-2409-19606

## 创建
- wiki/source-seed-2024-hyper-connections.md — WHY：原文（arXiv:2409.19606v3，ICLR 2025）此前未入库，[[hyper-connections]] 旧版全部引用实为 mHC 论文回顾章节的转述，需要一手 source-summary 承载。
- raw/hyper-connections-2409-19606.pdf — WHY：arXiv:2409.19606v3（ICLR 2025 排版版，2025-03-18），本次通读底本；`*.pdf` 按 .gitignore 不入 git。

## 修改
- wiki/hyper-connections.md — WHY：原页从 mHC 回顾转述 HC，记号与术语均为 mHC 口径；本次逐条对照原文改挂 `[^src-hyper-connections]`，按原文记号（$\mathbf{A_m}/\mathbf{B}/\mathbf{A_r}$、expansion rate $n$）重写，并补齐原文独有内容（DC/WC 分解、统一框架、序贯-并行对偶、初始化与输出缩放、开销表、全部实验表、可视化分析、边界）。原页的「规模化风险（复合映射增益约 3000）」为 mHC 论文的实验观察，从本页删除，仅保留在 [[manifold-constrained-hyper-connections]] 与 [[source-mhc-manifold-constrained-hyper-connections]]；原页「$H_l^{res}$ 贡献最大」的消融表述按原文 Table 3 口径重写（WC 不可训练的损失明显大于 B 不可训练）。
- wiki/index.md — WHY：Sources 段按字母序新增 source-seed-2024-hyper-connections；Techniques 段 hyper-connections 摘要按一手源更新。
- wiki/log.md — WHY：追加一条 ingest 记录。

## 新建交叉链接
- [[hyper-connections]] ↔ [[source-seed-2024-hyper-connections]]
- [[hyper-connections]] ↔ [[manifold-constrained-hyper-connections]]（链接原已存在，本次在「与 mHC 的关系」节互指并说明记号对应）

## 数值备注
- 任务提示与原文不符项（照实记录，未写入页面）：原文无 "depth multiplier $m$"（仅 expansion rate $n$）；无 "depth reduction" 或任何推理期加速章节；无分子/语音实验（附录仅视觉：DiT 生成 + ViT 分类）；无 scaling law 实验；$H^{pre}/H^{res}/H^{post}$ 为 mHC 记号，原文为 $\mathbf{A_m}/\mathbf{A_r}/\mathbf{B}$。
- §4 Implementation 输出模块权重初始化系数：pdftotext 版式断行导致歧义，经整页渲染核对为 "by a factor of $\sqrt{n}$"，页面照录原文，未按物理直觉改写。
- OLMo-1B-DHC×4 的 V3 Eval Loss：Table 1 为 2.514，Table 2/3 同配置为 2.515；两处并存，未替论文择一。
- Eq. 10 的 $\overline{\mathbf{H}}$ 上划线与 Eq. 11 的转置在 `-layout` 提取中丢失，经公式形状与子代理对 arXiv HTML v3 的交叉核对后转写；Table 1 数字另经页面渲染逐项复核。
