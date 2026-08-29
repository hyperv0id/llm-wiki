---
title: "Source: CoSTI - Consistency models for (a faster) spatio-temporal imputation"
type: source-summary
tags:
  - consistency-models
  - spatiotemporal-imputation
  - generative-models
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# Source: CoSTI (KBS 327, 2025, 114117)

- **raw 文件**：`raw/solis-garcia-costi-arxiv-2025.pdf`
- **著录**：Solís-García, Vega-Márquez, Nepomuceno, Nepomuceno-Chamorro（University of Seville）。Knowledge-Based Systems 327 (2025) 114117，DOI 10.1016/j.knosys.2025.114117。PDF 为 Elsevier/ScienceDirect 正式期刊排版（页眉含卷期号与 DOI），2025-03-31 投稿、2025-07-14 录用、2025-07-21 在线，CC BY 开放获取。用户著录与 PDF 核实一致。

## 核心论点

论文提出 CoSTI，将 Consistency Models（Song et al., ICML 2023）经 Consistency Training 适配到多变量时间序列插补（MTSI），以 1–2 步采样替代扩散插补模型 50–100 步迭代去噪；作者报告插补时间最多降低 98%，精度与 CSDI/PriSTI/TIMBA 相当（摘要、Sec. 6 口径）。论文自述这是 CM 首次用于 MTSI[^src-costi]。

## 贡献

方法侧（Sec. 3–4）：在 Karras 设计空间的标准 CT 之上，加入 PriSTI 式条件信息（线性插补 $\mathcal{X}_t$、邻接矩阵、掩码）、仅在真值位置的损失、dropout 0.2、AdamWScheduleFree 优化器与 $N$ 从 10 到 200 的线性课程；架构为双分支 U-Net（STFEM 扩展 PriSTI 条件特征提取模块；NEM 源自 CSDI，时间 transformer 换为双向 Mamba、保留 transformer 做 cross-attention）；100 次采样取中位数做确定性插补。实验侧（Sec. 5，作者报告）：6 个数据集、3 种缺失场景，Table 3 推理时间（如 AQI-36 0.005 h vs TIMBA 0.44 h），Table 4–6 精度（PhysioNet 2019 上最优，Pems08 最差），消融显示条件头贡献最大（Table 10）。Table 6 沿用 GRIN 基准（Cini et al.），五组设置中 MAE/MSE 均低于 GRIN；Table 4/5 中 CSDI/PriSTI/TIMBA 的数字转引自 TIMBA 论文，PhysioNet 2019 除外（作者自行运行）（Sec. 5.2）。

## 局限

论文自述（Sec. 6）：固定图结构；训练稳定性随数据集与初始化变化。另据 Table A.13，训练时间无一致优势（如 METR-LA 79.42 h vs CSDI 43.67 h），加速主张限于推理侧；部分设置 MAE 略高于扩散基线（Sec. 5.4.2 自述）。

[^src-costi]: [[source-costi]]
