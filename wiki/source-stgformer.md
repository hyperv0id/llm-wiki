---
title: "STGformer: Efficient Spatiotemporal Graph Transformer for Traffic Forecasting (Wang et al., arXiv:2410.00385v2, 2024)"
type: source-summary
tags:
  - traffic-forecasting
  - graph-transformer
  - linear-attention
  - large-scale
created: 2026-08-30
last_updated: 2026-08-30
source_count: 1
confidence: medium
status: active
---

# STGformer: Efficient Spatiotemporal Graph Transformer for Traffic Forecasting

## 核心主张

论文提出面向大规模交通预测的时空图 Transformer，核心为 [[stg-attention|STG-Attention]] 模块：SGC 式图传播保留 0..k 阶输出，空间与时间共享同一组 Q/K/V 投影（时间注意力转置复用 QᵀK），以分解内积线性注意力替代 softmax，模块时间/内存复杂度 O(N+T)，各阶输出再经 1×1 卷积递归门控从低阶到高阶交互；全模型仅单层注意力块，替代 STAEformer 式堆叠 2L 层的时空可分离注意力[^src-stgformer]。数据嵌入沿用 STAEformer：FC 投影 + 星期/日内嵌入 + 时空位置编码 Xste[^src-stgformer]。

## 论文报告的实验

- 效率：8600 节点 CA 图批量推理较 STAEformer 100× 加速、99.8% GPU 内存降幅（仅见于摘要与引言贡献列表，正文实验章节未提供该测量的表/图与批量设置）；FLOPs 比值约 0.00131、计算量降 99.869%（T=12、N=8600、d=32、K=3、|E|=201,363、L=3 设置，Sec IV-D）[^src-stgformer]。
- 精度：LargeST SD/BA/LA（Table I）平均 MAE 均为对比模型最低、全部指标一致优于 STAEformer（论文自述 consistently outperforms STAEformer across all evaluated datasets）；共 8 格被对手微弱超过——SD h3 MAE 14.97 vs 14.92、h3 RMSE 24.96 vs 24.95、h6 RMSE 29.26 vs 29.24、平均 RMSE 29.52 vs 29.51，BA h3 MAPE 12.72% vs 12.12%、h6 MAPE 15.12% vs 14.89%、平均 MAPE 15.22% vs 15.04%（以上 7 格对手为 D2STGNN）；BA 平均 RMSE 33.50 vs 33.41（GWNET）。PEMS03/04/07/08（Table III）4 数据集 × 3 指标的平均格全部最低；PEMS03 平均 RMSE 27.55→25.08（−8.97%）；PEMS 设置计算成本约为 STAEformer 的 0.2%[^src-stgformer]。
- 跨年泛化（Table II，2019 训练→2020 测试）：较 STAEformer 三个子集全部指标一致改进（SD h3 RMSE 31.55→27.09、BA 32.66→28.20、LA 33.97→29.52，论文自述 −14.14%/−13.66%/−13.10%）；平均 MAE 为三子集对比模型最低，平均 RMSE/MAPE 被 GWNET（LA：40.85 vs 41.76、22.51% vs 27.04%）与 STID（BA：39.30 vs 39.73、21.68% vs 22.54%；SD MAPE：26.88% vs 27.54%）超过；论文将稳健性归因于 STG-attention 单块结构与更少参数[^src-stgformer]。
- 消融（Fig 5，SD/BA）：去掉全部自注意力退化最重（模型退化为前馈），去掉图高阶交互次之[^src-stgformer]。

## 范围与局限

精度实验的 LargeST 部分为 SD/BA/LA 三个子集，无 CA 全图行，8600 节点仅用于效率测量与 FLOPs 计算；LargeST 预测设置为 12→12 短窗。Xste 按节点索引，参数量含随 N 变化成分；泛化证据限于同图跨年，论文未评测跨网迁移。PDF 为 IEEE 期刊模板排版，未见接收 venue 信息，本页按 arXiv 预印本著录[^src-stgformer]。

源文件：raw/stgformer-nie-2024.pdf。文件名后缀 "nie" 与 PDF 作者栏不符：PDF 一作为 Hongjun Wang，通讯作者 Xuan Song；Tong Nie 为同领域另一位研究者，非本文作者。

[^src-stgformer]: [[source-stgformer]]
