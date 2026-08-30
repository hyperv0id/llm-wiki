---
title: "STGformer"
type: entity
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

# STGformer

**STGformer**（Wang et al., arXiv:2410.00385v2，2024-10；SUSTech–东京大学联合研究中心、东京大学、吉林大学、滴滴）是面向大规模交通预测的时空图 Transformer。论文将其定位为 GCN 与 Transformer 的折中：图卷积擅长局部高阶交互但感受野受限，时空 self-attention 能覆盖全局但无视显式图结构、计算随节点数平方增长；STGformer 用单层 [[stg-attention|STG-Attention]] 同时获得图结构信息与全局时空交互[^src-stgformer]。

## 机制要点

- **图传播**：SGC 式简化（去非线性、折叠逐层权重），但保留 0..k 阶输出 X_k = L^k X，供后续逐阶交互[^src-stgformer]。
- **统一时空注意力**：空间与时间视为统一实体，单组 Q/K/V 投影；空间注意力 softmax(QKᵀ/√C) 跨节点，时间注意力转置复用 softmax(QᵀK/√C) 节点内跨时间[^src-stgformer]。
- **线性化**：以分解内积替代 softmax，模块时间/内存复杂度 O(N+T)[^src-stgformer]。
- **递归门控交互**：各阶注意力输出按 p_{n+1} = a_n(q_n) ⊙ g_n(p_n) 从低阶到高阶经 1×1 卷积逐级交互[^src-stgformer]。
- **数据嵌入**：沿用 [[staeformer|STAEformer]] 的嵌入层——FC 投影 + 星期嵌入 + 日内嵌入 + 时空位置编码 Xste ∈ R^{N×T×d}，拼接为输入[^src-stgformer]。

## 论文报告的数字

- 效率：8600 节点 CA 图批量推理较 STAEformer 100× 加速、99.8% GPU 内存降幅（摘要）；FLOPs 比值约 0.00131、计算量降 99.869%（T=12、N=8600、d=32、K=3、|E|=201,363、L=3，Sec IV-D）；PEMS 设置计算成本约为 STAEformer 的 0.2%[^src-stgformer]。
- 参数量：SD 256K、BA 491K、LA 705K（STAEformer 对应 1.7M / 3.3M / 4.7M，Table I）[^src-stgformer]。
- 精度：LargeST SD/BA/LA（Table I）平均 MAE 均为对比模型最低，全部指标一致优于 STAEformer（论文自述 consistently outperforms STAEformer across all evaluated datasets）；SD 平均 MAE 17.36 vs 18.01（MAE −3.61%、RMSE −2.83%、MAPE −6.73%）；平均 RMSE/MAPE 个别格被 D2STGNN 微弱超过（SD 平均 RMSE 29.51 vs 29.52、BA 平均 MAPE 15.04% vs 15.22%）。PEMS03/04/07/08（Table III）4 数据集 × 3 指标的平均格全部最低；PEMS03 平均 RMSE 27.55→25.08（−8.97%），计算成本约为 STAEformer 的 0.2%[^src-stgformer]。
- 跨年泛化（Table II，2019 训练→2020 测试）：较 STAEformer 三个子集全部指标一致改进，SD horizon-3 RMSE 31.55→27.09（−14.14%）、BA 32.66→28.20（−13.66%）、LA 33.97→29.52（−13.10%）；平均 MAE 为三子集对比模型最低，平均 RMSE/MAPE 在部分子集被 GWNET/STID 超过（LA 平均 RMSE 40.85 vs 41.76、MAPE 22.51% vs 27.04%；BA STID RMSE 39.30 vs 39.73、MAPE 21.68% vs 22.54%；SD STID MAPE 26.88% vs 27.54%）；论文将稳健性归因于 STG-attention 单块结构与更少参数[^src-stgformer]。
- 消融（Fig 5，SD/BA）：去掉全部自注意力退化最重（模型退化为前馈模块），去掉图高阶交互次之；去掉空间或时间自注意力各自造成明显退化[^src-stgformer]。

## 范围注记

- 精度实验的 LargeST 部分为 SD/BA/LA 三个子集（Table I 无 CA 全图行）；8600 节点 CA 图仅用于效率测量（批量推理）与 FLOPs 计算[^src-stgformer]。
- 预测设置为 12 步输入 → 12 步输出（LargeST 15 分钟粒度、PEMS 5 分钟粒度），6:2:2 时序划分，masked RMSE/MAE/MAPE[^src-stgformer]。
- 数据嵌入的时空位置编码 Xste ∈ R^{N×T×d} 按节点索引（论文原文，沿用 STAEformer）[^src-stgformer]；若按 STAEformer 原设计为可学习自适应嵌入，则参数量含随 N 增长的成分（STAEformer 原文未 ingest、未在 raw/ 核实，此推断未在原文层验证）。论文的泛化证据为同图跨年（2019→2020），未评测跨网 zero-shot 迁移[^src-stgformer]。
- PDF 为 IEEE 期刊模板排版，未见接收 venue 信息；本页按 arXiv 预印本著录[^src-stgformer]。

## Related Pages

- [[stg-attention]] — STG-Attention 机制页
- [[staeformer]] — 效率与精度对照基线，嵌入层来源
- [[large-scale-spatial-temporal-graph]] — 大规模 ST 预测方法谱系
- [[spatio-temporal-ood-learning]] — 跨年泛化证据的归类位置
- [[traffic-forecasting]]

[^src-stgformer]: [[source-stgformer]]
