---
title: "STID"
type: entity
tags:
  - time-series
  - traffic-forecasting
  - mlp
  - spatial-embedding
created: 2026-09-16
last_updated: 2026-09-16
source_count: 6
confidence: medium
status: active
---

# STID

**STID**（Spatial-Temporal IDentity，Shao et al. 2022）是交通预测中结构最简的一类基线：用可学习的空间身份与时间身份嵌入加一个轻量预测头，不含显式图传播[^src-adafre][^src-rstib]。本页汇总本 wiki 各源文件对 STID 的使用与报告数字，不代替原论文摘要。

## 在本 wiki 源文件中的角色

- **对照基线**：[[source-patchstg|PatchSTG]] 把 STID 归为 non-spatial 基线，与 GWNET/AGCRN/STGODE/RPMixer（静态空间）和 DSTAGNN/D2STGNN/DGCRN/STWave/BigST（动态空间）并列，并在 LargeST 上报告"仅靠空间可区分性（STID）就在 CA 这类大规模数据集上超过空间传播方法"，作者把原因归于消息传递中的过平滑比缺失空间交互危害更大[^src-patchstg]。
- **backbone**：[[adafre|AdaFre]] 选择 STID 作为频率分支的 backbone，论文自述理由是凸显频率自适应机制的增益、避开复杂 backbone 的混杂效应[^src-adafre]。
- **提示来源**：[[rstib-mlp|RSTIB-MLP]] 的可学习时空 prompt 是 STID 时空身份嵌入的扩展，包括静态空间 prompt、动态过渡 prompt、time-of-day 与 day-of-week prompt[^src-rstib]。
- **效率基准**：[[rstib-mlp|RSTIB-MLP]] 报告 PEMS04 上自身约 180 秒/epoch，比纯 MLP 基线 STID 的约 150 秒/epoch 略慢，即以 STID 作为速度下界[^src-rstib]。
- **稀疏/噪声场景下的简单基线**：[[source-st-ood|ST-OOD]] 的场景化评估报告 STID/MLP 配合轻度 dropout 在 OUT 场景常胜过复杂 STGNN[^src-st-ood]。

## 报告过的数字

| 数据/设置 | STID 结果 | 对照 | 来源 |
|-----------|-----------|------|------|
| PeMSD7，12→12 | MAE 19.61 | AdaFre 18.58，次优 STAEFormer 19.14 | [^src-adafre] |
| PeMSD4，效率 | 123K 参数 / 10 ms per iter / 15 s per epoch / 1679 MB | AdaFre 337K / 14ms / 23s / 2137MB | [^src-adafre] |
| TaxiBJ 人流 | 27.36 MAE | UniST 26.84 | [^src-unist] |
| Crowd（南京） | 3.85 MAE | UniST 3.00 | [^src-unist] |
| TrafficSH | 0.742 | UniST 0.665 | [^src-unist] |
| EvalST 长时传感器预测 | MAPE 19.9 | UrbanFM 17.0，D2STGNN 20.5 | [^src-urbanfm] |

这些数字来自不同论文各自的数据切分与实验协议，不能横向拼成统一排名。

## 关联页面

- [[adafre]] / [[source-adafre]] — 以 STID 为频率分支 backbone
- [[rstib-mlp]] — 扩展 STID 时空身份嵌入的鲁棒性变体
- [[source-patchstg]] — LargeST 上 STID 对空间传播方法的比较
- [[source-st-ood]] — STID/MLP 在分布外场景的稳健性
- [[source-unist]] — UniST 与 STID 的多场景对照
- [[source-urbanfm]] — EvalST 基准中的 STID
- [[staeformer]] — 同期 Transformer 基线
- [[traffic-forecasting]] — 交通预测总览

[^src-adafre]: [[source-adafre]]
[^src-rstib]: [[source-rstib-mlp]]
[^src-patchstg]: [[source-patchstg]]
[^src-st-ood]: [[source-st-ood]]
[^src-unist]: [[source-unist]]
[^src-urbanfm]: [[source-urbanfm]]
