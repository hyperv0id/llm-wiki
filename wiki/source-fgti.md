---
title: "Source: Frequency-aware Generative Models for Multivariate Time Series Imputation (FGTI, NeurIPS 2024)"
type: source-summary
tags:
  - diffusion-models
  - frequency-domain
  - data-imputation
  - time-series
  - neurips-2024
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# Source: FGTI（Frequency-aware Generative Models for MTS Imputation）

**著录**：Xinyu Yang, Yu Sun, Xiaojie Yuan（南开大学），Xinyang Chen（哈工大深圳）。NeurIPS 2024。raw 文件：`raw/yang-fgti-neurips-2024.pdf`。版式核实：每页页脚 "38th Conference on Neural Information Processing Systems (NeurIPS 2024)"、文末含 NeurIPS Paper Checklist，为官方 proceedings 排版，用户著录与 PDF 一致[^src-fgti]。代码：github.com/FGTI2024/FGTI24（参考文献 [1]）[^src-fgti]。

**核心论点**：时间序列插补误差主要来自残差项（STL 分解意义下），而残差项与高频分量相关、深度网络又难以建模高频，因此把频域信息显式加入生成式插补的条件（Figure 1 调查 + 附录 Table 7 分项实验支持该归因）[^src-fgti]。

**贡献**：提出 FGTI——high-frequency filter（截止频率以上分量，默认 F=0.3）与 dominant-frequency filter（幅值 top-κ，默认 κ=10）提取两组频域条件；Transformer 编码后经 time-frequency 与 attribute-frequency 两个 cross-attention 模块（Q、K 来自频域表示、V 来自时域表示）融入 DDPM 式去噪网络；命题 3.1 论证频域条件严格降低反向过程条件熵（附录 A.1）[^src-fgti]。

**证据**：作者报告在 KDD/Guangzhou/PhysioNet 的 MCAR 10-40% 设置下 RMSE/MAE 均优于 15 个基线（Table 1；如 KDD 10% RMSE 0.406 vs CSDI 0.459），CRPS 优于 TimeCIB/GAIN/CSDI/SSSD/PriSTI 五个概率生成式基线（Table 5），MAR/MNAR 补充实验结论一致（Figure 4、图 8-9），下游空气质量预测与死亡率预测最佳（Figure 6）；GRIN/PriSTI 用单位阵邻接矩阵，对比实验忽略真实缺失值（应用研究保留）[^src-fgti]。

**自述局限**：资源消耗略高于 CSDI（Sec 4.4，NeurIPS checklist 指向此处）；附录图 7 绘有 MoE 模块但正文未说明[^src-fgti]。论文同时引用 MTSI 综述（arXiv:2402.04059，其参考文献 [48]）[^src-fgti]。

[^src-fgti]: [[source-fgti]]
