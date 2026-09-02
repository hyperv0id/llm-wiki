---
title: "MultiSPANS: A Multi-range Spatial-Temporal Transformer Network for Traffic Forecast via Structural Entropy Optimization"
type: source-summary
tags:
  - traffic-forecasting
  - transformer
  - structural-entropy
  - attention-mask
  - spatio-temporal
created: 2026-09-02
last_updated: 2026-09-02
source_count: 1
confidence: medium
status: active
---

# MultiSPANS (WSDM 2024)

**作者**：Dongcheng Zou、Xuefeng Li、Hao Peng（北航）；Senzhang Wang（中南大学，通讯）；Yuandong Wang（清华）；Chunyang Liu、Kehua Sheng、Bo Zhang（滴滴）。**发表**：WSDM '24（2024-03，Merida, Mexico），DOI 10.1145/3616855.3635820。**代码**：github.com/SELGroup/MultiSPANS[^src-multispans]。

## 核心主张

论文针对时空 Transformer 的两个问题：离散时间点上的 token 信息不足以学习成对注意力并建模高阶全局时间性，以及图结构难以被 Transformer 直接利用——现有 GNN 输出融合或简单掩码/编码设计缺乏理论指导[^src-multispans]。受视觉 Transformer patching 启发，论文提出 MultiSPANS（Multi-range Spatiotemporal Prediction Attention Network with Structural entropy optimization）从多个范围捕获时空依赖：多滤波卷积模块（MFCL）生成信息更丰富的 ST-token，交错堆叠的时间/空间 Transformer 建模全局依赖，结构熵驱动的层次图感知机制把路网抽象为编码树并导出多层注意力掩码与层级相关分数[^src-multispans]。论文自述首次将结构熵理论用于优化空间注意力机制[^src-multispans]。

## 主要贡献（论文自述）

1. 提出 MultiSPANS 框架，实验报告在真实路网数据集上取得新 SOTA[^src-multispans]；
2. 提出可插拔的时空卷积模块，以高计算效率嵌入更长历史窗口[^src-multispans]；
3. 首次（论文自述）用结构熵理论优化空间注意力机制，挖掘路网层次结构[^src-multispans]。

## 实验

- 设置：PEMSD4/PEMSD8（流量/速度/占有率，5 分钟间隔），全通道输入、单通道输出构成 4 个子集；12 个基线分四类（VAR/SVR；AE/LSTM；TGCN/DCRNN/STGCN/MTGNN/GWNet；ASTGCN/STTN/GMAN），实现来自 LibCity[^src-multispans]。
- 主结果：论文报告相对 SOTA 平均提升 MAE 2.57%、MAPE 2.16%、RMSE 3.78%；最强项 PEMSD8-speed 为 MAE 1.36 / MAPE 2.84 / RMSE 3.26（提升 4.23%/3.73%/4.96%）；PEMSD4-flow MAE 19.07 / RMSE 30.46，PEMSD8-flow RMSE 23.87[^src-multispans]。
- 长窗口（表 2，PEMSD4-flow）：以 stride 1/3/4 把 12/36/48 步历史压成统一 12 长度隐藏态；48 步窗口 + 8 个时间滤波（尺寸 1–24）时 MAE 18.85 / MAPE 13.17 / RMSE 30.18，参数 332.3K、269.15s/epoch；同窗 STTN 为 MAE 19.31、699.8K 参数、931.18s/epoch，STGCN 为 MAE 20.97、1565.5K 参数[^src-multispans]。
- 消融（表 3，PEMSD4-flow，数字为去掉组件后的 MAE 恶化幅度）：MFCL 共 5.24%（时间滤波 1.98% > 空间滤波 1.49%）；多层掩码 2.33%；层级相关分数 1.52%；两者合计 4.55%；掩码换成 Infomap 层次社区检测只剩 1.83%，论文据此认为结构熵最小化更适合路网层次抽象[^src-multispans]。

## 局限

- 论文仅在 PEMSD4/8 两个数据集上评估，未报告更大规模路网或跨数据集迁移的实验[^src-multispans]。
- 编码树由给定路网图一次性导出，论文未讨论掩码与分数随时间的动态更新机制[^src-multispans]。
- 论文自述未来工作包括把结构熵引导的注意力机制推广到图与空间数据、并从层次网络分析视角分析 Transformer 可解释性[^src-multispans]。

[^src-multispans]: [[source-multispans]]
