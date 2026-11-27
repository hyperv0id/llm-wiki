---
title: "How Good Can Linear Models Be for Time-Series Forecasting? (SearchCast)"
type: source-summary
tags:
  - time-series-forecasting
  - linear-model
  - hyperparameter-search
  - preprocessing
created: 2026-10-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# How Good Can Linear Models Be for Time-Series Forecasting?

长程预测默认"更大架构换更高精度"，线性模型被当作最弱对照。作者认为这印象来自评测惯例：所有模型固定同一套预处理，Transformer 能靠学到的参数部分吸收糟糕的回看窗或弱归一化，线性模型不能。Ridge 固定为模型后，预算全投向逐数据集、逐序列调预处理；这个无非线性的模型在八个 benchmark 的六个上追平或超过 PatchTST 等非线性基线[^src-2606-27282-ltsf-linear-models]。

## 核心做法

论文沿用既有结论：DLinear、NLinear 等线性变体在闭式岭回归下即可匹敌 SGD 训练版本，因此不再改造模型结构，自由度全给四条预处理轴。**回看长度**贡献最大：默认 $L=96$ 在两个相反方向同时出错，对平稳数据集少给一个数量级历史，对非平稳数据集又超量。**尾部局部归一化**只算窗口最近一小段的统计量，全窗口归一化几乎从不被选中；**增广**在时域或频域加噪；**分组共享粒度**指超参在 horizon 与序列两维共享到多细，最优粒度是数据集属性。搜索空间见 [[searchcast|SearchCast]]。

## 证据

1. **主结论**：线性类内 7/8 最优，对 PatchTST 等非线性基线 6/8 最优[^src-2606-27282-ltsf-linear-models]。
2. **涨点归因**：贡献最大的是回看长度——只扫 $L$ 时 ETT/Weather 误差降三到五成，ETTm2 中等长度饱和[^src-2606-27282-ltsf-linear-models]。
3. **证据缺口**：在 Electricity、Traffic 输给 PatchTST，且非线性基线未联合重调[^src-2606-27282-ltsf-linear-models]。

## 局限

论文自述的适用边界：结论限于标准数值长程 benchmark，回看长度标度、局部性与通道异质性只按该场景读；只报点估计 MSE，很小的差距应视为平手；逐序列搜索成本随通道数增长，故取较粗分组；把同一搜索用于更深模型属后续工作[^src-2606-27282-ltsf-linear-models]。

## 相关页面

- **派生页**：方法主结果 → [[searchcast|SearchCast]]，本页只留结论，机制与搜索网格落在该页。
- **派生页**：回看长度轴结论 → [[lookback-horizon-scaling]]，该页展开幂律规律与跨路线对照。
- **派生页**：归一化轴结论 → [[trailing-fraction-local-normalization]]，承载尾部比例 $r$ 的机制与选择频率。
- **派生页**：共享粒度轴机制 → [[grouped-hyperparameter-search]]，承载 $(g_h, g_s)$ 网格的组织方式。

---

arXiv:2606.27282v2（2026-06-29，Preprint）

[^src-2606-27282-ltsf-linear-models]: [[source-2606-27282-ltsf-linear-models]]
