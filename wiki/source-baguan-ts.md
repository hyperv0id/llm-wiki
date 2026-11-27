---
title: "Baguan-TS: A Sequence-Native In-Context Learning Model for Time Series Forecasting with Covariates"
type: source-summary
tags:
  - time-series
  - foundation-model
  - in-context-learning
  - covariate-aware-forecasting
  - retrieval-augmented
  - icml-2026
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# Baguan-TS: A Sequence-Native In-Context Learning Model for Time Series Forecasting with Covariates

时序零样本预测的两大路线各有一块短板：表格化 ICL 方案（以 [[tabpfn-ts|TabPFN-TS]] 为代表）靠手工构造时间与季节特征把预测改写成表格回归，虽能免梯度适配，但丢掉原始波形结构且推理偏慢；端到端序列基础模型直接学习原始波形，却缺乏推理期免梯度的多样例上下文适应能力。Baguan-TS 的核心巧思是把两者合成一条序列原生路线：将目标历史、多元协变量与检索到的近邻示范直接编成一张四维网格，由三维 Transformer 沿时间、变量、上下文三轴依次做注意力。为了让这个高容量模型训练稳定、不把输出平均化为平滑均值，论文再配两个组件——目标空间近邻检索局部校准（RBfcst）与训练期上下文自检索辅助损失（context overfitting）。论文报告其含协变量公开基准点预测胜率居首、真实能源集四项指标均居首[^src-baguan-ts]。

## 核心做法

- **归一化、目标空间近邻检索与四维张量编码**：对目标历史做实例级 Z-score 归一化，在目标值空间检索若干条历史–未来成对近邻片段充当上下文示范（论文称 RBfcst，即特征无关的目标空间检索式局部校准）；查询与示范统一切分时间 patch，协变量 patch 经随机傅里叶特征映射，叠加未来掩码指示项后拼成「上下文 × 时间 × 变量」四维张量[^src-baguan-ts]。
- **三轴分解注意力主干**：不做全网格联合注意力，而是逐层依次沿时间轴（旋转位置编码）、变量轴（可学习变量嵌入对齐目标与协变量）、上下文轴（无位置编码以保持示范置换对称）执行单轴多头自注意力，把计算量从三轴乘积降为三轴之和[^src-baguan-ts]。
- **训练期上下文过拟合目标与分箱概率输出**：训练时把一条上下文片段原样复制进查询位置并监督自检索重建（论文称 context overfitting），逼模型真正读取示范而非输出平稳均值；输出端经 MLP 映射为离散分箱 softmax 概率分布，推理时另把时间轴塌缩为逐点模式，与完整三维模式集成出概率预测[^src-baguan-ts]。

## 证据

1. **定位与主结论**：论文报告 Baguan-TS 点预测胜率居首，归一化分位数损失 SQL（Scaled Quantile Loss）较同属免梯度 ICL 路线的 TabPFN-TS 降低约 4.8%，27 个自有能源数据集四项指标均居首；加权分位数损失 WQL（Weighted Quantile Loss）概率校准不及 TabPFN-TS[^src-baguan-ts]。
2. **真实涨点归因**：拿掉 RBfcst 局部检索校准掉点最狠，但换协变量特征空间检索亦能拿回大半收益——涨点来自「近邻示范局部校准」本身；2D 模式单独最弱，2D/3D 集成再添增益[^src-baguan-ts]。
3. **证据缺口与负结果**：仅支持单目标加多元协变量；防过平滑策略缺下游指标消融，3D 结构无逐轴消融，论文未设 Limitations 章节[^src-baguan-ts]。

## 相关页面

- [[baguan-ts]] — 机制拆解页：将本页算子配方展开为分阶段维度变换与跨模型对照矩阵
- [[tabpfn-ts]] — 同槽位互斥：同为免梯度零样本 ICL，但以手工表格特征工程换取协变量接入与校准优势
- [[unified-covariate-adaptation]] / [[cora-tsfm]] — 同槽位互斥：冻结单变量骨干、下游训练适配器的协变量路线，与本文纯推理期 ICL 相对
- [[in-context-learning]] — 同构/特例：通用上下文学习范式在原始波形上的多示例特化实例
- [[retrieval-based-condition-augmentation]] — 跨槽位正交：同为检索机制，但 RCA 在预训练语料构造期过滤条件，RBfcst 在推理期做目标空间局部校准

---

**来源**：Linxiao Yang、Xue Jiang、Gezheng Xu、Tian Zhou 等 12 人（阿里达摩院 DAMO Academy，部分作者已转蚂蚁集团），ICML 2026。

[^src-baguan-ts]: [[source-baguan-ts]]
