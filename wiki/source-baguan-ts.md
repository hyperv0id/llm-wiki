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
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# Baguan-TS: A Sequence-Native In-Context Learning Model for Time Series Forecasting with Covariates

**来源**：Linxiao Yang、Xue Jiang、Gezheng Xu、Tian Zhou 等 12 人（阿里达摩院 DAMO Academy / 部分转蚂蚁集团），ICML 2026（PMLR 306，Seoul）；提出序列原生上下文学习（ICL）含协变量预测模型 **Baguan-TS**[^src-baguan-ts]。

面向协变量的零样本上下文学习模型（如 [[tabpfn-ts|TabPFN-TS]]）需要手工构造表格特征且推理较慢，而 [[timesfm|TimesFM]]、[[chronos|Chronos]] 等端到端序列基础模型虽直接读取原始波形，却缺乏推理期免梯度更新的多样例 ICL 能力[^src-baguan-ts]。为此，Baguan-TS 直接在序列原生四维网格上沿时间、变量、上下文三轴依次分解注意力，并配合目标空间近邻检索校准（RBfcst）与训练期自检索过拟合辅助损失，在不经下游微调的情况下实现含协变量零样本概率预测[^src-baguan-ts]。

## 算子配方与基线差分

剥离论文自定义模块术语后，Baguan-TS 由标准算子组成：先对目标历史做 $z$-score 归一化并在目标空间检索 $K_{\text{ctx}}=C$ 个近邻片段（Y-space RBfcst）；再对查询与上下文切分时间 patch，叠加随机傅里叶特征 $e=[\cos(Wq+b);\,\sin(Wq+b)]$ 与未来掩码指示项，拼出四维张量 $[C{+}1, S, M{+}1, D]$；主干按时间轴（带 RoPE）、变量轴（加可学习变量嵌入）、上下文轴（无位置编码）依次执行多头自注意力；训练期额外把一条上下文片段原样复制进 query 并加自检索辅助损失（context overfitting）以防止输出坍缩为平稳均值；输出端通过 MLP 映射到 $[-10,10]$ 的 5,000 个离散分箱 softmax 分布，从箱心求均值、从 CDF 取分位数，并在推理期将时间轴塌缩为 $S=1$ 得到 2D 模式与 3D 模式集成预测[^src-baguan-ts]。

相对 [[tabpfn-ts|TabPFN-TS]]，Baguan-TS 省去人工表格特征工程，单次前向约 $0.43\text{ s}$（约为 TabPFN-TS 延迟的 $1/10$），双模式集成耗时 $3.67\text{ s}$（显存 $2715\text{ MB}$）；相对 [[unified-covariate-adaptation|UniCA]] 等冻结骨干适配器及单变量序列模型 [[timesfm|TimesFM]]、[[chronos|Chronos]]、[[tirex|TiRex]]，则跳过下游适配训练而在预训练阶段原生支持多示例协变量 ICL[^src-baguan-ts]。

## 证据

1. **定位与主结论**：`fev-bench-cov` 点预测（SQL $0.7997$、MASE）领先 TabPFN-TS（$0.8404$）与 TimesFM-2.5（$0.8276$），27 能源集居首；概率校准 WQL 落后 TabPFN-TS（$0.1724$ vs $0.1643$）[^src-baguan-ts]。
2. **真实涨点归因**：去 RBfcst 掉点最明显（$0.3859\to 0.4284$），但 X-space 检索已达 $0.4005$，主涨点在局部校准而非 Y-space；2D/3D 集成亦有增益[^src-baguan-ts]。
3. **证据缺口与负结果**：仅单目标+多元协变量；context-overfitting 缺指标表且 3D 未逐轴消融；22.4M 参数未见核心切片正文，无 Limitations 节[^src-baguan-ts]。

## 相关页面

- [[baguan-ts]] — Baguan-TS 实体机制页与分阶段维度流
- [[tsfm-covariate-adaptation-comparison]] — 协变量适配与原生建模全景对比
- [[unified-covariate-adaptation]] · [[cora-tsfm]] — 冻结 TSFM 协变量适配器（同槽位互斥）
- [[tabpfn-ts]] · [[timesfm]] · [[chronos]] · [[tirex]] — 零样本与 `fev-bench-cov` 基线对照
- [[in-context-learning]] · [[retrieval-based-condition-augmentation]] — 概念与检索对照

[^src-baguan-ts]: [[source-baguan-ts]]
