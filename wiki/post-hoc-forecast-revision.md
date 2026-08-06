---
title: "Post-hoc Forecast Revision"
type: technique
tags:
  - time-series-forecasting
  - post-hoc-revision
  - error-estimation
  - local-context
  - global-retrieval
  - model-agnostic
created: 2026-08-06
last_updated: 2026-08-06
source_count: 2
confidence: medium
status: active
---

# Post-hoc Forecast Revision（预测后修订）

**Post-hoc forecast revision（预测后修订）** 指在骨干模型完成预测之后、用独立修订模块修正其输出的技术路线。[[pir|PIR]] 是这一路线的具体实例：不修改骨干，而是先估计逐实例误差识别失效样本，再结合局部与全局上下文修订，最后不确定性加权融合[^src-pir]。

## 流水线

给定输入 x、骨干中间预测 ȳ 与可用上下文 C，PIR 学习修订函数 Y = f_φ(X, ȳ, C)[^src-pir]：

1. **失败识别（Identify）**：不确定性估计器 δ = f_ue(x, ȳ, E) 以逐实例 MSE 为代理预测误差，误差越大的实例获得越大的修订权重（机制见 [[error-based-uncertainty-estimation]]）[^src-pir]。
2. **局部修订（Local Revising）**：逐变量投影 ȳ（CoVariateEmb）与外生信息（ExoVariateEmb，数值或文本）拼接为 H0 = [h_co, h_exo]，经带通道注意力的 Transformer 与线性头输出 y_local。论文的动机：协变量间的领先-滞后依赖，以及时间戳、节假日等可提前获得的外生先验，能缓解局部窗口内的分布突变；论文称该设计对 channel-independent 骨干尤其有益（[[channel-independence]]）[^src-pir]。
3. **全局修订（Global Revising）**：在仅由训练输入-目标对构成的检索库中，以实例归一化编码 + 余弦相似度检索 top-K 相似实例，Softmax 加权求和其目标作为 y_global——用"相似实例有相似未来"的先验覆盖长尾罕见模式（[[instance-level-variation]]）[^src-pir]。
4. **融合（Fusion）**：y_pred = ȳ + α·y_local + β·y_global；α = σ(Linear(δ)) 与 δ 正相关，β = σ(MLP(δ, w)) 同时利用估计误差与检索相似度；训练目标 L = L_pr + λ·L_ue（λ = 1）[^src-pir]。

## 设计要点

- **模型无关**：不依赖骨干内部结构，可插拔接入任意预测模型；检索库只依赖原始序列数据，论文称可扩展到多源数据集[^src-pir]。
- **估计-修订闭环**：识别结果直接决定修订强度，形成按实例自适应的后处理[^src-pir]。
- **局部与全局互补**：局部上下文应对突发分布变化（论文举例自然节律如节假日），全局上下文应对罕见数值模式[^src-pir]。

## 同族与邻近方法

- **[[prediction-refinement]]（TSDiff，NeurIPS 2023）**：同为预测后处理，机制不同——TSDiff 把扩散模型的隐式密度作为能量先验，在数据空间迭代精炼初值，属概率式、无需外生信息与检索库；PIR 是误差估计驱动的加权融合，输出点预测，需要协变量/外生信息与检索库[^src-pir]。
- **[[test-time-computing-st]]（ST-TTC，NeurIPS 2025）**：同样在冻结骨干后追加后处理模块，但 TTC 利用时序标签自相关在推理期在线校准；论文未描述 PIR 的推理期更新机制，按论文描述其随骨干离线训练（与预测任务联合）[^src-st-ttc]——机制不同，共享"不动骨干"的插件精神。
- **检索增强家族**：PIR 的全局修订是轻量非生成式实例检索（检索目标加权求和即修订项），区别于 [[gtr|GTR]]（可学习全局周期参数按绝对时间检索）、[[ratd|RATD]]（检索参照引导扩散去噪）、[[retrieval-guidance|Retrieval Guidance]]（检索信息分析性偏置得分函数）与 [[retrieval-augmented-spatio-temporal-forecasting|RAST]]（时空双维 FAISS 检索）[^src-pir]。

## 相关页面

- [[pir]] — 框架总览
- [[error-based-uncertainty-estimation]] — 识别组件
- [[instance-level-variation]] — 要解决的问题
- [[prediction-refinement]] · [[test-time-computing-st]] — 邻近后处理范式
- [[gtr]] · [[ratd]] · [[retrieval-guidance]] · [[retrieval-augmented-spatio-temporal-forecasting]] — 检索增强家族

[^src-pir]: [[source-pir]]
[^src-st-ttc]: [[source-st-ttc]]
