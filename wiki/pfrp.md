---
title: "PFRP"
type: entity
tags:
  - time-series-forecasting
  - retrieval-augmented
  - univariate
  - plug-and-play
  - model-agnostic
  - aaai-2026
created: 2026-08-19
last_updated: 2026-08-19
source_count: 1
confidence: low
status: active
---

# PFRP

PFRP（Predicting the Future by Retrieving the Past）是一个模型无关的单变量时序预测增强框架，由 Du, Han & Guo（HKUST）发表于 AAAI 2026。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 动机

传统深度学习预测模型在训练后丢弃训练数据，推理时仅依赖回溯窗口内的局部上下文（local context），无法显式引用全局历史中的相似模式。PFRP 观察到时序中不同时期的子序列往往呈现高度相似的规律（如电力消耗数据中跨年的相似周模式），提出显式存储并检索这些全局历史模式以增强预测。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 核心组件

### Global Memory Bank (GMB)

GMB 存储历史样本的回溯窗口特征（作为 key）和对应预测区间序列（作为 value）。通过 [[predictive-contrastive-learning|PCL]] 训练 MLP 编码器后，对所有训练样本做 [[k-medoids-clustering|K-medoids]] 聚类，仅保留 K 个 medoid 样本。K-medoids 相比 K-means 的优势在于使用真实样本作为聚类中心，确保存储的模式是真实连贯的历史序列。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

### 检索与门控

推理时用编码器将当前回溯窗口编码为 query，与 GMB 中 K 个 key 计算余弦相似度取 top-k。引入两个门控机制：[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

- **Confidence Gate**：将 query 与每个 retrieved value 拼接为完整序列，经 MLP+sigmoid 输出存在概率 $p_i \in (0,1)$，调制原始 top-k 权重 $w^{(a_i)} \cdot p_i$ 后 Softmax 归一化。
- **Output Gate**：MLP 从当前回溯窗口 x 输出 scale α（初始化为全 1）和 shift β（初始化为全 0），对加权聚合的全局预测 $\bar{y}_1$ 做 $y_1 = \alpha \cdot \bar{y}_1 + \beta$ 变换。

### Dynamic Fusion

将全局预测 $y_1$ 与局部预测 $y_2$（来自任意预测模型）加权融合：$y = w_1 \cdot y_1 + w_2 \cdot y_2$，其中 $w_1, w_2$ 由调制后的 top-k 权重经 MLP+Softmax 生成。当历史中无高度相似序列时，top-k 权重趋小，模型自动降低对全局预测的依赖。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 效率

GMB 构建在 Electricity 上固定耗时 186 秒（PCL 134s + K-medoids 52s）。PFRP 仅增加 1.57 MB 模型参数，训练时间增加轻微。推理时仅需固定大小 GMB 检索，优于 RAFT 的全训练集遍历。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

## 与相关方法的关系

- **[[gtr|GTR]]**（ICLR 2026）：同为检索增强的即插即用模块，但 GTR 通过可学习全局周期参数矩阵按绝对时间位置检索周期模式，PFRP 通过特征相似度从 GMB 检索历史模式片段。GTR 面向多变量，PFRP 仅单变量。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]
- **[[pir|PIR]]**（NeurIPS 2025）：同为即插即用的检索增强预测框架，但 PIR 在后处理阶段以实例归一化编码 + 余弦相似度检索 top-K 相似实例做 softmax 加权修订，PFRP 在训练阶段即构建 GMB 并引入 PCL + 双门控 + 动态融合。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]
- **[[ratd|RATD]]**（NeurIPS 2024）：检索增强的时间序列扩散模型，检索 k-NN 参照引导去噪，PFRP 不依赖扩散模型，效率更高。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]
- **[[retrieval-augmented-spatio-temporal-forecasting|RAST]]**（AAAI 2026）：在时空双维度执行 FAISS 向量检索，面向交通预测；PFRP 仅在时间维度检索，面向单变量 TSF。[^src-predicting-the-future-by-retrieving-the-past-aaai2026]

[^src-predicting-the-future-by-retrieving-the-past-aaai2026]: [[source-predicting-the-future-by-retrieving-the-past-aaai2026]]
