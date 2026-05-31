# Ingest 报告：LogTrans

## 论文信息

- **标题**: Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting
- **作者**: Shiyang Li, Xiaoyong Jin, Wenhu Chen, Yao Xuan, Yu-Xiang Wang, Xiyou Zhou, Xifeng Yan (UCSB)
- **发表**: NeurIPS 2019
- **arXiv**: 1907.00235v3

## 创建

- wiki/source-logtrans.md — WHY：首篇将 Transformer 成功应用于时间序列预测的论文，核心贡献包括卷积自注意力（增强局部性）和 LogSparse 自注意力（突破 O(L²) 内存瓶颈），为后续 Informer/Autoformer/FEDformer/PatchTST 等 LSTF 模型提供了起点
- wiki/logtrans.md — WHY：LogTrans 是 Transformer-for-time-series 研究范式的开创实体。文章提出的卷积自注意力（k>1 因果卷积生成 query/key）和 LogSparse（指数间隔注意力 + 路径存在性定理）至今仍是时序 Transformer 设计空间中的关键组件
- wiki/logsparse-self-attention.md — WHY：LogSparse 自注意力是第一个有可证明信息流保证的结构化稀疏注意力机制，与 ProbSparse/Sparse Transformer/Reformer 并列构成高效注意力的技术谱系

## 修改

- wiki/logtrans.md — WHY：修正 source_count（1→2，实际引用 src-logtrans + src-zhou-informer-2021）；添加 [[logsparse-self-attention]] 交叉链接
- wiki/index.md — WHY：LogTrans source-summary 和 entity 页面未收录入索引
- wiki/log.md — WHY：Ingest 活动未记录

## 新建交叉链接

- [[logtrans]] ↔ [[logsparse-self-attention]]
- [[source-logtrans]] ↔ [[logtrans]]
- [[logtrans]] ↔ [[probsparse-self-attention]]（已有，通过区别论述）
- [[logtrans]] ↔ [[informer]]（已有，Informer 将 LogTrans 作为主要基线）
- [[logtrans]] ↔ [[lstf]]（已有，LogTrans 是该范式的首个 Transformer 模型）

## 核心贡献摘要

1. **卷积自注意力**：用 k>1 因果卷积替换标准 self-attention 中的 1×1 线性投影，使 query/key 携带局部上下文信息（形状匹配 vs 点值匹配），在 traffic-c 上 k=9 相比 k=1 降低 ~9% R₀.₅ 损失，训练收敛更快
2. **LogSparse 自注意力**：每层每 cell 仅关注 O(log L) 个指数间隔历史位置，堆叠 ⌊log₂L⌋+1 层保证全信息流通（定理 1），总内存 O(L(log L)²)。远距离 cell 对之间路径数呈超指数增长（O(log(l-j)! 条路径）
3. **Transformer 首次成功用于时序预测**：在合成数据上验证了 Transformer 的长期依赖建模能力（LSTM 随 t₀ 增大性能骤降，Transformer 保持准确）；在 7 个真实数据集上对比 ARIMA/ETS/TRMF/DeepAR/DeepState 取得最佳

## 与现有 Pages 的关系

- [[informer]] 将 LogTrans 作为主要基线，其 [[probsparse-self-attention]] 的 data-driven sparsity 是对 LogSparse 固定指数模式的改进
- [[ltsf-linear]] 挑战了 LogTrans 开启的 Transformer-for-LTSF 范式
- [[patchtst]] 在 LogTrans 基础上通过 patching + CI 证明 Transformer 可战胜线性模型
- [[autoformer]]、[[fedformer]] 均在 LogTrans 开创的路线图上发展
