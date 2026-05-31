# Ingest 报告：Pyraformer (Liu et al., ICLR 2022 Oral)

## 创建
- wiki/source-pyraformer.md — WHY：ICLR 2022 Oral 论文（~934 citations），首创金字塔注意力机制，首次在理论上同时达成 O(L) 复杂度和 O(1) 最大路径长度，需 source-summary 页面记录核心方法、实验结果和局限性
- wiki/pyraformer.md — WHY：Pyraformer 是 LSTF 演化链中效率与结构并重的关键节点，代表多分辨率图结构建模路线（区别于 Autoformer 的分解/Fourier 和 FEDformer 的频率域路线），需独立实体页面记录架构、性能、消融发现和历史地位

## 修改
- （无）——Pyraformer 引用 Informer 为其直接对比基线，但在 Informer 实体页面中未反向提及 Pyraformer。Informer 页面已含全面历史意义章节，暂不更新以避免重复链过长

## 新建交叉链接
- [[pyraformer]] ↔ [[informer]] （直接对比基线，MSE -24.8% ~ -28.9%）
- [[pyraformer]] ↔ [[lstf]] （Pyraformer 是 LSTF 演化链中的关键模型）
- [[pyraformer]] ↔ [[autoformer]] （同期 Transformer 创新，不同技术路线）
- [[pyraformer]] ↔ [[fedformer]] （同达 O(L) 但实现途径不同）
- [[pyraformer]] ↔ [[patchtst]] （不同序列压缩思路：金字塔树 vs patch tokenization）
- [[pyraformer]] ↔ [[itransformer]] （正交的维度处理思路）
- [[pyraformer]] ↔ [[ltsf-linear]] （批判视角：简单线性基线挑战 Transformer 有效性）
