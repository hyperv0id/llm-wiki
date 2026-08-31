---
title: "STG-Attention"
type: technique
tags:
  - attention
  - linear-attention
  - graph-transformer
  - traffic-forecasting
created: 2026-08-30
last_updated: 2026-08-31
source_count: 1
confidence: medium
status: active
---

# STG-Attention

**STG-Attention** 是 [[stgformer|STGformer]]（arXiv:2410.00385，2024）提出的时空注意力模块。论文用它在一层内同时完成图卷积式局部高阶交互与 Transformer 式全局交互，替代「空间注意力 × 时间注意力堆叠 2L 层」的分离式设计——论文以 STAEformer 为该堆叠路线的代表（Fig 1/4 对照）[^src-stgformer]。

## 问题

论文的出发点（Sec III 复杂度分析）是两条已有路线的互补短板[^src-stgformer]：

| 路线 | 复杂度 | 短板 |
|------|--------|------|
| 时空图卷积（Chebyshev 多项式） | O(K\|E\|C) | 感受野限于 K 跳局部 |
| 时空 self-attention | O(TN² + NT²) | 全局但平方开销，高阶交互靠堆层获得 |

## 机制（四步）

1. **图传播保留各阶**：[X_0 | X_1 | … | X_k] = GraphPropagation(X_emb)，X_k = L^k X。采用 SGC 式简化（去非线性、折叠权重），与 SGC 的区别是保留各阶输出用于后续交互[^src-stgformer]。
2. **统一 Q/K/V**：空间与时间视为统一实体，单组投影 Q = h·w_Q、K = h·w_K、V = h·w_V；空间注意力 softmax(QKᵀ/√C) 跨节点，时间注意力转置复用 softmax(QᵀK/√C) 在节点内跨时间[^src-stgformer]。
3. **线性化**：以分解内积替代 softmax——A_s = (1/n)(QKᵀ)V、A_t = (1/n)(QᵀK)V（论文称 efficient attention，引 Katharopoulos et al. 2020 等），模块时间/内存复杂度降为 O(N+T)[^src-stgformer]。
4. **递归门控交互**：各阶注意力输出从低阶到高阶逐级交互：p_{n+1} = a_n(q_n) ⊙ g_n(p_n)，其中 a_n 为上述时空注意力模块，g_n 在 n=0 时为恒等映射、否则为维度匹配线性层[^src-stgformer]。

## 复杂度

论文给出总 FLOPs：FLOPs(STGformer) = KC(|E| + N + T + NTC)，对比 STAEformer 的 L(TN²C + NT²C)；在 T=12、N=8600、d=32、K=3、|E|=201,363、L=3 设置下比值为 0.00131，即计算量降 99.869%（Sec IV-D）[^src-stgformer]。

## 论文报告的证据

- 端到端效率：8600 节点 CA 图批量推理较 STAEformer 100× 加速、99.8% GPU 内存降幅（摘要与引言贡献列表；正文实验章节未提供该测量的表/图与批量设置）[^src-stgformer]。
- 消融（Fig 5，SD/BA）：去掉全部自注意力退化最重（模型退化为前馈），去掉图高阶交互次之——两条支路均有贡献[^src-stgformer]。
- 精度与跨年结果：见 [[stgformer]] 页「论文报告的数字」。

## 谱系位置

- 本 wiki 归类：其线性化属于分解内积一族（论文引 Katharopoulos et al. 2020），与 [[linear-attention-unified-framework]] 记录的线性注意力循环形式同族；特点是把「空间×时间」作为统一实体共用一组 Q/K/V。
- 本 wiki 归类：与 [[adaptive-graph-agent-attention]]（FaST，node→agent→node 两段 attention）、[[query-aggregate-attention]]（STUNet，两段 query-aggregate）同属「避开 N² 配对交互」的交通空间混合路线；差异在于 STG-Attention 保留完整 QKᵀ 形式、靠线性化降复杂度，agent/query 路线则改变交互拓扑，[[patchstg|PatchSTG]] 则走固定掩码的结构化稀疏路线（见 [[patchstg-sparse-attention-form]]）。

[^src-stgformer]: [[source-stgformer]]
