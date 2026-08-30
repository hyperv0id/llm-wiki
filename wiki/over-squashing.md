---
title: "Over-Squashing：GNN 消息传递的信息挤压瓶颈"
type: concept
tags:
  - graph-neural-network
  - message-passing
  - long-range
  - bottleneck
  - iclr-2021
created: 2026-08-30
last_updated: 2026-08-30
source_count: 2
confidence: medium
status: active
---

# Over-Squashing：GNN 消息传递的信息挤压瓶颈

Over-squashing（信息挤压）是 Alon & Yahav（ICLR 2021）提出的概念，用于解释 GNN 难以在远距离节点之间传播信息的现象：GNN 在沿长路径聚合消息时存在一个瓶颈，当层数足够覆盖长程交互时，每个节点的感受野（receptive field）随层数指数增长，而聚合操作必须把流入一个节点的全部信息压缩进固定长度的向量，于是指数增长的信息被「挤压」进固定尺寸的表示，来自远距离节点的消息无法有效到达目的地（Abstract, Sec 1, Sec 3）[^src-over-squashing]。论文自述其贡献是提出并论证这一现象，而非提出新的 GNN 变体（Sec 1）[^src-over-squashing]。

## 问题：长程交互与感受野的矛盾

论文把一个预测问题所要求的节点间交互范围称为问题半径（problem radius）$r$（Sec 1, Sec 3）[^src-over-squashing]。要让节点收到距离为 $r$ 的信息，GNN 至少需要 $K \geq r$ 层，否则这些远距离节点互不可知——论文把这一限制称为 under-reaching（Sec 1, Sec 4.2）[^src-over-squashing]。但层数增加时，感受野内的节点数指数增长：论文引用 Chen et al. (2018) 给出 $N_v^K = O(\exp(K))$（Sec 3）[^src-over-squashing]。结果是在式 (2)、(3) 的求和处，指数增长的信息被压进固定长度的向量，关键消息无法抵达远端，模型只学到训练数据中的短程信号，测试时可能泛化很差（Sec 3）[^src-over-squashing]。

## 与 RNN seq2seq 瓶颈的类比

论文把这个瓶颈类比为无注意力机制的传统 RNN seq2seq 模型的瓶颈：后者要把整条输入序列封装进固定尺寸向量（Sec 1, Fig 1）[^src-over-squashing]。区别在于增长率——RNN 的感受野随递归次数线性增长，而 GNN 的感受野指数增长，论文据此称 GNN 的瓶颈在渐近意义上更严重（Sec 1, Fig 1）[^src-over-squashing]。

## 与相邻概念的区分

### 与 under-reaching

Under-reaching 指固定层数 $K$ 下，信息沿边最远只能传播 $K$ 跳，节点无法感知 $K$ 跳以外的节点；论文把这一限制的讨论归因于 Barceló et al. (2020) 对 GNN 一阶逻辑表达力的分析（Sec 4.2, Sec 6）[^src-over-squashing]。论文区分了二者并称 over-squashing 的限制比 under-reaching 更紧：即使信息在 $K$ 跳内可达，也可能在传播途中被挤压（Sec 6）[^src-over-squashing]。QM9 实验被论文用作排除解释：数据集平均图直径 6.35±0.91、最大 10、90 分位为 8，而多数模型用 K=8 层训练，即至少 90% 的样本不存在 under-reaching；再加层到 10 层也没有改进，论文据此认为改进来自 over-squashing 的缓解而非可达性增加（Sec 4.2）[^src-over-squashing]。

### 与 over-smoothing

> [!warning] 概念区分：over-squashing 不是 over-smoothing
> 二者是不同的现象，不可混写或互相替代。over-smoothing 指层数增加后节点表示趋于不可区分；over-squashing 指指数增长的感受野信息被压进固定长度向量。论文还指出 over-smoothing 的实证证据主要来自短程任务，并以假设口径提出：在长程问题上，性能退化的解释是 over-squashing 而非 over-smoothing（Sec 1, Sec 6）[^src-over-squashing]。

论文用两个构造性例子说明二者可以独立发生（Appendix E）[^src-over-squashing]：

- 三个节点的完全图（问题半径 r=1）：加深层数可能出现 over-smoothing，但不存在被挤压的持续增长的信息量，因此没有 over-squashing。
- Tree-Neighbors-Match 任务：不同层的节点持有的信息量不同，没有理由收敛到相同表示，因此没有 over-smoothing，但 over-squashing 存在。

wiki 侧的对应条目是 [[over-smoothing-in-gnns]]；该页记录的是 over-smoothing 的机制与缓解，与本页概念分立。

## 实证证据（作者报告口径）

### Tree-Neighbors-Match 合成基准

论文构造了可控问题半径的 Tree-Neighbors-Match 任务：以目标节点为根、深度为 depth 的二叉树，树叶（带标签的绿色节点）信息沿边单向流向根，模型须依据「与目标节点蓝色邻居数相同的叶节点」的标签作答，问题半径 r = depth（Sec 3, Sec 4.1, Fig 2, Fig 5）[^src-over-squashing]。作者报告：从 r=4 起，部分 GNN 开始无法拟合训练集——GCN 在 r=4 的训练准确率为 70%；r=5 时所有 GNN 都无法完美拟合；GCN 与 GIN 最多完美拟合到 r=3，GGNN 与 GAT 在 r=4 仍达到 100%（Sec 4.1, Fig 3）[^src-over-squashing]。实验使用 d=32 的隐藏维度（Appendix A）[^src-over-squashing]。

### 控制实验：长程本身 vs 挤压量

为排除「长距离本身」导致无法拟合的替代解释，论文在附录中重复实验：保持叶节点到目标节点的距离（链长至多 6），但把信息挤压量降到与 r=2 相同。作者报告此时所有 GNN 都能轻松拟合到接近 100%，即问题在于挤压的信息量而非长距离本身（Appendix A）[^src-over-squashing]。

### 架构差异：GCN/GIN 与 GAT/GGNN

作者把 Fig 3 中 GNN 的差异归因于聚合计算：GCN 与 GIN 先把所有邻居消息求和压缩进单个向量、再与目标节点自身表示结合，因此必须把来自全部叶节点的信息压进一个向量；GAT 用注意力按目标节点表示对入边加权，在最后一层可以只吸收相关入边、只压缩「一半」的信息；GGNN 的 GRU 单元被论文假设（归因 Levy et al., 2018 的视角）以逐元素注意力的方式起类似过滤作用（Sec 4.1, Sec 5 脚注 1）[^src-over-squashing]。这对应摘要的论断：同等吸收所有入边的 GNN（GCN、GIN）比 GAT、GGNN 更易受 over-squashing 影响（Abstract）[^src-over-squashing]。

### FA 层干预实验（概述）

论文在多个已充分调参的真实基准上，把模型的最后一层改为 [[fully-adjacent-layer|全邻接层]]（fully-adjacent layer）以打破瓶颈。作者报告：QM9 上六种 GNN 误差率平均相对降低 42%（Sec 4.2, Table 1, Table 4）；NCI1 与 ENZYMES 上平均相对降低 4.8% 与 12%（Sec 4.3）；VarMisuse 上 SeenProjTest 88.4%、UnseenProjTest 83.8%（论文自述为新 SOTA）（Sec 4.4, Table 3）[^src-over-squashing]。细节与消融见 [[fully-adjacent-layer]]。

## 隐藏维度的组合下界（Sec 5）

论文在 Tree-Neighbors-Match 上做组合计数分析，给出完美拟合训练集所需最小隐藏维度 $d$ 相对问题半径 $r$ 的组合下界（Sec 5）[^src-over-squashing]：

$$b^{f \cdot d} > \frac{(m^r)!}{(m!)^{m^r - 1}}$$

其中 $m$ 为树的分叉数（实验取 2）、$b=2$ 为计数基、$f=32$ 为浮点位数（Sec 5, Eq 4-5）[^src-over-squashing]。作者报告：阶乘比常数底指数增长更快，$d$ 必须随 $r$ 急剧增大；d=32 时最大问题半径只有 r=7（Sec 5）[^src-over-squashing]。经验最小维度高于组合下界——论文解释为即使存在某种编码方案，梯度下降也不保证找到它；实验中 d=512 经验上最多拟合 r=7（Sec 5, Fig 4）[^src-over-squashing]。

口径说明：这是组合计数论证，论文称为 combinatorial lower bound（Sec 5），不是以定理-证明形式给出的结果；且分析范围有限——脚注 1 说明该分析对 GCN 与 GIN 成立，GAT 类使用接收方表示聚合消息、只需压缩一半叶节点信息，$r$ 的上界至多增加 1（Sec 5 脚注 1）[^src-over-squashing]。

## 论文自述的边界

- 长程解释是假设口径：论文在引言与相关工作中以「we hypothesize」提出 over-squashing（而非 over-smoothing）解释长程问题的性能退化，并假设长程模型实际过拟合的是短程信号与数据伪迹（Sec 1, Sec 4.1, Sec 6）[^src-over-squashing]。
- 短程问题不在论文范围内：论文明确把 focus 放在需要长程信息的问题上；引文网络、社交网络、商品推荐等短程领域不是其关注对象，这类问题用少量层即可表现良好（Sec 3）[^src-over-squashing]。
- FA 层是演示性方案：论文自述其目的只是证明 over-squashing「普遍且未被处理到连最简单的方案都有效」，主要贡献不是方案本身（Sec 4.2）[^src-over-squashing]。
- 在真实分子等场景中，最优消息传播结构未知，只能使用给定的关系（如化学键）作为边，无法像合成问题那样直接加边（Sec 3）[^src-over-squashing]。

## 相关工作与后续定位

论文相关工作中列举了此前「事实上绕开挤压」但未以 over-squashing 命名瓶颈的手段：Gilmer et al. (2017) 加 virtual edges 缩短长距离、Scarselli et al. (2008) 加 supersource 节点、Allamanis et al. (2018) 设计 16 种 shortcut 边类型的程序分析（Sec 6）[^src-over-squashing]。在本 wiki 中，[[graphgps|GraphGPS]] 的动机部分把 over-squashing 与 over-smoothing、1-WL 表达力并列为 MPNN 的限制，作为每层并行引入全局注意力分支的理由之一（GraphGPS Sec 2, Sec 3.3）[^src-graphgps]；[[topology-aware-graph-transformer|TGT]] 的全局注意力分支与 [[fully-adjacent-layer|FA 层]] 都属于让节点绕过逐层局部聚合直接交互的路线（wiki 组织口径）。

## 相关页面

- [[over-smoothing-in-gnns]] — 另一个概念：层数加深导致节点表示趋同；与本页概念分立，不可混用
- [[fully-adjacent-layer]] — 论文用于验证瓶颈的缓解方案及消融
- [[virtual-nodes-traffic]] — 本 wiki 中以 Virtual Nodes 图改写缓解 over-squashing 的长期交通预测机制页（后续应用路线）
- [[graphgps]] — 把 over-squashing 列为 MPNN 限制的图 Transformer 框架
- [[topology-aware-graph-transformer]] — 注意力矩阵层面的 local/global 混合（对照设计）
- [[source-over-squashing]] — 论文源摘要

[^src-over-squashing]: [[source-over-squashing]]
[^src-graphgps]: [[source-graphgps]]
