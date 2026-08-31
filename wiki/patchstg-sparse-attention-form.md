---
title: "PatchSTG 双注意力的稀疏注意力形式"
type: analysis
tags:
  - attention
  - sparse-attention
  - spatial-temporal
  - traffic-forecasting
  - transformer
created: 2026-08-31
last_updated: 2026-08-31
source_count: 2
confidence: medium
status: active
---

# PatchSTG 双注意力的稀疏注意力形式

问题：[[patchstg|PatchSTG]] 的 depth/breadth dual attention 是否本质上是稀疏注意力，掩码结构能否显式写出。结论：可以。把 patch 后的 token 看成 $R\times P$ 网格，dual attention 等价于两个固定掩码下的结构化稀疏注意力，一层算子是二者的复合。各公式的原始出处均归因到论文；$\delta$ 掩码改写与谱系定位为本页推导与归类。

## 形式化

**记号**（论文 Sec 4.2）：$N$ 个传感器点经 [[leaf-kdtree|leaf KDTree]] 划分、cosine-similarity padding 与子树回溯（见 [[irregular-spatial-patching]]），得到 $R$ 个 patch、每 patch $P$ 个点（$P=C\cdot N_p$，$N_p$ 为 2 的幂），填充后共 $M=R\times P\geq N$ 个 token。patched 输入 $\mathcal{X}\in\mathbb{R}^{R\times P\times d}$，token 记 $(r,p)$：$r$ 为 patch（子树），$p$ 为 patch 内位置（BFS 序）[^src-patchstg]。

**Depth attention（块内）**：论文 Eq. 8 在 $R$ 个 patch 上各做一个 $P\times P$ 的 multi-head softmax attention（注意力张量形状 $\mathbb{R}^{R\times P\times P}$）[^src-patchstg]。改写成 $M\times M$ 矩阵上的带掩码注意力（本页改写）：

$$A^{\mathrm{D}}_{(r,p),(r',p')}=\delta_{r,r'}\cdot\operatorname{softmax}_{p'}\!\left(\frac{q_{r,p}{}^{\top}k_{r,p'}}{\sqrt{d/o}}\right)$$

掩码为块对角 $\Omega^{\mathrm{D}}=\mathbf{1}[r=r']$：每个点只 attend 同 patch 的地理近邻。

**Breadth attention（跨块、同位置）**：论文 Sec 4.3 将张量转置为 $\mathbb{R}^{P\times R\times d/o}$，对每个 patch 内位置 $p$ 在全部 $R$ 个 patch 间做 attention（注意力张量形状 $\mathbb{R}^{P\times R\times R}$）[^src-patchstg]：

$$A^{\mathrm{B}}_{(r,p),(r',p')}=\delta_{p,p'}\cdot\operatorname{softmax}_{r'}\!\left(\frac{q_{r,p}{}^{\top}k_{r',p}}{\sqrt{d/o}}\right)$$

掩码为位置对齐 $\Omega^{\mathrm{B}}=\mathbf{1}[p=p']$：每个点 attend 其他 patch 中同 BFS 序位置（各区域内结构可比位置）的点。

论文 Table 1 给出的 patching 范式形式 $\big\Vert_{i=1}^{R}(Q_{(i)}K_{(i)}^{\top})V_{(i)}$ 即 depth 方向的块拼接记法；breadth 是同一网格另一条轴上的对偶操作 [^src-patchstg]。

**一层复合算子**（省略残差与归一化；论文中 breadth 的 Q/K/V 直接取自 depth 输出，两层注意力交替堆叠）[^src-patchstg]：

$$\mathcal{X}^{(l)}=A^{\mathrm{B}}\big(A^{\mathrm{D}}\mathcal{X}^{(l-1)}\big),\qquad l=1,\dots,L$$

论文实验设置取 $L=5$（Sec 5.1.4）[^src-patchstg]。

**复杂度核对**：每个 token 一层的非零 key 预算为 $P+R-1$（同 patch $P$ 个 + 同位置 $R$ 个，减自身重复计数一次）；logit 计算量 $RP^2+PR^2=M(P+R)$，对比稠密注意力的 $M^2$。论文 Sec 4.5 报告 depth/breadth 复杂度分别为 $O(RP^2d)$、$O(PR^2d)$，主导复杂度 $O(\max(P,R)\,Md)$，与该计数一致（$P,R\ll N$，$M\approx N$）[^src-patchstg]。

## 稀疏模式的三个特征

1. **模式静态、内容动态**。$\Omega^{\mathrm{D}}$ 与 $\Omega^{\mathrm{B}}$ 由预处理阶段 KDTree 一次确定：数据无关、不参与学习、无 top-k 选择；动态的只有 attention 权值。本页将其归入 static structured sparsity（Longformer/Swin 一族），区别于 content-based 稀疏（Reformer、[[quest-attention|QUEST]] 一族）与核近似（[[performer|Performer]]）。
2. **掩码承载语义**。$\Omega^{\mathrm{D}}$ 承载地理局部性（同子树点地理相邻；论文消融中 w/o LKDT 在四个数据集上均为最大退化项）[^src-patchstg]；$\Omega^{\mathrm{B}}$ 承载 BFS 序对齐。论文自述的 fidelity 源于此：每个位置 $p$ 有独立的 $R\times R$ 全局注意力，而非把 patch 压缩成单 token——Table 1 中 patching 行 Information Loss = ✘ 的含义是 K/V 不做低秩压缩 [^src-patchstg]。
3. **全局感受野靠复合获得**。单层内不存在「不同 patch 且不同位置」点对的直接交互；两跳路径 $(r,p)\to(r,p'')\to(r',p'')$ 使一层之后感受野覆盖全图。复合 $A^{\mathrm{B}}A^{\mathrm{D}}$ 是两个独立 softmax 的乘积，不能改写成并集掩码 $(\Omega^{\mathrm{D}}\cup\Omega^{\mathrm{B}})$ 下的单个 softmax——这既是表达力限制，也是效率来源（本页分析）。

## 对齐方式的一般化：同位置是否必要

追问（2026-08-31）：breadth 的同 index 一一对应若换成「只看第一个位置」「随机一个位置」，效果损失多大；该对齐是算力方便还是建模必要。

### 通用形式（本页推导）

对位置轴取划分 $\psi:[P]\to[K]$（$C_k=\psi^{-1}(k)$），breadth 推广为类分组注意力：

$$A^{\mathrm{B}}_{(r,p),(r',p')}=\delta_{\psi(p),\psi(p')}\cdot\operatorname{softmax}_{r',\,p'\in C_{\psi(p)}}\!\left(\frac{q_{r,p}{}^{\top}k_{r',p'}}{\sqrt{d/o}}\right)$$

logit 总量 $R^2\sum_k |C_k|^2$（类 $k$ 含 $R|C_k|$ 个 query，各 attend $R|C_k|$ 个 key）。$K=P$（$\psi=\mathrm{Id}$）退回 breadth（$PR^2$）；$K=1$ 退回跨块稠密（$M^2$）；均衡类下 $R^2P^2/K$。PatchSTG 取最便宜端 $K=P$。此族之外有两条正交的省算力轴（本页归类）：**查询子采样**——只对 $S\subseteq[P]$ 的位置做 breadth，logit $|S|R^2$，其余位置的全局信息推迟到后续 composite 经 depth 扩散获得；**anchor 池化**——全体 query 对 $K_a$ 个 anchor token 做 cross-attention，logit $MK_a$，即 [[query-aggregate-attention|STUNet]] 与 [[adaptive-graph-agent-attention|FaST]] 的拓扑路线（$K_a=R$ 时与 breadth 同价但值池由 $P$ 个降为 1 个，$K_a<R$ 才真正省算力）。

### 替换变体的分层后果（本页推导；论文无对应实验）

- **只看 $p'{=}0$**（所有 query attend 各块第 0 位置的 token，logit 不变 $MR$）：一层内跨块值带宽 $M\to R$（差 $P$ 倍——同位置下每个 token 的 value 恰好进入自己所在列的全局混合）；每位置独立全局视角 $P\to 1$（query 仍随 $p$ 不同，但 value 池相同）。$L$ 层 depth 扩散可部分补偿，预期是劣化而非崩溃。
- **随机配对**（每 query 与各块随机位置建边，无类结构）：logit 与带宽期望不变（每 key 期望被读 $R$ 次，与同位置相同），视角数不减；损失的只有对应先验——softmax 竞争池从「同 BFS 序可比位置」变为任意位置，cosine padding 点的索引语义错位（索引语义载重的间接证据：w/o PadSim 在 CA 上 17.35→17.87）[^src-patchstg]。该变体是唯一能干净分离「实现方便 vs 语义先验」的消融，论文未做（本页核对全文无此变体）。
- **单 key**（每 query 只读 1 个随机 token）：softmax 退化为固定路由，全局聚合能力丧失，等价于随机传播。

### 证据边界与结论

- 论文直接证据只有 w/o FGGC（breadth 时把 patch 内全部点融成单 patch 点，即「只看第一个」的更激进版本，Q/K/V 一并融合）：四数据集 MAE 全退化 +1.1–2.8%（Table 4：SD 16.90→17.37、GBA 19.50→19.72、GLA 18.96→19.49、CA 17.35→17.63），量级与 w/o Depth 相当（+0.7–2.2%）、小于 w/o Breadth（+2.0–4.7%）[^src-patchstg]。推断「只看第一个」劣化不劣于 FGGC、不优于完整 breadth（非论文结论）。
- 论文对同 index 的全部自述理由（Sec 4.3/5.4）：同 index 点已混入 depth 局部信息；逐 index 相关模式多样（Fig. 6）故 fidelity [^src-patchstg]。「BFS 同序 = 结构可比位置」是本页重构的先验解释，论文未论证。
- 结论：同位置对齐**不是**省算力妥协——同价替代变体在带宽或先验上严格更弱，而同位置在 $O(MR)$ 预算内同时保住带宽 $M$、视角 $P$ 与零 gather（reshape 即可实现，属实现红利而非设计动因）。推导通用空间注意力时建议保留两条不变量：每层跨块值带宽（每个 value 进入一次全局混合）、每位置独立竞争池；将「精确 BFS 对应」松弛为可学习对齐或软分组（$K<P$ 或可学习 $\psi$），代价按 $R^2\sum_k|C_k|^2$ 增长。

## 边界

- 「本质是稀疏注意力」在复杂度意义上成立；但 PatchSTG 的贡献点不在掩码模式本身——固定块稀疏/轴向注意力是已有技术——而在为不规则分布点集构造语义正确的 $R\times P$ 网格：行等价于地理局部性，依赖 leaf KDTree 的平衡不重叠划分，消融中 METIS/KMeans 替代均退化（论文 Table 4）[^src-patchstg]。
- 「无损」是论文自述口径：指 K/V 不压缩、注意力权重逐点可读；跨块异位点对的直接交互确实被稀疏掉，且划分在整个训练与推理期静态固定（论文将自适应重划分列为未来工作）[^src-patchstg]。
- [[lets-group|Let's Group]]（IJCAI 2025）从「静态子图划分无法捕捉动态时空依赖」批评这一族方法（将 PatchSTG 的地理坐标划分与 FCGCN/Louvain、LarSTL/METIS 并列），提出按特征相似度用可学习记忆向量划分子图（该文 Sec 1–2，作者自述）[^src-lets-group]。

## 谱系定位

本 wiki 将其归入「避开 $N^2$ 配对交互」的交通空间建模路线之一（与 [[stg-attention]] 页的谱系互补）：

| 路线 | 代表 | 机制 |
|------|------|------|
| 线性化 | [[stg-attention\|STG-Attention]]（STGformer） | 保留 QKᵀ 形式，分解内积降复杂度 |
| 改变交互拓扑 | [[query-aggregate-attention]]（STUNet）、[[adaptive-graph-agent-attention]]（FaST） | 经中间 token/agent 中转 |
| 结构化稀疏 | PatchSTG（本页） | 固定掩码 + 双 softmax 复合 |

与 [[source-patchtst|PatchTST]] 的对照：同为 patch 化缩减注意力规模，PatchTST patch 时间轴（通道内独立、规则滑窗），PatchSTG patch 空间轴（跨点交互、KDTree 划分），掩码来源不同。

本页三路线之外的正交观察：MAGE（kernel 近似线性化路线）与 STOP（经共享单元中转的改变拓扑路线）的代表作与 PatchSTG 共享同一更深层的骨架承诺——共享低维中介 + 逐节点个性化项 + 静态 support；该骨架在参数化族层面的三种展开见 [[patchstg-mage-stop-shared-skeleton]]。

[^src-patchstg]: [[source-patchstg]]
[^src-lets-group]: [[source-lets-group]]
