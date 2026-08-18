---
title: "WIRE：图的波致旋转位置编码"
type: technique
tags:
  - position-encoding
  - graph
  - rope
  - transformer
  - spectral
  - linear-attention
  - icml-2026
created: 2026-08-09
last_updated: 2026-08-13
source_count: 1
confidence: medium
status: active
---

# WIRE：图的波致旋转位置编码 (Wave-Induced Rotary Encodings)

WIRE（Wave-Induced Rotary Encodings）是 Reid 等人在 ICML 2026 提出的图位置编码方法，论文将其定位为 RoPE 在图结构数据上的推广：用图 Laplacian 的特征向量定义每个节点的谱坐标，再对 query/key 施加旋转[^src-2509-22259]。与基于偏置的相对位置编码不同，WIRE 直接旋转 token 而非修改注意力 logits，因此可与线性注意力兼容[^src-2509-22259]。

## 动机：图上的位置编码困境

图缺少规范的坐标系。文本有 token 顺序、图像有像素网格，而图的节点排列本身是任意的，不存在天然坐标，这让位置编码的设计更加复杂[^src-2509-22259]。

已有的两类方案各有缺陷：

- **谱 APE**：用图 Laplacian 的谱构造绝对位置编码，在网格图的特殊情形下接近文本/图像上的正弦 APE，但一般不具备不变性[^src-2509-22259]。
- **谱 RPE**：计算节点对的最短路径距离或有效电阻等结构量作为注意力偏置，效果更好，但必须实例化 $N \times N$ 注意力矩阵，无法与线性注意力兼容[^src-2509-22259]。

RoPE 在 LLM 与 ViT 中的成功引出一个问题：能否把旋转位置编码也用到图数据上？WIRE 给出的答案是肯定的[^src-2509-22259]。

## WIRE 算法

论文的 Alg. 1 由三步构成[^src-2509-22259]：

1. **谱分解**：计算图 Laplacian 最低的 $m \le N$ 个特征向量与特征值 $\{\boldsymbol{u}_k, \lambda_k\}_{k=0}^{m-1}$，可用精确方法或近似迭代方法[^src-2509-22259]。
2. **谱特征**：为每个节点定义谱坐标，例如 $\boldsymbol{r}_i = [\boldsymbol{u}_k[i]]_{k=0}^{m-1} \in \mathbb{R}^m$[^src-2509-22259]。
3. **旋转**：用这些谱特征施加 RoPE 旋转 $\boldsymbol{z}_i \to \mathrm{RoPE}(\boldsymbol{r}_i)\boldsymbol{z}_i$，对 query 与 key 均适用[^src-2509-22259]。

### 为什么叫 "wave"

Laplacian 特征向量对应图上逐级升高的振荡频率：$\boldsymbol{u}_0$ 在连通图上为常数，$\boldsymbol{u}_1$（Fiedler 向量）变化缓慢、可用于把图切成两部分；截断到 $m$ 个频率即保留图上最慢的 $m$ 个振荡模式[^src-2509-22259]。

### 参数量

WIRE 唯一可学习的参数是频率 $\{\boldsymbol{\omega}_n\}_{n=1}^{d/2} \subset \mathbb{R}^m$，每层共 $dm/2$ 个；通常 $m \ll d$，与网络其余部分相比极小[^src-2509-22259]。论文报告在实验中 $m=3$ 时 WIRE 参数不足整个模型参数的 1%[^src-2509-22259]。

## 核心性质

### RoPE 是 WIRE 的特例（Theorem 2）

路径图 $P_N$ 的第二特征向量 $\boldsymbol{u}_1[i] = -\cos\left(\frac{i+1/2}{N}\pi\right)$ 沿节点单调变化，与 token 位置坐标 $[0, 1, \ldots, N-1]$ 只差简单的双射坐标变换（缩放、平移、余弦压缩）；选取特定频率 $\boldsymbol{\omega}_i$ 后即可恢复 LLM 中使用的标准 RoPE[^src-2509-22259]。二维网格图是路径图的笛卡尔积 $P_{N_x} \square P_{N_y}$，谱随之分解，等价于对每个轴独立施加一维 RoPE，即 ViT 中的 RoPE[^src-2509-22259]。

### 有效电阻依赖（Theorem 3）

取谱特征 $\boldsymbol{r}_i = [\boldsymbol{u}_k[i]/\sqrt{\lambda_k}]$、频率随机采样自零均值高斯分布时，期望的 query-key 内积满足[^src-2509-22259]：

$$\mathbb{E}[(\mathrm{RoPE}(\boldsymbol{r}_i)\boldsymbol{q}_i)^\top \mathrm{RoPE}(\boldsymbol{r}_j)\boldsymbol{k}_j] = \boldsymbol{q}_i^\top \boldsymbol{k}_j \left(1 - \frac{\omega^2 R(i,j)}{2}\right) + \mathcal{O}(\omega^4)$$

其中 $R(i,j)$ 是节点 $i,j$ 之间的有效电阻，它构成节点集上的度量，且是最短路径距离的下界（树上取等号）[^src-2509-22259]。即期望上 WIRE 的领头效应是按有效电阻成比例地压低 query-key logits，节点相距越远注意力越被压低[^src-2509-22259]。值得注意的是，这一不变性无需实例化注意力矩阵即可获得，而此前这类性质只能通过昂贵的 RPE 软掩码实现[^src-2509-22259]。

### 线性注意力兼容

旋转直接作用于 token（而非 query-key 对的 logits），因此 WIRE 可与 Performer 等线性注意力结合，实现论文所称的"线性注意力下的拓扑掩码"，这是高效 transformer 社区长期追求的目标[^src-2509-22259]。

### 节点置换等变（Lemma 1）

WIRE 变换对节点排序置换等变，唯一的歧义来自特征向量的符号翻转与简并子空间内的旋转[^src-2509-22259]。论文进一步指出，随机化 WIRE 的极限变换在这些符号与基变换下是精确规范不变的（gauge invariant），这可能解释了方法的鲁棒性[^src-2509-22259]。

### 表达力

WIRE 可以区分在 1 维 Weisfeiler-Lehman 同构测试下不可区分的图（将颜色替换为节点特征），论文将其作为 sanity check 提出，并指出这一性质也为高阶 GNN 等模型共有[^src-2509-22259]。

## 实验证据

### 合成任务

- **单色连通子图回归**：5×5 网格随机删边后预测最大单色连通子图的规模（10,000 训练 / 1,000 测试图），论文报告 WIRE 相对无 WIRE 基线（$m=0$）一致提升；图越接近网格时低维谱特征就够用，删边越多、拓扑越复杂时高频特征越有帮助[^src-2509-22259]。
- **最短路径距离预测**：Watts-Strogatz 图（$N=10$，$k=2$，重连概率 $p=0.6$）上预测两节点最短路径距离，测试 RMSE 从基线（$m=0$）的 0.065 降到 $m=5$ 时的 0.038，几乎减半[^src-2509-22259]。

### 点云

论文在 $k$-NN 图（$k=20$，$m=10$ 谱特征）上报告 WIRE 一致超过无位置编码基线：ModelNet40 分类准确率 Transformer 为 93.4% vs NoPE 91.8%、Performer 为 90.8% vs 90.1%；ShapeNet 分割 Transformer 为 93.2% vs 93.1%、Performer 为 93.0% vs 92.8%（单 seed）[^src-2509-22259]。点云场景下 WIRE 天然具有 SE(3) 不变性（$k$-NN 图及其谱在整体平移旋转下不变），而基于笛卡尔坐标的 RoPE 不具备该性质[^src-2509-22259]。

### GNN benchmark

论文在 GraphGPS + ReLU Performer 线性注意力设置下评估了 12 个数据集（MNIST、CIFAR10、PATTERN、CLUSTER、OGB 四项、LRGB 三项、MalNet-Tiny 等），报告加 WIRE 普遍提升多个百分点，且经常大幅缩小甚至弥合 Performer 与全秩 softmax transformer 之间的差距[^src-2509-22259]。例如 MalNet-Tiny 上 WIRE Performer 与全秩 transformer 效果相当，但可在单张 T4 12GB GPU 上训练[^src-2509-22259]。

## 实现效率

- **旋转本身 O(d)**：RoPE 旋转可写成逐元素 Hadamard 乘积加置换（论文 Eq. 11），无需矩阵乘法，只有 $\mathcal{O}(d)$ 次运算[^src-2509-22259]。
- **谱分解**：精确对角化 Laplacian 需 $\mathcal{O}(N^3)$，但 Lanczos、图粗化等近似方法很多对原图规模是 $\mathcal{O}(N)$，且 WIRE 通常只需要最低的少数几个特征向量[^src-2509-22259]。
- **成本摊销**：实践中研究者几乎总是已经计算某种结构特征用作 APE，同一特征直接用于 WIRE 的额外开销可忽略[^src-2509-22259]。
- **RWPE 变体**：完全避开谱分解，稀疏图上 $\mathcal{O}(N)$ 计算，论文报告同样带来准确率提升，但失去理论保证[^src-2509-22259]。

## 局限性

- 精确对角化 $\mathcal{O}(N^3)$ 不适用于超大图；近似方法或 RWPE 变体可以缓解，但会损失理论保证[^src-2509-22259]。
- 特征向量存在符号翻转与简并子空间基选择歧义，实践中通过数据增强式训练让模型学习这些歧义（论文认为足够），也可叠加 SignNet 等规范不变变换（论文发现影响不大）[^src-2509-22259]。
- WIRE 并非精确地按有效电阻不变：Theorem 3 是对随机频率的期望结果，且含 $\mathcal{O}(\omega^4)$ 修正项；实践中使用单个可学习实例而非随机集成平均[^src-2509-22259]。

## 相关页面

- [[rope]] — RoPE 技术详解；WIRE 在网格图上恢复标准 RoPE，是其特例
- [[roformer]] — RoFormer 模型实体
- [[spectral-kernel-linear-attention]] — WIRE 旋转读作核的随机特征后落地为线性注意力（本 wiki 分析）
- [[source-2509-22259]] — WIRE 论文源摘要

[^src-2509-22259]: [[source-2509-22259]]
