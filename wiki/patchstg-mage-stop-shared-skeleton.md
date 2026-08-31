---
title: "PatchSTG / MAGE / STOP：同一个骨架的三种参数化"
type: analysis
tags:
  - spatio-temporal
  - traffic-forecasting
  - attention
  - adaptive-graph
  - mixture-of-experts
  - model-comparison
created: 2026-08-31
last_updated: 2026-08-31
source_count: 5
confidence: medium
status: active
---

# PatchSTG / MAGE / STOP：同一个骨架的三种参数化

问题：[[patchstg|PatchSTG]]（KDD 2025）、[[mage|MAGE]]（NeurIPS 2025）、[[stop|STOP]]（ICML 2025）是否在做同一个东西。方法：把每层空间交互写成 $H'=\mathcal{M}(H)\,H$（$H\in\mathbb{R}^{N\times d}$，另加逐节点项），比较 $\mathcal{M}$。结论：三家服从同一组约束——**每节点显式交互预算 $O(c)$、$c\ll N$；单层复合后等效矩阵稠密；输入只改边权、不改边结构**——并在这组约束下给出三种互不等价的参数化。§1 写下三个算子，§2、§3 从公式读出共同点与差异，§4 给出共同缺失的自由度。公式改写与秩/自由度计数为本页推导，论文自述与实验数字随文标注。

## 1. 三个算子

### 1.1 PatchSTG：可分离参数化

patch 后 token 构成 $R\times P$ 网格，$(r,p)$ =（patch，patch 内 BFS 位置），掩码形式见 [[patchstg-sparse-attention-form]]：

$$A^{\mathrm D}_{(r,p),(r',p')}=\delta_{rr'}\,D^{(r)}_{pp'},\qquad A^{\mathrm B}_{(r,p),(r',p')}=\delta_{pp'}\,B^{(p)}_{rr'}$$

$D^{(r)}$（patch 内 $P\times P$）与 $B^{(p)}$（同位置跨 patch $R\times R$）均为 softmax 输出。一层复合 $\mathcal{M}=A^{\mathrm B}A^{\mathrm D}$，两个 $\delta$ 使求和塌缩为单项：

$$\mathcal{M}_{(r,p),(r',p')}=B^{(p)}_{rr'}\,D^{(r')}_{pp'}$$

从这条分解读出三个事实：

- **稠密、满秩**：右端恒正，任意点对一层可达；rank 亏损失效参数集是代数簇、测度零，generic 参数下 $\mathcal{M}$ 满秩。$P{+}R{-}1$ 只是单层计算图的 key 预算，不是等效矩阵的 support。
- **自由度可分离**：独立 logit 共 $PR^2+RP^2=M(P+R)$ 个（与论文复杂度 $O(\max(P,R)\,Md)$ 一致[^src-patchstg]）；代价是每条边权被约束为「位置级 $B^{(p)}$ × patch 级 $D^{(r')}$」两个低阶量之积，不能独立指定。
- **输入依赖**：每个 logit 都是当前样本的函数——权值完全动态，边结构锁死在可分离族内。

### 1.2 MAGE：静态图库 + 标量路由

$$\mathcal{M}(X)=\sum_{k=1}^{K_G}\operatorname{diag}\!\big(\alpha_{\cdot k}(X)\big)\,A^{(k)},\qquad A^{(k)}=\mathrm{Softmax}(E_1^{(k)})\,\mathrm{Softmax}(E_2^{(k)})^{\top}-\lambda_k\,\mathrm{Softmax}(E_3^{(k)})\,\mathrm{Softmax}(E_4^{(k)})^{\top}$$

- **静态部分**：$K_G$ 张差分专家图（默认 16），纯参数，$E_i^{(k)}\in\mathbb{R}^{N\times d_G}$。
- **动态部分**：路由 $\tilde\alpha_{ik}=\mathrm{Sigmoid}(H_i^{(c-1)\top}\theta_k+\gamma_k)$ 逐样本、逐层、逐节点（Top-$K{=}4$ 激活）[^src-mage]，故等效图 $\mathcal{M}(X)$ 随样本变——但输入只进入 $N\times K_G$ 个标量，边形状只能被逐节点缩放，不能重排。
- **秩**：$\operatorname{rank}A^{(k)}\le 2d_G$，行空间相加，$\operatorname{rank}\mathcal{M}\le\min\{N,\,2K_Gd_G\}=1024$（默认配置；论文自述口径为表示矩阵秩 $\le\min\{d,Kd_G\}$、$K\ge\lceil d/d_G\rceil$ 时恢复满秩 $d$[^src-mage]）。

### 1.3 STOP：$K$ 秩瓶颈

空间分支为两段 softmax 之积（aggregation 节点→ConAU、diffusion ConAU→节点，见 [[centralized-message-passing]]）：

$$H'_{\text{spatial}}=S_d\,(S_a\,H),\qquad S_a\in\mathbb{R}^{K\times N},\quad S_d\in\mathbb{R}^{N\times K}$$

- $\operatorname{rank}(S_dS_a)\le K$；$S_dS_a$ 元素 $=\sum_k S_{d,ik}S_{a,kj}$ 恒正，故稠密、全图两跳可达[^src-stop]。
- 逐节点项显式：$\hat Y=Y_t+Y_s$，$Y_t$ 为逐节点时间分支——秩约束只辖空间分支[^src-stop]。
- 输入依赖：节点↔单元间的 logit。

## 2. 共同骨架

三条性质，逐条从 §1 公式验证：

**(a) 每节点预算 $O(c)$，成本线性于 $N$**。显式 key 预算：PatchSTG $P{+}R{-}1$、MAGE $K{=}4$、STOP $K$；复杂度 $O(\max(P,R)Md)$[^src-patchstg] / $O(N\,d\,d_G)$[^src-mage] / $O(KNd)$[^src-stop]。（$c$ 是超参：论文最优 patch 数随数据集增长，SD 16 → CA 512[^src-patchstg]，故 $c\ll N$ 但非与 $N$ 无关。）

**(b) 单层全图可达**。三家等效混合矩阵都稠密：PatchSTG、STOP 恒正（§1.1、§1.3），MAGE 的差分图 generic 非零（相消需要参数落在测度零集上）。都不靠堆层换感受野。

**(c) reweight，not re-route**。三家的输入依赖通道固定：PatchSTG 的 logit 族固定在可分离族内、MAGE 的边形状固定为 $K_G$ 张静态图、STOP 的单元集合固定。没有任何一家能随样本改变「谁与谁交互」的结构。

(c) 丢掉的任意逐对交互，三篇各自的消融独立报告为任务不需要：MAGE 纯线性配置（16:0）已 Pareto 最优、加全秩图无收益[^src-mage]；STOP 的 -graph 变体（去消息传递）在 OOD 设定下反而更好[^src-stop]；PatchSTG 引 STID 洞察——大图上消息传递 over-smoothing 的伤害大于缺失空间交互[^src-patchstg]。

## 3. 差异：从公式读出

| | 输入依赖入口 | 等效矩阵秩 | N 绑定参数 | 独占证据 |
|---|---|---|---|---|
| PatchSTG | 每个 logit（权值全动态）| 满秩（generic，可分离族）| learnable spatial embedding；掩码几何派生 | LargeST 四数据集 SOTA（发表时）[^src-patchstg] |
| MAGE | 仅 $N{\times}K_G$ 路由标量 | $\le 2K_Gd_G=1024$ | $E^{(k)}\in\mathbb{R}^{N\times d_G}$（transductive）| 17 数据集 94% 指标；比 PatchSTG 快 4.7×[^src-mage] |
| STOP | 节点↔单元 logit | $\le K$（空间分支）| 无（归纳）| OOD +17.01% / 归纳 +18.44%[^src-stop] |

前三列互不等价，均可由 §1 公式直接验证：可分离族是 $M^2$ 维空间中的测度零子集；秩 $\le K$ 的构造给不出秩 $K{+}1$ 的混合；静态图库的动态只有逐节点缩放一种形状。N 绑定列决定迁移设定：MAGE 只能图内；STOP 天然归纳（新增节点直接读共享单元）；PatchSTG 掩码可随坐标重建、embedding 表须剥除（跨图押注的另一极端是 [[stunet|STUNet]]：显式邻接 token 化并冻结[^src-stunet]）。效率证据三家共享同一锚 D²STGNN（10×[^src-patchstg] / 118–960×[^src-mage] / ~20×[^src-stop]），不构成区分。

## 4. 共同空白：内容动态 support

(c) 的补集即三家共同缺失的自由度：随样本改变的边结构。MAGE 的 Top-K 选的是静态算子、不选边；PatchSTG 的划分在预处理期定死（论文将自适应重划分列为未来工作[^src-patchstg]）；STOP 的单元集合固定。已收录工作中对该空白的正面攻击是 [[lets-group|Let's Group]]（IJCAI 2025）：批评静态子图划分无法捕捉动态时空依赖（将 PatchSTG 的地理坐标划分与 FCGCN/Louvain、LarSTL/METIS 并列），改为按特征相似度、以可学习记忆向量划分子图（该文 Sec 1–2，作者自述[^src-lets-group]）。

最小检验：固定候选集（地理 K 近邻），对比「静态排序选邻居」与「按当前状态重排序选邻居」——一档之差即可分离该自由度是否承重。

## 关联

- [[patchstg-sparse-attention-form]] — PatchSTG 掩码形式与对齐方式的一般化
- [[stg-attention]] — 「避开 $N^2$ 配对交互」路线谱系；本页是该谱系三条路线代表作的共同骨架层
- [[centralized-message-passing]] / [[sparse-balanced-mixture-of-experts-st]] — STOP / MAGE 机制页

[^src-patchstg]: [[source-patchstg]]
[^src-mage]: [[source-mage]]
[^src-stop]: [[source-stop]]
[^src-stunet]: [[source-stunet]]
[^src-lets-group]: [[source-lets-group]]
