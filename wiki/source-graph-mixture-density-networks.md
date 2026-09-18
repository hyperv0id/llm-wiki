---
title: "Graph Mixture Density Networks"
type: source-summary
tags:
  - mixture-of-experts
  - mixture-density-network
  - graph-neural-network
  - conditional-density-estimation
  - multimodal-output
created: 2026-09-17
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# Graph Mixture Density Networks

**Authors**: Federico Errica, Davide Bacciu, Alessio Micheli（University of Pisa）。[^src-graph-mixture-density-networks]
**Venue**: ICML 2021（"Proceedings of the 38th International Conference on Machine Learning, PMLR 139, 2021"）。[^src-graph-mixture-density-networks]
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/H8V2EDFK/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 与 MoE 的边界（本页重点）

这篇论文走 **Mixture Density Network (MDN)** 路线而非 MoE，且在 Related Works（Sec 2）与 Sec 3 两处明确划界。论文对 MoE 的定义是：Jacobs et al. 1991 / Jordan & Jacobs 1994 提出的模型由多个神经网络（local experts）组成，每个 expert 期望解决一个子任务，一个 gating network 按输入为 local experts 的贡献加权，整体输出是各 expert 输出的加权组合[^src-graph-mixture-density-networks]。论文自述的取舍在 Sec 3：**"Differently from the Mixture of Experts, which would require a new DGN encoder for each output distribution i, we follow the Mixture Density Network approach and share $h^{\mathcal V}_g$ between the sub-networks"**——MoE 要为每个输出分布配一个独立的 DGN 编码器，参数多，处理输入的成本越高 MDN 相对 MoE 的效率优势越大（Related Works：input 是 vectorial 时 MDN 已更省，处理结构化输入时差距进一步放大），且多个 DGN 编码器在大数据集上 computationally intractable[^src-graph-mixture-density-networks]。MDN 让子网络之间 cooperate：共享的隐表示同时用于产生 mixing weights 与各输出分布的参数，代价是共享编码必须装下任务的全部信息。

## 问题

回归常用 MSE / Cross-Entropy 训练，模型近似条件期望 $\langle y|x\rangle$；当目标分布是 multimodal 时（同一输入对应多个可能结果），这类模型会输出被平均后的值。MDN（Bishop 1994）可在向量输入上近似任意条件分布，但真实问题常是关系型数据——网络结构本身影响结果（如流行病传播）[^src-graph-mixture-density-networks]。论文提出的 GMDN 把 DGN 与 MDN 结合，输出以任意拓扑图为条件的 multimodal 分布（graph 级或 node 级）；论文自述 "To the best of our knowledge, this is the first DGN that can learn multimodal output distributions conditioned on arbitrary input graphs"[^src-graph-mixture-density-networks]。

## 方法

Eq.1–8（Sec 3）[^src-graph-mixture-density-networks]：

- **建模**：引入潜在变量 $Q_g$（Categorical，C 个状态）对 $P(y_g|g)$ 做边缘化（Eq.1），MLE 最大化似然。每个图用 DGN（实验用 GIN 卷积，Eq.2，保证置换不变）编码到节点表示 $h^{\mathcal V}_g$，图级任务再做 readout $h_g=r_g(h^{\mathcal V}_g)$（Eq.3）。
- **混合权重**：$P(Q_g|g)=\sigma(r^Q_g(h^{\mathcal V}_g))$，softmax 生成 $C$ 个 mixing weights（Eq.4）；各子网络 $\Phi_i$ 输出对应分布参数，如多变量 Gaussian 时 $\mu_i,\Sigma_i=\Phi_i(h_g)=f_i(r^i_g(h^{\mathcal V}_g))$（Eq.5）。子网络为 Linear model（超参默认）；实验中 $C\in\{3,5\}$、卷积层数 $\{2,5,7\}$、hidden 64、readout $\{$sum, mean$\}$。
- **训练**：EM 框架下做 MLE——E 步解析计算指示变量后验 $E[z^g_i|D]$（Eq.7），M 步无闭式解、用梯度上升最大化完全似然下界（Eq.6），即 Generalized EM。另加可选的 Dirichlet 正则 $\log\pi(Q_g|\alpha)$（Eq.8），$\alpha=1_C$ 即均匀先验（无正则），目的是防止 $Q_g$ 的概率质量塌缩到单一状态[^src-graph-mixture-density-networks]。

## 实验

自建随机图随机 SIR 流行病模拟基准 + 两个真实化学回归任务，指标为 test log-likelihood（10 次训练取平均，括号内标准差），holdout 80/10/10；同一图的模拟不跨 train/test 切分[^src-graph-mixture-density-networks]。

- **TABLE 1（BA-100 / ER-100，各 12,000 测试样本）**：GMDN log L −0.67(.02) / −1.56(.04)，优于 DGN −0.90(.35) / −1.96(.16)、MDN −1.17(.05) / −2.54(.07)、HIST −1.16 / −2.32、RAND −4.60。一个如实记录的观察：HIST 的 log-likelihood 在两个数据集上都好于 MDN，论文据此推论结构信息是主要增益来源[^src-graph-mixture-density-networks]。另一变体（DGN 以 L1 训练后接 MDN）log L ≈ −16，论文归因于 DGN 把不同分布但同均值的图嵌入压得相似、信息严重丢失。
- **Fig.4（C=5）**：mixing weights 随 $R_0=\beta/\gamma$ 增大把多数子网络 shut down，子网络 3、4 控制输出分布；$R_0$ 高时一个子网络就够。
- **Fig.6（迁移）**：ER-100 训练的 GMDN 迁移到 size 50–500 的 BA/ER 图总体更好，论文解释为源任务更难使模型学到更完整的 SIR 动态。
- **TABLE 2（alchemy_full / ZINC_full）**：GMDN log L −0.57(1.4) / −0.75(.10)，均为最好；MAE（次要指标，用子网络加权均值作预测）alchemy_full 0.61、ZINC_full 0.49(.04)，与 DGN 的 0.49(.03) 相当——论文明确说明 MAE 不反映模型的不确定性建模，log-likelihood 才是该任务的自然指标[^src-graph-mixture-density-networks]。

## 局限

论文无独立 Limitations 章节，结论中也未自述局限。可如实记录的自述性限制散见正文：mixing weights 需防塌缩到单一状态（以 Dirichlet 正则缓解）；混合不同分布族（不同 family、各 family 用几个）会带来选择问题，论文为简洁只用单一分布族[^src-graph-mixture-density-networks]。

## 与其它 MoE 方法的差异

论文未与任何具体 MoE 方法做实验对比，对比停留在 Sec 2 的概念层面与 Sec 3 的机制取舍：MoE = local experts + gating network，每 expert 一套完整网络；GMDN 沿 MDN 路线，共享一个 DGN 编码器，子网络（无 encoder、仅 Linear $\Phi_i$）输出各分布的参数，mixing weights 与分布参数都从同一 $h^{\mathcal V}_g$ 产生[^src-graph-mixture-density-networks]。论文没有 top-k 稀疏路由，也没有 load balancing loss；防塌缩用的是 EM 后验 + Dirichlet 先验，训练收敛性质由 GEM 的局部收敛保证给出。

[^src-graph-mixture-density-networks]: [[source-graph-mixture-density-networks]]
