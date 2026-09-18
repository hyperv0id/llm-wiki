---
title: "FlowNet: Modeling Dynamic Spatio-Temporal Systems via Flow Propagation"
type: source-summary
tags:
  - spatio-temporal
  - forecasting
  - mixture-of-experts
  - flow-propagation
  - neurips-2025
last_updated: 2026-09-17
source_count: 1
confidence: medium
status: active
---

# FlowNet: Modeling Dynamic Spatio-Temporal Systems via Flow Propagation

**Authors**: Yutong Feng¹、Xu Liu²、Yutong Xia²、Yuxuan Liang¹（通讯，yuxliang@outlook.com）。¹香港科技大学（广州），²新加坡国立大学。[^src-flownet]
**Venue**: NeurIPS 2025（页眉 "39th Conference on Neural Information Processing Systems (NeurIPS 2025)"）。[^src-flownet]
**证据源**: `/run/media/jcheng/WD-Data/yjs/Zotero/storage/UTYX5IA4/.zotero-ft-cache`（PDF 全文纯文本缓存）

## 问题

论文认为现有 STGNN 与 attention 方法依赖 similarity-driven connectivity，把表面统计相关当作系统演化机制，忽略了支配动态系统的非对称 flow 交换（Figure 1 的城市案例：相似观测只是 flow 的症状而非驱动）；且 flow 传播有双重异构性——早晚高峰流向相反、节假日目的地随功能切换[^src-flownet]。据此提出 Spatio-Temporal Flow 范式，三条设定：flow tokens $\varphi\in\mathbb{R}^d$ 是可量化的信息载体；信息传递服从源-目的守恒律（源 token 减少、目的 token 等比例增加）；每个节点的交互邻域由可学习、上下文感知的半径决定[^src-flownet]。

## 方法

- **Preprocess**：滑窗切 patch，$X'_p=\mathrm{Partition}(X;P,S)\in\mathbb{R}^{N\times P\times M}$，再经可学习嵌入 $X_p=X'_p W_e+b_e\in\mathbb{R}^{N\times M\times d}$；patching 使 token 数按 $S$ 倍缩减[^src-flownet]。
- **ASM（Adaptive Spatial Masking）**：对每节点每 patch 预测感知半径 $r_i^t=\mathrm{Softplus}(\tilde{X}_i^t W_h+b_e)$，随后构造时变 mask $M^t[i,j]=\mathrm{Sigmoid}(r_i^t-d_{ij})$，可微、可反传；距离度量依系统选择（城市用 Manhattan，自然系统用 Euclidean）[^src-flownet]。
- **FAM（Flow Allocation Module）**：两个结构相同、参数不同的 FEM（Causal Temporal MSA，channel-independent）分别估计节点保留的 $\hat{\Phi}_o$ 与待分配的 $\hat{\Phi}_a$；再用 origin/destination 向量加 head-folded 静态嵌入计算 flow logit $q_{ij}^t=\alpha\cdot(\hat{O}_i^t)^\top \hat{D}_j^t\cdot M^t[i,j]$（$\alpha=1/\sqrt{d'}$），在邻域内 softmax 归一化得分配矩阵 $\Lambda^t$，最终 $\hat{\Phi}^t=\hat{\Phi}_o^t+(\Lambda^t-I)\hat{\Phi}_a^t$，显式实现源减少与目的增加[^src-flownet]。
- **级联与 M-MLP**：模块堆叠用 hyper-connection（depth-connection 加权模块间连接、width-connection 支持同层信息交换）替代普通残差；FAM 之间插入 MLP 融合多头表示。关键改动（缓存行 183 原句）：「We replace vanilla linear layers with a Mixture of Linears (MoL). Unlike Mixture of Experts (MoE), which treats entire blocks as experts, MoL treats each linear projection as an independent expert. For an MLP with $L$ layers, this design creates a combinatorial parameter space of $L\times E$ distinct linear transformations compared to only $E$ transformations in equivalently sized MLP-MoE designs.」即 expert 是单个线性映射而非整个 block，$L$ 层 MLP 换 MoL 后参数组合空间从 MLP-MoE 的 $E$ 个变换扩大到 $L\times E$ 个；实现为「16 experts inside the M-MLP」（Figure 2 标题将 M-MLP 全称为 Mixed Multi-Layer Perceptron），MoL 线性层用 Kaiming 初始化、GeLU 激活。论文未描述 MoL 的 gating/路由机制与负载均衡（全文无 gating、router、load balancing、auxiliary loss 等表述，缓存中查不到）[^src-flownet]。
- **训练**：MAE 损失，Adam，batch size 8，lr $1\times 10^{-3}$ 每 20 epoch 减半至第 60 epoch，第 20 epoch 起早停；堆叠 2 层 FAM、flow token 4 heads、hidden 维度 64，A100 80GB[^src-flownet]。

## 实验

数据集：PEMS04F（Caltrans 交通流，307 节点，5 min）、DeepBase（美国 1661 个流域逐日 baseflow）、SINPA（新加坡 1687 个停车场 15 min 空位）；短/长期任务分别为 12 步/288 步（PEMS04F、SINPA）与 32 步/360 步（DeepBase）；z-score 标准化；Table 1 报告 5 个随机种子平均的 MAE/RMSE[^src-flownet]。作者报告 FlowNet 在 3 数据集 × 2 任务 × 2 指标全部领先，带 ∗ 表示对 second-best 在 0.05 水平显著：PEMS04F 短期 18.48∗/29.03（次优 STAEformer 18.73/29.29）、长期 22.79∗/38.21∗（次优 SCINet 24.46/41.27）；DeepBase 短期 0.43∗/0.93∗（次优 GWNET 0.49/1.02）、长期 0.50/1.08（次优 STGCN 0.51/1.10，未标显著）；SINPA 短期 39.03∗/76.04∗（次优 GWNET 59.04/103.75）、长期 31.39/59.06（次优 STGCN 33.41/62.86，未标显著）；STAEformer 在 DeepBase 与 SINPA 长期任务 OOM[^src-flownet]。消融（Table 2，PEMS04F，Δ 为变差幅度）：去掉 allocation flow 影响最大（短期 MAE Δ5.54%、长期 Δ4.90%）；去掉 retained flow 短期影响最小（Δ0.08%）而长期较大（Δ1.32%）；破坏守恒律短期 Δ0.24% 大于去 retained、长期 Δ0.83% 小于去 retained——论文据此推断短期系统近似封闭、长期需考虑信息总量波动[^src-flownet]。效率（Figure 3）：FlowNet 训练时间与显存介于 Transformer 与 STGNN 类方法之间（PEMS04F 短期 65.74 s/1.50 GB；DeepBase 短期 434.00 s/23.70 GB，低于 STTN 934.39 s/63.85 GB）[^src-flownet]。

## 局限

Appendix E（论文自述）：FlowNet 的计算效率受 pairwise 操作约束——Flow Allocation 因逐节点 flow 再分配达到 $O(N^2)$，FEM 随预测 horizon 平方增长（$O(T^2)$）；在节点规模大或长期预测场景下效率低于 STGNN 类方法，但速度与显存均优于 Transformer-based 模型；作者认为精度收益在许多现实应用中值得此 trade-off，future work 将探索 sparse flow tokenization 与 hierarchical grouping 缓解瓶颈[^src-flownet]。（缓存该句在 "We argue that the accuracy" 处因页界截断，后半句 "gains justify this trade-off…" 经同目录 PDF 第 23–24 页核对补全。）[^src-flownet]

## 与其它 MoE 方法的差异

论文在方法一节把 MoL 与 MoE 并列对照：MoE 把整个 block 当 expert，MoL 把每个线性投影当独立 expert，并给出 $L\times E$ 对 $E$ 的参数空间比较；related work 引用了 Moirai-MoE（sparse MoE 时序基础模型）作为背景。但论文未做与任何 MoE 模型的实验对比，也未描述 MoL 自身的 gating/路由与负载均衡，均属论文未提及[^src-flownet]。

[^src-flownet]: [[source-flownet]]
