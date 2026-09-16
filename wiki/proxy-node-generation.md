---
title: "Proxy Node Generation (GPN)"
type: technique
tags:
  - federated-learning
  - traffic-forecasting
  - attention
  - auxiliary-nodes
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Proxy Node Generation (GPN)

代理节点是本地生成的 $N$ 个虚拟节点，用来顶替因数据隔离而拿不到的跨区域节点[^src-fedhint]。

## 生成过程

给定局部历史 $X_t\in\mathbb{R}^{|V_m|\times T_1}$[^src-fedhint]：

1. 映射 key/value：$K_t=X_tW_K$，$V_t=X_tW_V$，$W_K,W_V\in\mathbb{R}^{T_1\times d_{att}}$。
2. 对 $K_t$、$V_t$ 各做一次 [[time-shifted-filters|time-shifted filter]]，得 $\tilde{K}_t$、$\tilde{V}_t$。
3. 共享 query 做注意力：$P_t=\mathrm{softmax}(Q\tilde{K}_t^\top/\sqrt{d_{att}})\tilde{V}_t\in\mathbb{R}^{N\times d_{att}}$，$Q\in\mathbb{R}^{N\times d_{att}}$ 跨客户端共享。
4. 投影回时域：$\mathbf{X}^{proxy}_t=P_tW_p$，$W_p\in\mathbb{R}^{d_{att}\times T_1}$。

与局部节点纵向拼接得 $X^{all}_t\in\mathbb{R}^{(|V_m|+N)\times T_1}$[^src-fedhint]。

## 在模型中的位置

空间编码器是 AGCRN：自适应邻接 $\tilde{A}=I+\sigma(EE^\top)$，$E\in\mathbb{R}^{(|V_m|+N)\times e}$，配 GRU 门控[^src-fedhint]。编码器分两支，各带一张掩码，避免全局信息与区域特有信息互相干扰[^src-fedhint]：

- **global encoder**：$M^{global}$ 只把「局部节点 ↔ 代理节点」置 1，输出 $H^{global}_t$；
- **local encoder**：$M^{local}$ 只把「局部节点 ↔ 局部节点」置 1，输出 $H^{local}_t$。

拼接后经线性层预测全部 $|V_m|+N$ 个节点，取前 $|V_m|$ 行为本地预测：$\hat{X}^{all}_t=[H^{global}_t\;\|\;H^{local}_t]W_o$[^src-fedhint]。聚合时这一整块（隐藏成分提取器 + global encoder）算 shared 参数，见 [[personalized-aggregation-strategy|PAS]]。

## 实证

去掉代理节点（w/o GPN）后四个数据集全部掉点：PEMS03 11.95→12.70、PEMS04 16.11→17.11、PEMS07 17.19→18.13、PEMS08 12.38→13.07（MAE）。PEMS03/04/07 上这是四个消融项里降幅最大的一项，PEMS08 上略小于去掉 global encoder[^src-fedhint]。$N=64$、$d_{att}=32$ 在 PEMS03 上最优[^src-fedhint]。

## 与虚拟节点的区别

[[virtual-nodes-traffic|Virtual Nodes]] 同样往图上追加辅助节点，但目的是缓解长期预测的 over-squashing：它们连接全部真实节点，权重由邻接矩阵学。代理节点不连真实节点、不进入物理拓扑，只由注意力从本地数据生成[^src-fedhint]。

## 相关页面

- [[hidden-global-components|Hidden Global Components]]
- [[time-shifted-filters|Time-Shifted Filters]]
- [[inter-client-dependency|Inter-Client Dependency]]
- [[virtual-nodes-traffic|Virtual Nodes]]

[^src-fedhint]: [[source-fedhint]]
