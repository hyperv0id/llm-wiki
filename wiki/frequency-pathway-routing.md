---
title: "Frequency Pathway Routing"
type: technique
tags:
  - routing
  - frequency-domain
  - spatio-temporal
  - top-k
  - adaptive-computation
created: 2026-09-16
last_updated: 2026-09-16
source_count: 1
confidence: medium
status: active
---

# Frequency Pathway Routing

**Frequency pathway routing** 指在频率维上做输入条件化的稀疏选择：给每个节点对每个频带打一个相关性分，只激活得分最高的 $K$ 个频带所对应的分支，再按归一化权重融合各分支输出[^src-adafre]。[[adafre|AdaFre]] 用两段式结构实现它——一个路由模块决定"用哪些频率"，一个融合模块决定"各出多少力"[^src-adafre]。

## 为什么需要路由

论文的动机是频率分量的重要性随上下文变化：低频（通勤模式）在高峰时段占主导，高频（事故、封路）在平峰或特殊事件时更重要，而多数既有模型对所有频率做固定权重融合[^src-adafre]。经验依据是 Figure 1(b)——PeMSD4 单个传感器的频率能量在一天内变化，清晨低频主导、傍晚高频能量出现尖峰[^src-adafre]。

## 机制

三段结构[^src-adafre]：

1. **打分**：共享打分函数 $g_\phi(\cdot)$ 把输入映射为每节点对 $P$ 个频带的相关性分数

$$R=g_\phi(X)\in\mathbb{R}^{N\times P}$$

2. **概率化与稀疏选择**：带温度 $\tau$ 的 softmax 得到 $\hat{r}_{ip}$，在频率维取 top-$K$

$$\hat{r}_{ip}=\frac{\exp(r_{ip}/\tau)}{\sum_{j=1}^{P}\exp(r_{ij}/\tau)},\qquad \tilde{I}=\mathrm{TopK}(\hat{R},K)$$

论文式(10) 的分母写作 $\sum_{j=1}^{K}$，按该式"对频率分布做温度 softmax"的语义应为 $\sum_{j=1}^{P}$，见 [[source-adafre]] 的不一致记录[^src-adafre]。

3. **分支计算与加权融合**：被选频率各自进入对应 backbone 得到 $\hat{Y}^{(\tilde{I}_k)}$，再按 softmax 归一化权重聚合

$$\hat{Y}=\sum_{k=1}^{|\tilde{I}|}\alpha_k\cdot\hat{Y}^{(\tilde{I}_k)},\qquad \sum_{k=1}^{|\tilde{I}|}\alpha_k=1$$

路由粒度是**节点级**而非样本级：$R$ 的形状为 $N\times P$，每个节点独立选择自己的频带组合[^src-adafre]。

## 与相关路由机制的对照

| 机制 | 选择对象 | 选择规则 | 是否含均衡约束 |
|------|----------|----------|----------------|
| **Frequency pathway routing**（AdaFre） | $P$ 个时间频带 | 节点级温度 softmax + top-$K$ | 有（[[frequency-balance-loss]]） |
| [[ams-moe\|AMS-MoE]] | 多个时间尺度 expert | 噪声注入 top-$K$ | 论文以噪声注入促探索 |
| [[xcpd\|XCPD]] 的 DyMoE | 低/中/高三个频率 expert | 按累计概率阈值选 1–3 个（非固定 $K$） | 无 |
| [[sparse-balanced-mixture-of-experts-st\|MAGE 稀疏专家]] | 图学习专家 | 每节点 top-$K$ + 符号 SGD 负载均衡 | 有（显式平衡） |
| [[memory-augmented-gating\|TESTAM 记忆门控]] | 元节点库 | 记忆库检索式门控 | 无 |

共同点是把"用哪个子模块"变成输入相关的决策；AdaFre 的特点是把选择维度固定为时间频率，并与空间侧同频率的谱嵌入绑定[^src-adafre]。

## 使用边界

- 默认 $P=4$、$K=2$。论文在超参分析中报告输入长度 288 步与 $P=4$ 最优，但 $K$ 虽被列入"systematic study"的对象，正文与 Figure 3 标题都未给出 $K$ 的结论[^src-adafre]。
- 论文把计算开销的降低归因于稀疏选择——每样本只激活 $P$ 中的 $K$ 条路径、避开全局时空注意力与全图传播[^src-adafre]。Table 3 只报告了 AdaFre 与 STID/GWN/AGCRN/HimNet/STDN 的对照，正文声称的"相对 STAEFormer、PDFormer、STFGNN 计算资源显著更少"在该表中没有对应数据[^src-adafre]。
- 论文未报告路由选择的分布随时间或节点的可视化，因此"模型是否按论文假设选择频率"缺少直接证据[^src-adafre]。

## 相关页面

- [[adafre]] — 使用该路由的模型
- [[band-limited-temporal-decomposition]] — 被选择的对象如何构造
- [[frequency-specific-spatial-embedding]] — 与频带绑定的空间侧
- [[frequency-balance-loss]] — 抑制路由坍缩的正则项
- [[mixture-of-experts]] — 更一般的路由/专家框架
- [[source-adafre]] — 源文件摘要

[^src-adafre]: [[source-adafre]]
