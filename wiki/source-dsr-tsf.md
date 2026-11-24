---
title: "DSR-TSF: Spectrum-Driven Dynamic Routing for Efficient Long-Horizon Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - periodicity
  - dynamic-routing
  - low-rank
  - lightweight
  - neurips-2026
created: 2026-11-24
last_updated: 2026-11-24
source_count: 1
confidence: medium
status: active
---

# DSR-TSF: Spectrum-Driven Dynamic Routing for Efficient Long-Horizon Time Series Forecasting

**来源**：Hua Wang、Bing Li（鲁东大学信息与人工智能学院）；Fan Zhang（山东工商学院计算机学院）；NeurIPS 2026[^src-dsr-tsf]。

[[sparsetsf|SparseTSF]] 假设时间序列的周期固定不变，按先验周期下采样重排后用共享线性头跨段预测；但同一变量（如 ETTh1 的油温通道）在不同历史窗口内的主导周期会发生漂移，导致固定周期步长与当前频谱结构错配。DSR-TSF 不做重采样或逐窗周期对齐，而是先通过快速傅里叶变换提取当前窗口的主导频谱特征作为门控信号，软加权多条可学习的时域周期响应分支对输入做自适应残差修正，随后再交回原有的分段重排与共享低秩线性或 MLP 头完成预测[^src-dsr-tsf]。

## 去黑话算子配方与路线差异

模型未引入改变复杂度的底层新算子，而是在 SparseTSF 的「周期分段 + 共享头」骨架前拼接谱条件软路由与多分支时域响应，并对共享头施加低秩约束[^src-dsr-tsf]：

- **频谱描述符与动态门控**：对去均值输入 $\bar{x}$ 做 `rFFT` 并去掉直流分量后，截取前 $K$ 个主导频率的对数幅值与归一化频率位置构成特征向量 $f=[\log(1+a_k),\,\hat{q}_k]_{k\le K}$，经线性投影并与可学习静态先验 $\alpha$ 相加后做温度 `softmax`，得到全部 $R$ 条分支的路由权重 $w = \operatorname{softmax}((\alpha + \lambda g(f))/\tau)$[^src-dsr-tsf]。
- **多分支残差响应与低秩头**：将输入与加权后的多条可学习时域周期响应基（含可选相位调制算子 $S_{\phi_r}$）做残差叠加，即 $\tilde{x} = \bar{x} + \sum_{r\le R} w_r S_{\phi_r}(B_r(\bar{x}))$，再按先验周期 $P$ 分段重排并送入跨段位共享的低秩 Linear 或 MLP 头输出预测[^src-dsr-tsf]。相对 SparseTSF（ETTh1 Linear 版），参数量由 $0.45\text{K}$ 增至 $0.90\text{K}$，MACs 由 $6.29\text{M}$ 增至 $20.22\text{M}$，推理延迟由 $0.35\text{ ms}$ 增至 $1.92\text{ ms}$；MLP 版受低秩头约束影响，MACs 由 $34.31\text{M}$ 降至 $23.40\text{M}$[^src-dsr-tsf]。
- **有向关系对照**：相对 [[sparsetsf|SparseTSF]] 与 [[cross-period-sparse-forecasting|Cross-Period Sparse Forecasting]] 属于**瓶颈接力 $\Delta$ 与复现对齐**（从固定周期单分支推进至谱驱动多分支残差与低秩共享头，但 Traffic 复现口径存在正文与主表矛盾）[^src-dsr-tsf]；与 [[frequency-pathway-routing|Frequency Pathway Routing]] 构成**同构对比**（同以频域信号条件门控，但 DSR-TSF 路由对象为时域响应基而非独立频带且无均衡损失）[^src-dsr-tsf]；属于 [[mixture-of-experts|Mixture of Experts]] 的**软 MoE 特例**（静态先验加频谱门控全选加权，无稀疏截断）[^src-dsr-tsf]；与 [[cyclenet|CycleNet]]、[[hybrid-periodicity-decoupling|Hybrid Periodic Decoupling]] 属于**跨槽位正交**（全局可学习周期表查表残差剥离 vs 单窗隐式谱驱动多分支响应修正）[^src-dsr-tsf]。

## 证据

1. **定位与主结论**：7 个 LTSF 集相对 SparseTSF 平均降 MSE 约 $2.37\%$（ETT 降 $0.40\%\text{–}4.24\%$），在 $0.9\text{K–}1.5\text{K}$ 参数槽位逼近强基线（ETTh1 $0.393$ 对 MixLinear $0.395$）[^src-dsr-tsf]。
2. **真实涨点归因**：核尺寸对齐变体（$0.433$）持平 SparseTSF（$0.432$），排除感受野解释；消融中去低秩头（ETTh1 $0.393\to 0.404$）退化略重于改静态路由（$0.393\to 0.402$），四 ETT 集顺序一致，低秩头贡献不低于动态路由[^src-dsr-tsf]。
3. **证据缺口与负结果**：正文称 Traffic「仍有提升」，但 Table 1 实测 MLP 版（$0.403$）劣于 SparseTSF/MLP（$0.396$）、Linear（$0.412$）仅持平且 MAE 落后（$0.274$ vs $0.264$）；超参 $(K, R, \tau)$ 敏感性正文缺位[^src-dsr-tsf]。

## 相关页面

- [[sparsetsf]] · [[cross-period-sparse-forecasting]] — 瓶颈接力 $\Delta$ 与 Traffic 复现口径矛盾对齐
- [[frequency-pathway-routing]] — 频域条件软路由选通时域基 vs 稀疏路由频带（同构对比）
- [[mixture-of-experts]] — 静态偏置加频谱全选加权前置分支（软 MoE 架构特例）
- [[cyclenet]] · [[hybrid-periodicity-decoupling]] — 显式全局周期查表 vs 隐式单窗谱驱动响应（跨槽位正交）

[^src-dsr-tsf]: [[source-dsr-tsf]]
