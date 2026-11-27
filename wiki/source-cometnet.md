---
title: "CometNet: Contextual Motif-guided Long-term Time Series Forecasting"
type: source-summary
tags:
  - time-series-forecasting
  - motif-retrieval
  - mixture-of-experts
  - periodicity
  - aaai-2026
created: 2026-11-24
last_updated: 2026-11-27
source_count: 1
confidence: medium
status: active
---

# CometNet: Contextual Motif-guided Long-term Time Series Forecasting

主流长时序列预测模型靠固定长度回看窗工作：窗口太短看不到跨数千步的长期规律，拉长窗口则计算与显存开销快速增长，关键依赖还会被无关历史噪声淹没。CometNet 绕开窗口本身：真实序列常由跨度数千步、反复重现的代表性长上下文模式（论文称 contextual motifs）支配；它离线从整条训练序列抽取少量主导母题建库，在线让 96 步短窗检索最相关母题并叠加自身在母题生命周期中的相对相位，在统一短窗下借入长程上下文[^src-cometnet]。

## 核心做法

模型全部由标准信号处理与门控线性模块拼装，分离线建库与在线预测两段[^src-cometnet]：

- **离线母题库抽取**：对训练序列做滑动均值去趋势，用 FFT 找主导周期作为多尺度集合；各尺度下采样后按 Pearson 相关聚类出候选母题，用 DTW 相似度建图取连通分量完成跨尺度去重，最后按质量分与多样性贪心选出少量主导母题[^src-cometnet]。
- **在线门控预测**：短回看窗经 MLP 编码后接双头门控——softmax 路由头在各母题专家间分配权重（无稀疏化的稠密专家混合），并行的 sigmoid 位置头估计当前窗口在对应母题生命周期中的相对相位，各专家预测按路由权重加权求和输出[^src-cometnet]。

## 证据

1. **定位与主结论**：统一 96 步回看窗下 8 个基准中 7 个平均 MSE/MAE 第一，优势随预测步长拉大[^src-cometnet]。
2. **真实涨点归因**：消融仅覆盖低维 ETT 的 $H=720$，禁用全部母题组件掉点最狠（ETTh1 误差大幅上升），去门控与位置头次之[^src-cometnet]。
3. **证据缺口**：862 维 Traffic 逊于 TimeMixer++，且建库成本与跨数据集迁移未评测[^src-cometnet]。

## 相关页面

- [[gtr|GTR]] — 同槽位互斥：可学习绝对相位周期矩阵检索 vs 离线 DTW 离散母题内容门控。
- [[contextual-pattern-bank|Contextual Pattern Bank]] — 同构对照：参数化增量模式库 vs 非参数聚类母题库。
- [[patchtst|PatchTST]] — 瓶颈接力 $\Delta$：窗内局部 patch 编码受感受野约束，CometNet 挂载窗外离线母题做长程外推。
- [[mixture-of-experts|Mixture of Experts]] — 架构特例：以离线母题原型作稠密专家，并融合生命周期相位信号。

来源：Weixu Wang, Xiaobo Zhou, Xin Qiao, Lei Wang, Tie Qiu（天津大学），AAAI 2026；扩展版 arXiv:2511.08049。

[^src-cometnet]: [[source-cometnet]]
