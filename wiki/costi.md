---
title: "CoSTI"
type: technique
tags:
  - consistency-models
  - spatiotemporal-imputation
  - few-step-generation
  - generative-models
  - kbs-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 5
confidence: medium
status: active
---

# CoSTI (Consistency models for (a faster) Spatio-Temporal Imputation)

**CoSTI** 是 Solís-García、Vega-Márquez、Nepomuceno、Nepomuceno-Chamorro（University of Seville）发表于 Knowledge-Based Systems 327 (2025) 文章号 114117 的多变量时间序列插补（MTSI）模型，2025-07-21 在线发表，代码开源（github.com/javiersgjavi/CoSTI）[^src-costi]。论文将 [[consistency-models|Consistency Models]] 引入 MTSI：以 Consistency Training（CT）直接训练一致性模型，用 1–2 步采样替代扩散插补模型（CSDI、PriSTI、TIMBA）的 50–100 步迭代去噪；作者报告插补时间最多降低 98%，精度与扩散模型相当（摘要与 Sec. 6 口径）[^src-costi]。论文自述这是 Consistency Models 首次被适配到 MTSI 问题（Sec. 1 称 "the first adaptation of CMs to the MTSI problem"，Sec. 6 以 "To the best of our knowledge" 限定重申）[^src-costi]。

## 问题与定位

扩散插补模型（CSDI、PriSTI、TIMBA）精度高，但推理需要 T 步迭代去噪，论文据此认为其不适合 ICU 监护、交通控制等时间敏感场景（Sec. 1）[^src-costi]。一致性模型把 PF-ODE 轨迹上任意噪声水平的点直接映射到轨迹起点，单步即可采样；论文选择 Consistency Training 而非 Consistency Distillation，理由是 CD 需要先训练一个扩散模型作为中间步骤，而论文引近期工作称 CT 的性能可以达到并超过 CD 且省去该中间成本（Sec. 2，引 Song & Dhariwal 2024、Geng et al. 2024、Lu & Song 2024）[^src-costi]。

与一致性路线的其他成员相比，CoSTI 沿用原始 [[consistency-models|CM]] 的样本空间自洽映射；而 [[consistency-fm|Consistency-FM]] 与 [[loft|LOFT]] 把一致性约束施加在流匹配的速度场上[^src-yang-consistency-fm-arxiv24][^src-loft]。CoSTI 与这两篇无引用关系（其参考文献不含二者），属于同一"少步生成用于插补"方向的两条实现路径[^src-costi]。

## 一致性训练基础（Sec. 3.2）

CoSTI 的训练框架直接取自 Song et al. 的 CM 与 Karras et al. 的设计空间[^src-costi]：

- 噪声调度用 Karras scheduler：$\sigma_{\min}=0.002$、$\sigma_{\max}=80$、离散化指数 $\rho=7$（Sec. 3.2、Sec. 5.3）[^src-costi]。
- 一致性函数用 skip 参数化（论文式 3，沿 Song et al.）：$f_\theta(x,\sigma)=c_{\text{skip}}(\sigma)x+c_{\text{out}}(\sigma)F_\theta(x,\sigma)$，边界条件 $c_{\text{skip}}(\sigma_{\min})=1$、$c_{\text{out}}(\sigma_{\min})=0$，$\sigma_{\text{data}}=0.5$（式 4、Sec. 5.3）[^src-costi]。
- 训练目标为 consistency matching loss（式 5）：$\mathbb{E}[\lambda(\sigma_i)\, d(f_\theta(x_{\sigma_{i+1}},\sigma_{i+1}),\, f_{\theta^-}(x_{\sigma_i},\sigma_i))]$，教师网络 $f_{\theta^-}$ 与学生共享参数、前向经 stopgrad；距离 $d$ 用 Pseudo-Huber 度量（$c=0.00054d$），权重 $\lambda(\sigma)=1/(\sigma_{i+1}-\sigma_i)$；噪声级 $i$ 按离散 Lognormal 分布采样（$P_{\text{mean}}=-1.1$、$P_{\text{std}}=2.0$，式 6，引 [26]）[^src-costi]。

## MTSI 适配（Sec. 4.1）

论文在标准 CT 之上做了五项领域改造[^src-costi]：

1. **条件信息**（Sec. 4.1.1）：沿用论文所称"与 PriSTI 类似思路"的条件设计，一致性模型参数化为 $F_\theta:(X_{t,\sigma_i},\mathcal{X}_t,A,M_t,\sigma_i)\to\hat X_t$，其中 $\mathcal{X}_t$ 为线性插补（论文记号，区别于含缺失值的 $\tilde X_t$）、$A$ 为邻接矩阵、$M_t$ 为掩码[^src-costi]。
2. **损失位置**（Sec. 4.1.2）：损失只在有 ground truth 的位置计算；每个 batch 动态生成合成缺失值（得到 $\tilde X_t$ 与更新后的掩码 $\tilde M_t$）模拟插补场景[^src-costi]。
3. **正则化**（Sec. 4.1.2）：dropout 从其前作 TIMBA 的 0.1 提高到 0.2，作者称对 CM 的过拟合抑制更重要[^src-costi]。
4. **优化器**（Sec. 4.1.3）：采用 Scheduler-Free 的 AdamWScheduleFree（引 Defazio et al. 2024）加 weight decay（1e-6），替代 CM 文献常用的 RAdam 配置；消融显示其 MAE/MSE 低于 RAdam 与 AdamW+MultiStepLR（Table 12）[^src-costi]。
5. **课程学习**（Sec. 4.1.3）：训练中把离散化级数 $N$ 从 $s_0=10$ 线性增至 $s_1=200$，先比较 PF-ODE 上相距较远的点、再逐步细化；课程调度消融（Table 11）中该线性调度（MAE 1.76 / MSE 9.01）优于 Song & Dhariwal 的线性 $s_1=1280$（1.82/9.46）、原始 CM 调度 $s_0=2$（1.80/9.30）、常数（1.86/9.30）、指数（1.79/9.44）与预训练+指数（1.79/9.18）（Table 11 中 CoSTI 行数字与 METR-LA point 主结果相同，论文未另行注明该表的数据集，此处按数字对应归入该设置）[^src-costi]。

**确定性插补**（Sec. 4.1.4）：与扩散模型类似做概率采样，取 $N=100$ 次随机前向的逐元素中位数作为最终插补（式 7），依据是中位数对离群值的稳健性[^src-costi]。

## 架构（Sec. 4.2）

U-Net 骨干、双分支（主信号 + 条件信息），在 bottleneck 汇合[^src-costi]：

- **STFEM**（Spatio-Temporal Feature Extraction Module，Sec. 4.2.1）：扩展自 PriSTI 的条件特征提取模块，组合双向 Mamba block、空间自注意力 transformer 与 MPNN（引 Graph WaveNet）；两个 STFEM 分别处理主流（$X_t,A$）与条件流（$\mathcal{X}_t,M_t,A$）[^src-costi]。
- **压缩与 cross-attention**（Sec. 4.2.2）：按 "time-then-graph"（引 Gao & Ribeiro 2022）顺序以因子 $f_t,f_s$ 依次压缩时间与空间维度；条件分支用自注意力压缩，主分支在注意力块中用 cross-attention 融合条件序列（式 8：Q 来自主序列 $Z$，K/V 来自条件序列 $H$）[^src-costi]。
- **NEM**（Noise Extraction Module，Sec. 4.2.3）：模块本身源自 [[csdi|CSDI]]、经 PriSTI 与 TIMBA 增强；本文将其适配到一致性模型。每个 NEM 的构成为：带 cross-attention 的时间 transformer → 双向 Mamba block → 带 cross-attention 的空间 transformer → 带自注意力的空间 transformer → 门控 MLP；因 Mamba 不原生支持 cross-attention，论文保留 transformer 承担该功能。每个 NEM 输出两路：下一 NEM 的输入与噪声估计 $H_{\text{NEM}_i}$，并含 $\sigma$ 的位置嵌入[^src-costi]。
- **重建**（Sec. 4.2.4）：对噪声估计求和后经 U-Net 重建段恢复维度，用 skip connections 整合浅层特征[^src-costi]。

## 采样与复杂度（Sec. 4.3）

- 采样算法（Algorithm 2）支持任意步数；实验聚焦 1 步与 2 步（分别记 CoSTI 与 CoSTI-2）。1 步默认从 $\sigma_{i_1}=\sigma_{\max}=80$ 出发；2 步的第二噪声级 $\sigma_{i_2}$ 按数据集实验确定（Table 2，如 AQI-36 为 20.92、METR-LA point 为 0.821）[^src-costi]。
- 复杂度分析（Sec. 4.3）：以注意力为主导项，每层 $O(NL^2d+LN^2d)$（$L$ 为序列长、$N$ 节点数、$d$ 通道数），$B$ 个 block 的前向为 $O(B(NL^2d+LN^2d))$；扩散模型需乘迭代步数 $T$，CoSTI 单次前向，论文据此给出约 $\sim T$ 的理论推理加速[^src-costi]。
- 训练算法（Algorithm 1）在 $\sigma=\sigma_{\min}$ 时跳过一次前向以小幅加速[^src-costi]。

## 实验（作者报告，Sec. 5）

**设置**：6 个数据集——AQI-36（36 节点，原始缺失 13.24%）、METR-LA（207 节点，8.10%）、PEMS-BAY（325 节点，0.02%）、PhysioNet Challenge 2019（40,336 患者、40 变量，78.43%）、ETTh1（17,420 步、7 特征）、PEMS08（170 节点）（Sec. 5.1、Table 1）。AQI-36/METR-LA/PEMS-BAY 沿用 GRIN 基准（Cini et al.）的数据划分（70/10/20）与随机种子，邻接矩阵用阈值化高斯核；PhysioNet/ETTh1/PEMS08 用 80/10/10，PhysioNet 邻接矩阵由变量相关构造[^src-costi]。评测缺失场景（Table 4/5 口径）：AQI-36 24.6%、PhysioNet 83.82%、ETTh-1 25.14%、Pems08 24.99%（Table 4）；METR-LA block 16.6% / point 31.1%，PEMS-BAY block 9.2% / point 25.0%（Table 5）。训练时用 Point/Block/Historical/Hybrid 四种合成缺失策略（Sec. 5.2）；全部实验 5 次随机种子，硬件为 NVIDIA RTX A5000 24GB（Sec. 5.2、Appendix A）。基线中 CSDI/PriSTI/TIMBA 的数字转引自 TIMBA 论文（Solís-García et al., arXiv:2410.05916），PhysioNet Challenge 2019 除外（作者自行运行）（Sec. 5.2）[^src-costi]。

**推理时间**（Table 3，测试集整体，1 步，单位小时）：AQI-36 CoSTI 0.005 vs CSDI 0.22 / PriSTI 0.33 / TIMBA 0.44；METR-LA 0.06 vs 1.74/2.44/3.65；PEMS-BAY 0.16 vs 4.62/5.99/8.71；PhysioNet 0.48 vs 8.16/15.86/18.19；ETTh-1 0.007 vs 0.11/0.22/0.26；Pems08 0.03 vs 0.49/0.62/1.29。正文称扩散基线取 $T$：AQI-36 为 100、其余数据集为 50，并报告 METR-LA 上 CoSTI 仅用 TIMBA 时间的 1.64%、AQI-36 上为 0.91%（Sec. 5.4.1）[^src-costi]。

**精度**（Table 4/5，MAE/MSE，均作者报告）[^src-costi]：

- CoSTI 在 PhysioNet Challenge 2019（83.82% 缺失）上优于全部三个扩散基线：MAE 2.47 / MSE 388.31 vs TIMBA 3.11/521.29、PriSTI 3.58/573.06、CSDI 3.93/2282.94（论文将 CSDI 的高 MSE 归因于训练收敛失败，Sec. 5.4.1）。
- 其余设置多为略逊于至少一个扩散基线：AQI-36 CoSTI-2 MSE 358.67 优于 CSDI 388.37 / PriSTI 376.11，但 MAE 9.90 vs TIMBA 9.56；METR-LA point CoSTI 1.76 vs TIMBA 1.69；PEMS-BAY point CoSTI 0.64 vs CSDI/TIMBA 0.58。
- 论文自述 Pems08 上 CoSTI 结果最差（CoSTI 11.09 vs CSDI 9.87），作者归因于该数据集复杂的时空结构与大量孤立节点（Table 1 中 81 个），并提出更精细的邻接矩阵构造可能改善（Sec. 5.4.1）[^src-costi]。
- 增加采样步数（CoSTI → CoSTI-2）在多数设置改善精度，论文称这体现可控的速度-精度权衡（Sec. 5.4.1）[^src-costi]。

**GRIN 基准对比**（Table 6，沿用 Cini et al. 基准）：对比 Mean/DA/KNN/Lin-ITP/KF/MICE/VAR/TRMF/BATF/V-RIN/GP-VAE/rGAIN/MPGRU/BRITS/GRIN 等 15 个非扩散方法（其中 V-RIN/GP-VAE/rGAIN 在论文分类中属非扩散的生成式方法）。在该表五组设置中 CoSTI 的 MAE/MSE 均低于 GRIN（如 AQI-36 10.13/377.48 vs 12.08/523.14；PEMS-BAY point 0.64/1.53 vs 0.67/1.55）[^src-costi]。论文自述：CoSTI 在多数据集上匹配扩散模型的 MSE，但部分场景 MAE 略高（Sec. 5.4.2）；两个变体在所有基准上处于 Pareto 前沿（Fig. 4，论文表述）[^src-costi]。

**其余分析**：缺失率敏感性（METR-LA point 10–90%，Tables 7/8）中 CoSTI 接近 PriSTI/TIMBA（90% 时 MAE 2.44 vs 2.43/2.41）；下游节点值预测（Table 9，AQI-36，协议沿用其前作）中与扩散模型相当、在节点 31 上占优；架构消融（Table 10，CoSTI 行 1.76/9.01 与 METR-LA point 主结果相同、按数字对应归入该设置，论文表题仅写 "across datasets"）中去除条件头退化最严重（MAE 1.76→6.36、MSE 9.01→153.51），去除 STFEM 次之（2.33/16.32），去除 NEM 与自注意力影响较小（1.77/9.17、1.77/9.14）[^src-costi]。

**训练时间**（Table A.13）：CoSTI 的训练时间并无一致优势——METR-LA 上 79.42 h（CSDI 43.67、PriSTI 49.25、TIMBA 77.00），AQI-36 上 1.99 h（TIMBA 0.56），PhysioNet 上 12.92 h（CSDI 7.35）；仅 PEMS-BAY 上低于全部三个基线（108.75 vs CSDI 114.67、PriSTI 118.33、TIMBA 196.42），其余设置均不低于 CSDI。论文的加速主张限于推理侧[^src-costi]。

## 口径与注意

- "首次将 CM 用于 MTSI" 为论文自述（Introduction、Sec. 6），非第三方认定[^src-costi]。
- Table 4/5 中 CSDI/PriSTI/TIMBA 的数字是 CoSTI 论文转引自 TIMBA 论文的数字（Sec. 5.2），不是各原论文自己报告的数字，也与 [[giflow|GiFlow]] 等后续论文复测的数字无关；三套口径不可混用[^src-costi]。
- 正文（Sec. 5.4.1）称 AQI-36 上 CoSTI 用 TIMBA 时间的 0.91%，但按 Table 3 数字直接相除为 1.14%（0.005/0.44）；论文未给出该百分比的计算口径，两处分立记录。
- Table 3 与 Table A.13 在 PriSTI 的 PhysioNet 推理时间上相差 0.01 h（15.86 vs 15.87），论文内部不一致，无实质影响。
- [[fence|FENCE]] 论文相关工作节将 CoSTI 与 CSBI、[[mtsci|MTSCI]]、DSDI 并列为"改进一致性与推理速度"的扩散插补扩展，定性为"用一致性训练降低推理时间"[^src-fence]；[[giflow|GiFlow]] 把 CoSTI 列为基线，其 Table 7 的 CoSTI 推理时间（Air-36 0.37 min 等）为 GiFlow 在 A100 上的复测口径，非 CoSTI 原文数字[^src-giflow]。
- [[loft|LOFT]]（KDD 2026）将自身贡献定位为"一致性路线在时空插补中的应用此前未被探索"（论文表述，本 wiki 按原文记录于 [[loft]] 页）；该表述与 CoSTI（2025-07 已刊出、同样面向时空插补的一致性模型工作）之间存在口径张力，见 [[loft]] 页的注记。

## 论文自述局限（Sec. 6）

- 采用固定图结构表示空间依赖，动态/可学习图是未来方向。
- 训练稳定性随数据集与初始化变化，需要更好的课程策略或初始化方案。
- 未来工作：按数据集优化噪声调度与采样步数、研究 latent consistency models 以进一步加速训练；论文认为其形式可自然扩展到时序预测与异常检测[^src-costi]。

## 相关页面

- [[source-costi]] — 源文件摘要（raw/solis-garcia-costi-arxiv-2025.pdf）
- [[consistency-models]] — Consistency Models，CoSTI 训练框架的源头（Song et al., ICML 2023）
- [[csdi]] — CSDI，NEM 模块的原始出处，其基线数字被 CoSTI 转引
- [[pristi]] — PriSTI，条件信息设计与 STFEM 的参考来源
- [[grin]] — GRIN，CoSTI 沿用的基准（Cini et al.）及其对比方法
- [[giflow]] — GiFlow（ICML 2026），将 CoSTI 作为一致性插补基线的流匹配插补
- [[fence]] — FENCE（AAAI 2026），相关工作节提及 CoSTI 的扩散插补方法
- [[loft]] — LOFT（KDD 2026），同为少步插补但走流匹配速度一致性路线
- [[rdpi]] — RDPI（AAAI 2025），另一条时空插补加速路线（残差条件扩散），两文互未对比

[^src-costi]: [[source-costi]]
[^src-fence]: [[source-fence]]
[^src-giflow]: [[source-giflow]]
[^src-yang-consistency-fm-arxiv24]: [[source-yang-consistency-fm-arxiv24]]
[^src-loft]: [[source-loft]]
