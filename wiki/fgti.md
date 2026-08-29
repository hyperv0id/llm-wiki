---
title: "FGTI"
type: technique
tags:
  - diffusion-models
  - frequency-domain
  - data-imputation
  - time-series
  - conditional-diffusion
  - neurips-2024
created: 2026-08-29
last_updated: 2026-08-29
source_count: 7
confidence: medium
status: active
---

# FGTI（Frequency-aware Generative Models for Multivariate Time Series Imputation）

**FGTI** 是 Xinyu Yang、Yu Sun、Xiaojie Yuan（南开大学）与 Xinyang Chen（哈工大深圳）发表于 NeurIPS 2024 的多变量时间序列插补模型[^src-fgti]。raw 文件 `raw/yang-fgti-neurips-2024.pdf` 为官方 proceedings 排版：每页页脚 "38th Conference on Neural Information Processing Systems (NeurIPS 2024)"，文末含 NeurIPS Paper Checklist（用户著录的 NeurIPS 2024 官方 proceedings 已在 PDF 内核实）[^src-fgti]。论文把插补建模为条件生成问题，在时域观测条件之外，用 high-frequency filter 与 dominant-frequency filter 从观测序列提取两组频域条件，经 cross-attention 融合进 DDPM 式去噪网络[^src-fgti]。代码与数据见 github.com/FGTI2024/FGTI24（论文参考文献 [1]）[^src-fgti]。

## 问题：插补误差主要落在残差项

论文的出发点是一个误差分解观察：多变量时间序列可分解为 trend、seasonal、residual 三项（论文引 STL 与 TIDER），而现有插补方法主要针对前两项优化[^src-fgti]。作者在预分解的 KDD 数据集（10% 缺失）上做了一个调查实验（论文 Figure 1），报告代表性插补方法的误差主要由残差项贡献，并据此称残差项是"被低估"的建模对象（论文自述）[^src-fgti]。

动机链条有两环[^src-fgti]：

1. 残差项与高频分量密切相关（论文引 van Driel et al. 2021、FEDformer、Liu & Cheng 2023）。
2. 深度网络对高频信息的建模泛化能力差——论文引 Rahaman et al. 2019（spectral bias）与 Tancik et al. 2020（Fourier features）支持这一前提。

由于生成式插补模型（VAE/GAN/扩散）默认只以时域观测值作为条件，论文主张把频域信息直接加入条件 C，即 C = 时域观测条件 X_C + 高频条件 C_H + 主频条件 C_D[^src-fgti]。

## 机制

### 两组频域条件（论文 Sec 3.1）

- **High-frequency filter**（Sec 3.1.1）：逐属性处理。先对每条属性序列 X_d 插值，再作 FFT 得到幅值向量 A，丢弃截止阈值以下的频率分量、保留其余，经 IFFT 回时域得到 C_H_d，按属性拼接为 C_H ∈ R^{D×L}；整体复杂度 O(DL log L)[^src-fgti]。
- **Dominant-frequency filter**（Sec 3.1.2）：取幅值最大的 top-κ 个频率分量（式 4），IFFT 回时域得 C_D ∈ R^{D×L}；论文将其作用表述为提供趋势/季节项的"背景结构信息"，并缓解高频条件对趋势/季节项插补的干扰[^src-fgti]。两个 filter 复杂度同为 O(DL log L)[^src-fgti]。

### 跨域表示学习（论文 Sec 3.2）

一个 Transformer 编码器（附录 A.3.1：位置编码层 + transformer encoder layer）将 Concat(C_H, C_D) 映射为频域表示 C_F ∈ R^{D×L×K}[^src-fgti]。随后两个 cross-attention 模块把 C_F 与生成模型的时域隐表示融合：

- **Time-frequency representation learning**（式 8）：按属性分段做 cross-attention，Q、K 由频域表示 C_F 投影得到，V 由时域隐表示 R_in 投影得到，输出按属性拼接为 R_t（式 9）[^src-fgti]。
- **Attribute-frequency representation learning**（式 10）：按时间戳分段再做一次 cross-attention，Q、K 同样来自 C_F，V 来自时频表示 R_t，输出 R_a（式 11）[^src-fgti]。

结构注记（wiki 层面对照）：两个模块中注意力权重都由干净的频域条件决定、被聚合内容来自当前（含噪）时域表示。这一"条件算注意力、输入供内容"的分工与 [[pristi|PriSTI]] 的先验引导注意力（Q,K←干净先验 H_pri、V←混合输入 H_in）同构，属于把干净条件与含噪输入分离的同一设计家族。

### 频率感知扩散（论文 Sec 3.3）

- 前向过程为标准 DDPM（式 12-13）；反向过程条件化为 (X_C, C_H, C_D)（式 14），去噪网络按 DDPM/classifier-free guidance 形式做 ε 预测参数化[^src-fgti]。
- **命题 3.1**：H(X̂_{t−1} | X̂_t, X_C, C_H, C_D) < H(X̂_{t−1} | X̂_t, X_C)。附录 A.1 用条件熵链式法则证明；论文以此论证加入两组频域条件能降低反向过程的不确定性[^src-fgti]。
- 训练目标（式 15）：L_θ = E‖ε − ε_θ(t, X̂_t, X_C, C_H, C_D)‖²；训练时每步随机选部分观测值作插补目标、其余作观测条件（与 [[csdi|CSDI]] 的自监督掩码训练同一路线，见 [[self-supervised-imputation-training]]）[^src-fgti]。
- 掩码策略默认值来自附录 A.4.3 的实验：random ratio mask（Table 3）与 random missing pattern（Table 4）[^src-fgti]。
- 版式注记：附录图 7 的去噪网络结构图中绘有 "MoE Module" 字样，但正文未说明该模块的作用与配置（如实记录，不以图推正文）[^src-fgti]。

## 证据（作者报告）

### 主实验（论文 Sec 4.2，Table 1）

三个含真实缺失的数据集[^src-fgti]：

- KDD（BRITS 论文数据集，论文口径）：北京 9 个站点、逐小时、2017-01-30 至 2018-01-31、8,034 条气象与空气质量读数、每站 11 项传感器读数、真实缺失 4.46%。
- Guangzhou：214 条匿名道路、每 10 分钟车速、2016-08-01 至 2016-09-30、真实缺失 1.29%。
- PhysioNet：11,988 名患者 ICU 入院 48 小时内的 37 项测量、缺失 79.71%（1,707 名患者 48 小时后死亡，用作下游死亡率预测标签）。

设置：15 个基线（Mean、BTMF、TIDER、BRITS、TST、SAITS、TimesNet、LaST、FreTS、GRIN、TimeCIB、GAIN、CSDI、SSSD、PriSTI）；MCAR 机制下按观测值计 10/20/30/40% 缺失率，5 次重复取平均；GRIN、PriSTI 等需要邻接矩阵的方法默认用单位阵；LaST、FreTS 按 TimesNet 设置改造为插补任务[^src-fgti]。评估口径注记：对比实验与模型分析忽略数据集的真实缺失值（无真值可评），仅应用研究保留它们（论文 Sec 4.1.1）[^src-fgti]。

作者报告 FGTI 在各缺失率下取得最佳插补精度（论文表述 "achieves the best imputation accuracy under various missing rates"、"surpasses the state-of-the-art generative imputation models"）[^src-fgti]。RMSE/MAE 节选（Table 1）[^src-fgti]：

| 设置 | FGTI | CSDI | PriSTI | SSSD |
|------|------|------|--------|------|
| KDD 10% | 0.406 / 0.149 | 0.459 / 0.177 | 0.472 / 0.169 | 0.697 / 0.397 |
| KDD 40% | 0.478 / 0.205 | 0.569 / 0.220 | 0.581 / 0.217 | 0.883 / 0.603 |
| Guangzhou 40% | 0.356 / 0.254 | 0.439 / 0.283 | 0.650 / 0.381 | 0.807 / 0.444 |
| PhysioNet 10% | 0.580 / 0.286 | 0.619 / 0.310 | 0.652 / 0.369 | 0.875 / 0.528 |
| PhysioNet 40% | 0.669 / 0.376 | 0.705 / 0.395 | 0.679 / 0.406 | 0.983 / 0.729 |

如实记录的一处并列：Guangzhou 10% 的 MAE 上 FGTI 与 PriSTI 同为 0.170（Table 1），FGTI 的 RMSE（0.230 vs 0.242）更低[^src-fgti]。

缺失机制扩展（Figure 4 与附录图 8-9）：在 KDD/Guangzhou/PhysioNet 上以 10% 缺失补测 MAR 与 MNAR（MAR 在 KDD 中与低温读数相关，MNAR 与特征读数偏低相关），论文报告各机制下结果"相对接近"且 FGTI 一致最优[^src-fgti]。

### CRPS 对比（附录 A.5，Tables 5-6）

在生成式基线（TimeCIB、GAIN、CSDI、SSSD、PriSTI）上按 [[crps|CRPS]] 评估分布质量：KDD 10% 下 FGTI 0.158，对比 CSDI 0.224、PriSTI 0.232、SSSD 0.352、TimeCIB 0.466、GAIN 0.709；论文报告所有缺失率与机制组合下 FGTI 的 CRPS 均最低，并将其作为命题 3.1 的经验证据[^src-fgti]。

### 消融（论文 Sec 4.3，Table 2）

四个变体（w/o Cross-domain、w/o Frequency condition、w/o Dominant-frequency filter、w/o High-frequency filter）在三个数据集上均劣于完整模型。PhysioNet 上去掉 high-frequency filter 的退化最大（RMSE 0.7294 vs 完整 0.5801），与"残差项依赖高频条件"的动机一致；KDD 上四个变体的 RMSE 差距较小（0.4128-0.4210 vs 0.4057）[^src-fgti]。

### 分项案例研究（附录 A.5.1，Table 7）

对 STL 分解后的 KDD 三项分别掩蔽、分别插补（10% MCAR）：趋势项在去掉高频条件后最好（论文表述：高频条件可能干扰趋势项插补）；季节项结论类似且主频条件的贡献小于趋势项；残差项主要对应高频条件——去掉主频滤波的 RMSE 0.4956 低于去掉高频滤波的 0.5129[^src-fgti]。口径注记：分项最优变体并非完整 FGTI（完整模型残差项 RMSE 0.5068），分项最优与整体最优是不同口径，论文亦未主张完整模型在单项上最优[^src-fgti]。

### 下游应用（论文 Sec 4.5，Figure 6）

- 空气质量预测：先插补 KDD 真实缺失，再用前 12 小时记录以 AdaBoost 回归器预测未来 6 小时平均 PM2.5（RMSE）；作者报告 FGTI 取得最高改善[^src-fgti]。
- 死亡率预测：对 PhysioNet 插补后训练 MLP 分类器（AUC）；作者报告 FGTI 最佳，且各插补方法相对不插补均有可观改善[^src-fgti]。

### 资源消耗（论文 Sec 4.4，Figure 5）

KDD 10% 设置下，FGTI 的运行时间与其他扩散方法处于同一量级，整体资源消耗略高于 CSDI（作者归因于两组频域条件的引入）；论文自述的局限即此项资源开销（NeurIPS checklist 第 2 问将局限性指向 Sec 4.4）[^src-fgti]。

## 超参数（附录 A.4.2）

截止频率 F=0.3、主频个数 κ=10 由敏感性实验选定（图 10-11）：F 过小混入过多低频、过大则高频信息不足；κ 过小得不到足够平滑信息、过大则高频混入主频条件[^src-fgti]。扩散相关超参沿用 CSDI 与 DiffWave 的推荐设置[^src-fgti]。

## 与其他方法的关系

- [[csdi|CSDI]]：FGTI 沿用条件扩散 + 自监督掩码训练框架，把条件从"观测值"扩展为"观测值 + 两组频域条件"；论文 Table 1 报告在三个数据集的 MCAR 设置下优于 CSDI[^src-fgti]。
- [[pristi|PriSTI]]：条件来源对照——PriSTI 以线性插值作为增强条件[^src-pristi]，FGTI 以频域条件作为增强条件；论文 A.5.2 自述其高频/主频信息"优于 PriSTI 使用的插值信息"（论文措辞），可视化（图 12-13）中将 CSDI 在快变点的不准确归因于条件不足[^src-fgti]。注意 FGTI 给 PriSTI 配的是单位阵邻接矩阵（Sec 4.1.3）[^src-fgti]。
- [[lscd|LSCD]]：LSCD（ICML 2025）的条件熵论证在 [[lscd]] 页记录为引自 "Yang et al. (2024a)" 的频率条件扩散理论，其不等式结构与本文命题 3.1 一致；因仓库内无 LSCD raw PDF，该引用与本文的对应关系未在 LSCD 参考文献层面核实。方法差异：LSCD 主张对含缺失序列先插值再 FFT 会扭曲频谱、改用可微 Lomb–Scargle 周期图[^src-lscd]，而 FGTI 的两个 filter 都是先插值再 FFT[^src-fgti]——两种频谱来源口径并存。
- [[loft|LOFT]]（KDD 2026）：流匹配 + 低秩先验插补，将 FGTI 列入 50-NFE 扩散基线在 PEMS 交通数据上对比，该表数字为 LOFT 复现口径[^src-loft]；FGTI 原文评测设置为 KDD/Guangzhou/PhysioNet 的 MCAR 10-40%[^src-fgti]，两套口径不可混用。
- [[frequency-aware-conditioning|频率感知条件化]]：FGTI 是该范式在时序插补任务上的实例——频域滤波提取条件信号、编码后经 cross-attention 注入生成网络[^src-fgti]；与图像生成中的 [[freqflow|FreqFlow]] 同族但任务与注入方式不同（FreqFlow 侧对照见 [[frequency-aware-conditioning]] 页）。
- [[frequency-diffusion|频域扩散]] / [[frequency-based-noise-control|频域噪声控制]]：改前向噪声频谱 vs 改条件信号，是频域介入扩散模型的两个不同位置，FGTI 属后者[^src-fgti]。
- [[fourier-imputation-loss|FIL]]：ImputeFormer 的 Fourier loss 对插补输出的频谱做 ℓ1 稀疏正则[^src-2312-01728]；FGTI 则从输入观测提取频域条件[^src-fgti]——输出端正则与输入端条件化的区别。
- [[frequency-aware-residual-representation|FR（HyperD）]]：交通预测中的 HyperD 也将高频段映射到残差分量[^src-hyperd-hybrid-periodicity-decoupling]；FGTI 在插补任务上给出同方向论断并附分项实验（Table 7）[^src-fgti]，是"高频↔残差"这一跨任务模式的第二处独立证据。
- [[mts-imputation-taxonomy|MTSI 综述]]：综述 Table 1 收录 FGTI（Category 列 generative、架构列 Diffusion, Attention）[^src-mts-imputation-survey]；FGTI 原文同时引用了该综述（其参考文献 [48] 即 arXiv:2402.04059，用于"预测模型可经改造用于插补"的论断）[^src-fgti]。

## 范围与口径

- "最佳插补精度""优于 SOTA 生成式插补模型"均为论文自述（Sec 4.2），实验范围是 3 个数据集、MCAR 10-40% 主表 + 10% MAR/MNAR 补充图、特定基线清单、单位阵邻接设置；未覆盖其后继的少步生成路线（[[loft|LOFT]]、[[costi|CoSTI]] 等以推理效率为主攻方向的工作不在此对比范围内）。
- 对比实验忽略真实缺失、应用研究保留真实缺失，两套评估口径在页内分开陈述。
- 命题 3.1 只给条件熵的严格下降，不含插补误差的直接界；论文以 CRPS 实验作经验支撑（A.5）。

## 关联页面

- [[source-fgti]] — 源文件摘要
- [[csdi]] / [[pristi]] — 条件扩散插补谱系；SSSD（S4 状态空间替代 Transformer 的变体，见 [[csdi]] 后续影响节）为 FGTI 的扩散基线之一
- [[self-supervised-imputation-training]] — 自监督掩码训练，FGTI 与 CSDI 共用的训练范式
- [[crps]] — CRPS，附录 A.5 的分布质量指标
- [[lscd]] — 同为频域条件扩散插补，频谱信号来源不同
- [[loft]] / [[costi]] — 少步生成方向的后继插补工作
- [[frequency-aware-conditioning]] / [[frequency-diffusion]] / [[frequency-based-noise-control]] — 频域介入生成模型的三种位置（条件信号 / 前向噪声谱）
- [[fourier-imputation-loss]] — 输出端频谱正则化的对照
- [[missing-not-at-random]] — FGTI 补测的 MAR/MNAR 机制背景
- [[mts-imputation-taxonomy]] — 综述归类（生成式-Diffusion 类）

[^src-fgti]: [[source-fgti]]
[^src-pristi]: [[source-pristi]]
[^src-lscd]: [[source-lscd]]
[^src-loft]: [[source-loft]]
[^src-2312-01728]: [[source-2312-01728]]
[^src-hyperd-hybrid-periodicity-decoupling]: [[source-hyperd-hybrid-periodicity-decoupling]]
[^src-mts-imputation-survey]: [[source-mts-imputation-survey]]
