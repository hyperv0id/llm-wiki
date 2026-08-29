---
title: "DiffPuter"
type: technique
tags:
  - data-imputation
  - diffusion-models
  - expectation-maximization
  - tabular-data
  - iclr-2025
created: 2026-08-29
last_updated: 2026-08-29
source_count: 2
confidence: medium
status: active
---

# DiffPuter

**DiffPuter** 是 Zhang 等人（Hengrui Zhang、Liancheng Fang，UIC；Qitian Wu，Broad Institute；Philip S. Yu，UIC）发表于 ICLR 2025 的缺失数据插补方法，把扩散模型与 Expectation-Maximization（EM）算法结合：M 步用扩散模型拟合完整数据的联合密度，E 步用混合前向/反向过程的条件采样更新缺失值估计[^src-diffputer]。本地 PDF 为 ICLR 2025 camera-ready 排版（每页页眉 "Published as a conference paper at ICLR 2025"，arXiv:2405.20690v2）[^src-diffputer]。论文将 DiffPuter 定位为首个把扩散生成模型整合进 EM 框架的插补方法（论文自述，第 2 节）[^src-diffputer]。

## 问题设定

论文研究的对象是表格数据（tabular data）的缺失值插补，且强调**训练数据本身含缺失**——区别于"在完整数据上训练、在不完整测试数据上评测"的设定（第 2 节）[^src-diffputer]。作者把生成式插补的两个难点归纳为：1）不完整似然问题——生成模型要估计缺失与观测数据的联合分布，但缺失部分未知，密度估计有内在误差；2）扩散模型直接建模所有维度上的联合分布，天然支持无条件采样，却缺少 VAE/GAN 那样的条件推断灵活性（第 1 节）[^src-diffputer]。

论文还区分 in-sample 与 out-of-sample 插补：前者只填补训练集内的缺失，后者要求模型泛化到未见记录（第 3.1 节）[^src-diffputer]。把缺失值直接当作可学习参数的方法（论文举 MOT、TDM 为例）难以用于 out-of-sample 场景（第 3.1 节）[^src-diffputer]。

## 机制：EM 交替

核心思想是把 $x^{obs}$ 当观测变量、$x^{mis}$ 当隐变量，用 EM 交替优化密度参数 $\theta$ 与缺失值估计（第 3.2 节）[^src-diffputer]。两个步骤的具体对应见 [[em-diffusion-interleaving]]：

- **M 步（密度估计）**：固定当前缺失值填充，训练一个扩散模型学习完整数据的联合密度 $p_\theta(x)=p_\theta(x^{obs},x^{mis})$。扩散采用 VE-SDE 的简化版（$\sigma(t)=t$，附录 C），去噪网络为五层 MLP（附录 D.4）。论文通过 Remark 2（引 Song et al. 2021a 的推论）论证 score matching 损失是真实数据负对数似然的上界，因此 M 步近似最大似然估计（第 4.1 节）[^src-diffputer]。
- **E 步（条件插补）**：固定扩散模型，对观测维执行前向加噪、对缺失维执行反向去噪，再按掩码合并（式 5-7；该混合方式引 Lugmayr et al. 2022 的 RePaint）[^src-diffputer]。Theorem 1 证明当步长 $\Delta t\to 0$ 时所得样本精确服从条件分布 $p_\theta(x\mid x^{obs})$（证明在附录 B.1）；对 $N$ 个样本取均值即缺失值的 Expected A Posteriori（EAP）估计，默认 $N=10$（默认值出自第 5.3 节与附录 D.4）[^src-diffputer]。

实现层面：离散变量经 one-hot 编码转为连续，各列标准化；缺失项用观测列均值初始化（标准化后等价于 0）；M 步与 E 步迭代 K 次，最后一次 E 步的输出作为最终插补结果（第 4.3 节与图 1）[^src-diffputer]。out-of-sample 插补直接用训练好的 score network 执行一次 E 步（第 4.3 节）[^src-diffputer]。

## 实验结果（作者报告）

**设置**：9 个表格数据集——连续特征 5 个（California、Letter、Gesture、Magic、Bean），混合特征 4 个（Adult、Default、Shoppers、News）（第 5.1 节）[^src-diffputer]。注意论文内部口径不一：摘要写 "ten diverse datasets"、第 1 节写 "nine benchmark tabular datasets"、第 5.1 节写 "ten public real-world datasets" 但只列出 9 个、图 2 题注写 "all the nine datasets"、附录 D.2 也写 "ten real-world datasets" 且称 "five datasets of both continuous and discrete features" 却只列 Adult/Default/Shoppers/News 4 个，按原文分别记录[^src-diffputer]。缺失机制按 Rubin 分类研究 MCAR/MAR/MNAR（沿用 Muzellec et al. 2020 与 Zhao et al. 2023 的协议，见 [[missing-not-at-random]]），主结果为 MCAR、缺失率 30%、10 个掩码取均值±标准差；70%/30% 训练/测试划分，连续列用标准化后的 MAE/RMSE，离散列用 Accuracy（第 5.1 节）[^src-diffputer]。基线在摘要中计为 17 个方法、第 5.1 节列 16 个（TDM、MOT、GRAPE、IGRM、EM、MICE、MIRACLE、SoftImpute、MissForest、MIWAE、GAIN、MCFlow、MissDiff、TabCSDI、ReMasker、HyperImpute）、图 2 题注写 "17 baselines"、表 1 题注写 "19 imputation methods"（额外计入 Mean/Median/MF 与 KNN 统计基线），同样按原文分别记录[^src-diffputer]。

| 实验 | 设置 | 结果（作者报告） |
|------|------|------|
| 连续列 in-sample（图 2） | MCAR，9 数据集 | 相对最有竞争力的基线平均提升 6.94%（MAE）、4.78%（RMSE）；摘要以更笼统口径给出同样数字[^src-diffputer] |
| 离散列 in-sample（表 1） | MCAR，4 数据集 Accuracy（题注写 "five datasets"，实列 4 个） | Adult 70.12 / Default 77.64 / Shoppers 58.82 / News 44.69，平均 62.82，题注称在 19 个方法中排第 1（18 个基线 = 第 5.1 节所列 16 个另加 Mean/Median/MF 与 KNN，19 含 DiffPuter 自身）；判别式 Remasker 平均 62.06 排第 2（表 1）[^src-diffputer] |
| out-of-sample（表 6，附录 E.1） | MCAR，6 数据集 | 题注报告 MAE 提升 13.37%、RMSE 提升 4.43%；表中 Average 行数值为 13.09%/4.60%，题注与表内数值不一致，按原文分别记录。第 5.2 节正文称 IGRM 在 out-of-sample 全部数据集上失败——表 6 中 IGRM 实际给出数值但大幅劣于其余方法、仅 Default 列 OOM，正文与表格口径不一，分别记录；GRAPE 由 in-sample 的较好表现明显退化（第 5.2 节、表 6）[^src-diffputer] |
| MAR / MNAR（表 7-10，附录 E） | 同协议 | 论文自述"在所有设置和几乎所有数据集上"保持优势（第 1 节）；表中个别列仍被基线超过（如 MNAR 下 Adult 列 ReMasker MAE 47.66 低于 DiffPuter 的 48.59，表 9）[^src-diffputer] |
| EM 迭代数（图 3） | 消融 | $k=1$（纯扩散、无迭代精炼）只达次优；4-5 次迭代即收敛到稳定状态[^src-diffputer] |
| 采样数 N（图 4） | 消融 | N 过小性能差且方差大，N≥10 稳定，默认 N=10[^src-diffputer] |
| 扩散采样步数 M（图 5） | 消融 | 默认 M=50；50→20 约省 25% 训练时间、性能约降 3%[^src-diffputer] |
| 极端缺失率（图 6） | 30%→99% | 缺失率接近 99% 时性能接近直接用观测均值插补的水平（论文正文称其为 performance lower bound，图 6 题注称性能被均值初始化所 upper-bound——两种措辞并存，按原文分别记录）——因为首步 E 步用列均值初始化[^src-diffputer] |
| 训练时间（表 2） | RTX 4090 | California 1927.2 s / Adult 2142.9 s，与 SOTA 方法同量级（HyperImpute 1277.3/1806.9 s、Remasker 1320.1/1902.4 s、IGRM 1267.5/3865.1 s），论文报告对应性能提升 8%-25%[^src-diffputer] |
| EM + 其他生成模型（表 3） | 消融 | 论文正文称 EM 与 MIWAE/HIWAE/VAEM/HH-VAEM 组合 "leads to performance improvements"；按表 3 数值，EM+MIWAE 与 EM+VAEM 在全部所列列、EM+HH-VAEM 在 Adult/Shoppers 两列优于各自单模型，但 EM+HIWAE 在 Default/Shoppers/News 三列劣于 HIWAE 单模型（如 Default 0.4314 vs 0.3989）——正文概括与表内数据不一致，按原文分别记录。所有组合均不及 DiffPuter（Adult 列 MAE：DiffPuter 0.3425，组合中最低的 EM+HH-VAEM 0.5402）[^src-diffputer] |

论文在第 5.2 节还报告两个横向观察：生成式方法更擅长插补连续列、判别式方法（Remasker、GRAPE、MOT）更擅长离散列；传统方法（混合高斯假设的 EM、KNN）仍优于许多早期深度生成插补方法[^src-diffputer]。

## 与相关方法的关系

- **MCFlow**（Richardson et al., 2020）：最接近的先例——归一化流做迭代插补；论文的区分点是 MCFlow 以最大似然而非期望恢复缺失值，条件插补靠软正则而非精确条件采样（第 2 节）[^src-diffputer]。
- **TabCSDI / MissDiff**：一步式扩散插补，不考虑训练集缺失导致的密度估计误差（第 2 节）[^src-diffputer]。TabCSDI 在本论文中作为扩散基线之一（第 5.1 节）；论文原文只说其在表格数据上采用条件扩散模型、学习以未掩码观测条目为条件的被掩码条目分布（第 2 节）——"TabCSDI 是 [[csdi|CSDI]] 条件扩散框架的表格适配版"系依其名称与机制所作的 wiki 推断，非本论文原文表述[^src-diffputer]。
- **[[prdim|PRDIM]]**（arXiv 2026，晚于本文）：同样在 EM 框架下结合扩散，但额外训练模式识别器显式建模 $p(M\mid X)$ 处理 MNAR，并采用 hard EM——PRDIM 论文将 DiffPuter 的做法刻画为 soft EM 并作为表格实验基线对比[^src-prdim]。

## 范围与局限

论文未设独立的局限性章节；以下为可从论文观察到的边界：

- 数据类型为表格数据，未评测时空或时序数据集（第 5.1 节）[^src-diffputer]。
- 极端缺失率下性能退化为接近均值插补的水平（图 6；论文正文称其为 performance lower bound、图 6 题注称性能被均值初始化所 upper-bound）[^src-diffputer]。
- 训练需在扩散模型训练与条件采样之间交替，总时长高于 MOT/TDM 等基线（表 2）；论文以轻量 MLP 去噪器与 M=50 缓解（第 5.3 节）[^src-diffputer]。
- E 步条件采样依赖扩散过程保持特征位置与维度这一性质（第 4.2 节），适用于表格这类维度固定的数据表示[^src-diffputer]。

## 关联页面

- [[source-diffputer]] — 源文件摘要
- [[em-diffusion-interleaving]] — EM 与扩散交替的核心机制（M 步 MLE / E 步 EAP）
- [[csdi]] — CSDI，条件扩散插补；表格方向的 TabCSDI 是本文的扩散基线（名称谱系说明见正文）
- [[prdim]] — PRDIM，EM+扩散路线的后续（hard EM + 模式识别器，处理 MNAR）
- [[missing-not-at-random]] — MCAR/MAR/MNAR 缺失机制，本文的评测协议来源
- [[score-based-sde]] — M 步所用 VE-SDE 扩散框架的理论来源
- [[nuwats]] / [[t1]] — 其余时序/表格插补方法（对照）

[^src-diffputer]: [[source-diffputer]]
[^src-prdim]: [[source-prdim]]
