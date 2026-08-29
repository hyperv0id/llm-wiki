---
title: "交通插补统一评测基准（Guo et al. 评测口径）"
type: concept
tags:
  - spatiotemporal-imputation
  - benchmark
  - traffic
  - missing-patterns
  - experimental-evaluation
created: 2026-08-29
last_updated: 2026-08-29
source_count: 1
confidence: medium
status: active
---

# 交通插补统一评测基准（Guo et al. 评测口径）

本页记录 Guo、Wei 等人（arXiv:2412.04733v2）对时空交通数据插补模型的统一实验评测[^src-guo-imputation-evaluation]。与单篇方法论文的"自评口径"不同，该评测将 11 个模型放在同一数据管线、同一缺失场景与统一超参搜索下复现比较；本 wiki 中被其收录的方法页面（如 [[pristi]]、[[imputeformer]]、[[std-plm]]）引用的"评测口径"排名均以本页协议为准，且评测复现数字与各方法原文数字分立、不可混用[^src-guo-imputation-evaluation]。

## 评测协议

- **缺失场景构造**：4 种缺失模式（[[traffic-missing-patterns|SRTR/SRTC/SCTR/SCTC]]）× 5 个缺失率（10%/30%/50%/70%/90%）= 20 个场景[^src-guo-imputation-evaluation]。
- **数据集**：PEMS04（307 传感器，5 分钟，2018-01/02，流量）、PEMS08（170 传感器，5 分钟，2016-07/08，流量）、TW（315 传感器，台湾，5 分钟，2020-11-01 至 11-21，论文新收集）、Seattle（323 检测器，1 小时，2015 全年，速度）[^src-guo-imputation-evaluation]。数据集相关性分析显示 PEMS04/08 时空相关性显著强于 TW 与 Seattle（日间 DTW 相似度 ≥80% 的占比，顺序 PEMS04/PEMS08/TW/Seattle——时间维：95.11%/96.47%/7.30%/4.64%；空间维：62.31%/72.62%/8.29%/1.30%）[^src-guo-imputation-evaluation]。
- **模型构造**：沿用官方代码、仅调整输入输出适配交通数据，对每个模型做大规模网格搜索调参，统一训练集训练、验证集早停[^src-guo-imputation-evaluation]。
- **评测指标**：有效性用 RMSE/MAE/MAPE，只统计缺失位置；效率比较训练时间、推理时间与内存；鲁棒性按测试样本方差将时段分为挑战期（top 25%）与稳定期（bottom 25%）分别报告[^src-guo-imputation-evaluation]。
- 评测流程（场景构造 → 预处理 → 模型构造 → 评估）代码公开于 github.com/wtl52656/imputation-benchmark[^src-guo-imputation-evaluation]。

## 被评测模型

11 个模型：BRITS、E2GAN、mTAN（时间依赖为主）；IGNNK（时空 kriging）、LATC（张量补全）、PriSTI（扩散）、GCASTN（生成-对比）、AGCRN 与 ASTGNN（预测模型经 masked SSL 改造为插补）、ImputeFormer（低秩 Transformer）、STD-PLM（PLM 统一预测+插补）[^src-guo-imputation-evaluation]。图与表中另含 LAST（末次观测值）朴素基线[^src-guo-imputation-evaluation]。计数不一致注记：摘要与贡献写 11，正文 Sec. IV 与 V.B 三处写 "10"（"select the following 10 models"、"10 recently proposed sequence imputation and prediction models"、"10 baselines"），实际列出 11 个[^src-guo-imputation-evaluation]。[[grin|GRIN]] 与 [[csdi|CSDI]] 在综述正文中被引为 STGI/TSPI 类代表工作，但未进入这 11 个评测模型清单[^src-guo-imputation-evaluation]。

## 主要结论（均为该评测在上述协议下的复现报告）

- **缺失率**（四数据集误差曲线，Fig 7-10）：缺失率低于 0.5 时各模型误差变化相对稳定；达到 0.7 与 0.9 时所有模型显著恶化，评测者归因于高缺失破坏数据内在时空关系[^src-guo-imputation-evaluation]。
- **总体排序**：ImputeFormer、GCASTN、BRITS、STD-PLM 为 top-4 深度模型（评测者对四数据集总体表现的归纳），共同特征是对缺失数据有专门设计；按进入 top-1/2/3 的次数计（Fig 13），BRITS、GCASTN、PriSTI、ImputeFormer、LATC 最稳健，评测者将共同点归因于先验知识引入——BRITS/GCASTN/PriSTI 用时延（time delay）机制，ImputeFormer/LATC 用低秩结构，并推测提取额外先验是可行方向（作者自述为猜测）[^src-guo-imputation-evaluation]。
- **数据集差异**（低/高缺失率分组雷达图，Fig 11/12）：PEMS04/08 上 ImputeFormer 在 SR- 模式最好、LATC 在 SC- 模式最好（两者共享张量分解思路），GCASTN 与 STD-PLM 相当；Seattle 上 BRITS 一致最优、GCASTN 第二（评测者归因于速度序列波动小，简单 RNN+时延即可有效建模，更复杂模型反而可能退化）；TW 上 IGNNK 在多数模式（尤其 TC-）最好，但 SCTR 下表现差、该模式下 LATC 最好，SCTC 下依赖复杂时空关系的模型倾向于表现更差[^src-guo-imputation-evaluation]。
- **挑战期与稳定期**（PEMS04、SRTR、0.5，Table III）：挑战期误差显著高于稳定期；GCASTN（挑战期 MAE 22.77/RMSE 34.96/MAPE 8.79，三指标均为表内最低）、STD-PLM、ImputeFormer 表现最好；只建模时间依赖的 BRITS/E2GAN 落后[^src-guo-imputation-evaluation]。
- **效率**（PEMS04、SRTR、0.5，Table IV）：LATC 内存最小（98MB）且无推理阶段；正文称其训练时间亦最小，但其 Table IV 中 E2GAN 训练 312.42s 低于 LATC 829.00s（正文与表格不一致，如实分立）；PriSTI 推理 8553.77s 显著长于其余模型（评测者归因于扩散迭代去噪与多次采样估计不确定性）；STD-PLM 内存最大（8744MB）、GCASTN 训练最久（138565.63s）；正文称深度方法中 E2GAN 推理最快、BRITS 次快，但表中 IGNNK 推理 1.59s 低于 BRITS 17.46s（同样如实分立）[^src-guo-imputation-evaluation]。
- **模型选择建议**（Sec V.F.2，评测者口径）：时间连续（temporal continuous）缺失模式选 GCASTN、空间连续（spatial continuous）缺失模式选 BRITS（论文未逐一点名对应 SRTR/SRTC/SCTR/SCTC 中的哪一个）；内存受限选张量补全（LATC）；实时推理选 TC 或 E2GAN/BRITS；PriSTI、ImputeFormer、IGNNK 在性能与内存间平衡较好[^src-guo-imputation-evaluation]。
- **预测与插补的关系**（Sec V.F.3）：ASTGNN、AGCRN 经 masked SSL 改造后在低缺失率下与专用插补模型可比，缺失率升高后失效——评测者归因于预测模型不区分缺失值与观测值，并推测合适的缺失指示或初始化机制有望统一两类任务（作者自述为推测）[^src-guo-imputation-evaluation]。

## 覆盖范围

评测模型均为该 v2（2025-10）之前的工作，未收录 [[fence|FENCE]]、[[loft|LOFT]]、[[mtsci|MTSCI]]、[[diffputer|DiffPuter]]、[[costi|CoSTI]]、[[rdpi|RDPI]]、[[lcr|LCR]]、[[fgti|FGTI]]、[[giflow|GiFlow]] 等 2024–2026 方法；这些方法在 [[loft]]、[[fence]] 等论文各自协议下的对比数字与本评测口径分立。与 wiki 其他评测类页面分工：[[probts]] 面向多变量时序点/分布预测、[[st-ood]] 面向时空预测的分布外泛化、[[mts-imputation-taxonomy]] 为无实验的综述归类，本页是唯一以交通插补为对象并附复现实验的评测记录[^src-guo-imputation-evaluation]。

## 关联页面

- [[source-guo-imputation-evaluation]] — 源摘要与版本核实
- [[traffic-missing-patterns]] — 评测所用的缺失模式四分类及其构造协议
- [[pristi]] · [[imputeformer]] · [[std-plm]] — 被评测且有 wiki 页面的方法（评测口径已回填）
- [[message-passing-imputation]] — GRIN 一系的消息传递插补（综述提及、未入评测清单）
- [[missing-not-at-random]] — Rubin 缺失机制分类，与本分类维度不同

[^src-guo-imputation-evaluation]: [[source-guo-imputation-evaluation]]
