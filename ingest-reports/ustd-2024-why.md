# Ingest 报告：USTD (SIGSPATIAL 2024)

## 创建
- wiki/source-ustd.md — WHY：首次将扩散模型统一用于时空预测+插值，是 ST diffusion 路线从"打不过 deterministic"到"超越 deterministic"的关键转折论文
- wiki/ustd.md — WHY：USTD 实体页面，记录解耦预训练 + 任务特定 denoiser 两阶段框架，包含完整架构、实验结果、消融分析

## 修改
- wiki/diffstg.md — WHY：在"后续影响"和"关联页面"添加 [[ustd]] 引用；USTD 直接继承了 DiffSTG 的"用扩散做 STG 预测"范式，并通过解耦训练解决了 DiffSTG 未能超越 deterministic 的核心问题
- wiki/specstg.md — WHY：在"关联页面"添加 [[ustd]] 引用；USTD 和 SpecSTG 共享"C/SDI→PriSTI→USTD/SpecSTG"的演化谱系
- wiki/traffic-forecasting.md — WHY：在"Probabilistic/Diffusion-Based"节添加 USTD 条目，USTD 是首个在交通预测上全面超越确定性 baselines 的扩散方法
- wiki/generative-time-series-forecasting.md — WHY：添加 USTD 为扩散方法条目和关联页面，USTD 提供了"预训练+条件扩散"的生成预测新范式
- wiki/spatio-temporal-foundation-model.md — WHY：在"Related Pages"添加 USTD；USTD 的任务统一路线与 foundation model 路线互补
- wiki/uniflow.md — WHY：在"Connection"添加 USTD；USTD 的 GSM 预训练策略可迁移用于 foundation model 编码器训练
- wiki/index.md — WHY：添加 source-ustd 和 ustd 条目
- wiki/log.md — WHY：追加 ingest 记录

## 新建交叉链接
- [[ustd]] ↔ [[diffstg]]
- [[ustd]] ↔ [[specstg]]
- [[ustd]] ↔ [[traffic-forecasting]]
- [[ustd]] ↔ [[generative-time-series-forecasting]]
- [[ustd]] ↔ [[spatio-temporal-foundation-model]]
- [[ustd]] ↔ [[uniflow]]
